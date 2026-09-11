"""
Multi-Speaker Player
- Single Host: play to local Bluetooth speakers via sounddevice
- Room: stream audio over HTTP so any browser on the network can listen
Run: python app.py
"""
import os
import queue
import random
import string
import subprocess
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ── Single-host state ─────────────────────────────────────────────────────────
_uploaded_path = None
_is_stream     = False
_stream_url    = None
_stream_title  = None
_ffmpeg_proc   = None
_reader_thread = None
_streams       = []
_streams_lock  = threading.Lock()
_volume        = 1.0
_BASE_SR = 48000
_BASE_CH = 2
_CHUNK   = 4096

# ── Room state ────────────────────────────────────────────────────────────────
_rooms        = {}   # code -> {playing, title, member_count, created_at}
_room_queues  = {}   # code -> [Queue, ...]  one per connected streaming client
_room_ffmpeg  = {}   # code -> Popen
_room_reader  = {}   # code -> Thread
_room_lock    = threading.Lock()


# ── Device helpers ────────────────────────────────────────────────────────────

def wasapi_devices():
    result = []
    try:
        apis = sd.query_hostapis()
        wasapi_idx = next((i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None)
        if wasapi_idx is None:
            return result
        for i, dev in enumerate(sd.query_devices()):
            if dev["hostapi"] == wasapi_idx and dev["max_output_channels"] > 0:
                result.append({
                    "id": i,
                    "name": dev["name"],
                    "channels": int(dev["max_output_channels"]),
                    "sample_rate": int(dev["default_samplerate"]),
                })
    except Exception as exc:
        print(f"Device query error: {exc}")
    return result


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        try:
            import miniaudio
        except ImportError as e:
            raise RuntimeError("miniaudio required for MP3") from e
        r = miniaudio.mp3_read_file_f32(path)
        data = np.array(r.samples, dtype=np.float32).reshape(-1, r.nchannels)
        return data, r.sample_rate
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data, sr


def prepare_audio(data, src_sr, dst_sr, src_ch, dst_ch):
    if dst_ch != src_ch:
        if dst_ch == 1:
            data = data.mean(axis=1, keepdims=True)
        elif src_ch == 1:
            data = np.tile(data, (1, dst_ch))
        elif dst_ch < src_ch:
            data = data[:, :dst_ch]
        else:
            reps = -(-dst_ch // src_ch)
            data = np.tile(data, (1, reps))[:, :dst_ch]
    if src_sr != dst_sr:
        n_out = int(round(len(data) * dst_sr / src_sr))
        xi = np.linspace(0, len(data) - 1, n_out)
        xp = np.arange(len(data))
        out = np.empty((n_out, data.shape[1]), dtype=np.float32)
        for c in range(data.shape[1]):
            out[:, c] = np.interp(xi, xp, data[:, c])
        data = out
    return data.astype(np.float32)


def _find_mme_device(target_name):
    try:
        mme_idx = next(
            (i for i, a in enumerate(sd.query_hostapis()) if a["name"] == "MME"), None
        )
        if mme_idx is None:
            return None
        for i, dev in enumerate(sd.query_devices()):
            if dev["hostapi"] == mme_idx and dev["max_output_channels"] > 0:
                dn, tn = dev["name"], target_name
                if dn == tn or tn.startswith(dn) or dn.startswith(tn[:20]):
                    return i
    except Exception as e:
        print(f"  MME lookup error: {e}", flush=True)
    return None


def _open_and_start_stream(did, samplerate, channels, callback_fn, dev_name=""):
    wasapi_cfgs = []
    if hasattr(sd, "WasapiSettings"):
        try:
            ws = sd.WasapiSettings(exclusive=False, auto_convert=True)
        except TypeError:
            ws = sd.WasapiSettings(exclusive=False)
        wasapi_cfgs = [
            (did, dict(extra_settings=ws, latency=0.5,    blocksize=0)),
            (did, dict(extra_settings=ws, latency=0.3,    blocksize=0)),
            (did, dict(extra_settings=ws, latency="high", blocksize=0)),
        ]
    wasapi_cfgs += [
        (did, dict(latency=0.5,    blocksize=0)),
        (did, dict(latency="high", blocksize=0)),
    ]
    mme_cfgs = []
    mme_id = _find_mme_device(dev_name) if dev_name else None
    if mme_id is not None:
        mme_sr = int(sd.query_devices(mme_id)["default_samplerate"])
        mme_ch = min(int(sd.query_devices(mme_id)["max_output_channels"]), channels)
        mme_cfgs = [
            (mme_id, dict(latency=0.5,    blocksize=0)),
            (mme_id, dict(latency="high", blocksize=0)),
            (mme_id, dict(latency="low",  blocksize=0)),
        ]
    else:
        mme_sr, mme_ch = samplerate, channels

    last_exc = None
    for target_id, kw in wasapi_cfgs + mme_cfgs:
        sr = mme_sr if target_id == mme_id else samplerate
        ch = mme_ch if target_id == mme_id else channels
        stream = None
        try:
            stream = sd.OutputStream(device=target_id, samplerate=sr, channels=ch,
                                     dtype="float32", callback=callback_fn, **kw)
            stream.start()
            api = "MME" if target_id == mme_id else "WASAPI"
            print(f"  [{dev_name}] started via {api} device {target_id} {kw}", flush=True)
            return stream
        except Exception as exc:
            last_exc = exc
            print(f"  [{dev_name}] device {target_id} {kw} FAILED: {exc}", flush=True)
            if stream:
                try: stream.close()
                except Exception: pass
    raise last_exc


def close_all_streams():
    global _ffmpeg_proc
    if _ffmpeg_proc:
        try: _ffmpeg_proc.kill()
        except Exception: pass
        _ffmpeg_proc = None
    with _streams_lock:
        for s in _streams:
            try: s.stop(); s.close()
            except Exception: pass
        _streams.clear()


# ── Single-host: YouTube stream playback ──────────────────────────────────────

def play_stream(selected):
    global _streams, _ffmpeg_proc, _reader_thread
    if not _stream_url:
        return jsonify({"error": "No stream loaded."}), 400

    dev_map = {d["id"]: d for d in wasapi_devices()}
    errors, new_streams = [], []
    speaker_setup = {}

    for sel in selected:
        did = int(sel["id"])
        delay_ms = max(0, int(sel.get("delay_ms", 0) or 0))
        if did not in dev_map:
            errors.append(f"Device {did} not found.")
            continue
        dv = dev_map[did]
        dst_sr = int(dv["sample_rate"])
        dst_ch = min(int(dv["channels"]), 2)
        spk_q = queue.Queue(maxsize=500)
        if delay_ms > 0:
            n_pad = int(delay_ms * dst_sr / 1000)
            spk_q.put(np.zeros((n_pad, dst_ch), dtype=np.float32))

        def make_stream_cb(q, ch):
            leftover = [np.empty((0, ch), dtype=np.float32)]
            def cb(outdata, frames, _t, status):
                vol = _volume
                arr = leftover[0]
                leftover[0] = np.empty((0, ch), dtype=np.float32)
                while len(arr) < frames:
                    try:
                        chunk = q.get_nowait()
                        if chunk is None:
                            n = min(len(arr), frames)
                            outdata[:n] = arr[:n] * vol
                            outdata[n:] = 0
                            raise sd.CallbackStop()
                        arr = np.vstack([arr, chunk]) if len(arr) else chunk
                    except queue.Empty:
                        n = len(arr)
                        outdata[:n] = arr * vol
                        outdata[n:] = 0
                        return
                outdata[:] = arr[:frames] * vol
                leftover[0] = arr[frames:]
            return cb

        try:
            stream = _open_and_start_stream(did, dst_sr, dst_ch,
                                            make_stream_cb(spk_q, dst_ch), dv["name"])
            speaker_setup[did] = {"q": spk_q, "dst_sr": dst_sr, "dst_ch": dst_ch}
            new_streams.append(stream)
        except Exception as exc:
            errors.append(f'"{dv["name"]}": {exc}')

    if not new_streams:
        return jsonify({"error": "Could not start any stream. " + " ".join(errors)}), 500

    with _streams_lock:
        _streams.extend(new_streams)

    captured_url = _stream_url

    def reader():
        global _ffmpeg_proc
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            for sp in speaker_setup.values(): sp["q"].put(None)
            return
        _ffmpeg_proc = subprocess.Popen(
            [ffmpeg_exe, "-reconnect", "1", "-reconnect_streamed", "1",
             "-reconnect_delay_max", "5", "-i", captured_url,
             "-f", "f32le", "-ar", str(_BASE_SR), "-ac", str(_BASE_CH), "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        bytes_per_chunk = _CHUNK * _BASE_CH * 4
        while True:
            raw = _ffmpeg_proc.stdout.read(bytes_per_chunk)
            if not raw: break
            n_frames = len(raw) // (_BASE_CH * 4)
            if n_frames == 0: break
            base = np.frombuffer(raw[:n_frames * _BASE_CH * 4],
                                 dtype=np.float32).reshape(-1, _BASE_CH)
            for sp in speaker_setup.values():
                dst_sr, dst_ch = sp["dst_sr"], sp["dst_ch"]
                chunk = (base.copy() if dst_sr == _BASE_SR and dst_ch == _BASE_CH
                         else prepare_audio(base, _BASE_SR, dst_sr, _BASE_CH, dst_ch))
                try: sp["q"].put(chunk, timeout=2)
                except queue.Full: pass
        for sp in speaker_setup.values(): sp["q"].put(None)

    _reader_thread = threading.Thread(target=reader, daemon=True)
    _reader_thread.start()
    resp = {"message": f'Streaming "{_stream_title}" to {len(new_streams)} speaker(s).'}
    if errors: resp["warnings"] = errors
    return jsonify(resp)


# ── Room helpers ──────────────────────────────────────────────────────────────

def _gen_code():
    return "".join(random.choices(string.ascii_uppercase, k=4))


def _get_ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _room_stop_internal(code):
    proc = _room_ffmpeg.pop(code, None)
    if proc:
        try: proc.kill()
        except Exception: pass
    with _room_lock:
        for q in _room_queues.get(code, []):
            try: q.put_nowait(None)
            except Exception: pass
    if code in _rooms:
        _rooms[code]["playing"] = False


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {e}", flush=True)
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template_string(HTML)


# Single-host API ----------------------------------------------------------

@app.route("/api/devices")
def api_devices():
    return jsonify(wasapi_devices())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global _uploaded_path, _is_stream
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received."}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".mp3", ".wav", ".flac", ".ogg"):
        return jsonify({"error": f'Format "{ext}" not supported.'}), 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name); tmp.close()
    if _uploaded_path and os.path.exists(_uploaded_path):
        try: os.unlink(_uploaded_path)
        except Exception: pass
    _uploaded_path = tmp.name
    _is_stream = False
    return jsonify({"message": f"Loaded: {f.filename}"})


@app.route("/api/youtube", methods=["POST"])
def api_youtube():
    global _is_stream, _stream_url, _stream_title, _uploaded_path
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Please enter a valid YouTube URL."}), 400
    try:
        import yt_dlp
    except ImportError as exc:
        return jsonify({"error": f"Missing: {exc}"}), 500
    ydl_opts = {"format": "bestaudio/best", "quiet": True,
                "no_warnings": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown")
            stream_url = info.get("url")
            if not stream_url:
                fmts = [f for f in info.get("formats", [])
                        if f.get("url") and f.get("acodec") not in (None, "none")]
                if not fmts:
                    fmts = [f for f in info.get("formats", []) if f.get("url")]
                if not fmts:
                    raise ValueError("No playable stream URL found.")
                fmts.sort(key=lambda x: x.get("abr") or x.get("tbr") or 0, reverse=True)
                stream_url = fmts[0]["url"]
    except Exception as exc:
        return jsonify({"error": f"Could not load video: {exc}"}), 500
    _is_stream = True; _stream_url = stream_url
    _stream_title = title; _uploaded_path = None
    return jsonify({"message": f"Ready: {title}", "title": title})


@app.route("/api/play", methods=["POST"])
def api_play():
    global _streams
    close_all_streams()
    body = request.get_json(silent=True) or {}
    selected = body.get("devices", [])
    if not selected:
        return jsonify({"error": "No speakers selected."}), 400
    if _is_stream:
        return play_stream(selected)
    if not _uploaded_path or not os.path.exists(_uploaded_path):
        return jsonify({"error": "No song loaded."}), 400
    try:
        audio, src_sr = load_audio(_uploaded_path)
    except Exception as exc:
        return jsonify({"error": f"Cannot read audio: {exc}"}), 500
    if audio.ndim == 1: audio = audio[:, np.newaxis]
    if len(audio) == 0: return jsonify({"error": "Audio file is empty."}), 400
    src_ch = audio.shape[1]
    dev_map = {d["id"]: d for d in wasapi_devices()}
    errors, new_streams = [], []
    for sel in selected:
        did = int(sel["id"])
        delay_ms = max(0, int(sel.get("delay_ms", 0) or 0))
        if did not in dev_map:
            errors.append(f"Device {did} not found."); continue
        dv = dev_map[did]
        dst_sr = int(dv["sample_rate"])
        dst_ch = min(int(dv["channels"]), 2)
        try:
            buf = prepare_audio(audio, src_sr, dst_sr, src_ch, dst_ch)
            if delay_ms > 0:
                n_pad = int(delay_ms * dst_sr / 1000)
                buf = np.vstack([np.zeros((n_pad, dst_ch), dtype=np.float32), buf])
            state = {"buf": buf, "pos": 0}
            lock = threading.Lock()
            def make_cb(s, lk, dev_id=did):
                def cb(outdata, frames, _time, status):
                    with lk:
                        pos = s["pos"]; b = s["buf"]; vol = _volume
                        remain = len(b) - pos
                        if remain <= 0:
                            outdata[:] = 0; raise sd.CallbackStop()
                        n = min(frames, remain)
                        outdata[:n] = b[pos:pos+n] * vol
                        if n < frames: outdata[n:] = 0
                        s["pos"] = pos + n
                        if n < frames: raise sd.CallbackStop()
                return cb
            stream = _open_and_start_stream(did, dst_sr, dst_ch,
                                            make_cb(state, lock), dv["name"])
            new_streams.append(stream)
        except Exception as exc:
            errors.append(f'"{dv["name"]}": {exc}')
    if not new_streams:
        return jsonify({"error": "Could not start any stream. " + " ".join(errors)}), 500
    with _streams_lock:
        _streams.extend(new_streams)
    resp = {"message": f"Playing on {len(new_streams)} speaker(s)."}
    if errors: resp["warnings"] = errors
    return jsonify(resp)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    close_all_streams()
    return jsonify({"message": "Stopped."})


@app.route("/api/volume", methods=["POST"])
def api_volume():
    global _volume
    body = request.get_json(silent=True) or {}
    v = max(0, min(100, int(body.get("volume", 100))))
    _volume = v / 100.0
    return jsonify({"message": f"Volume: {v}%"})


# Room API -----------------------------------------------------------------

@app.route("/api/room/create", methods=["POST"])
def api_room_create():
    code = _gen_code()
    with _room_lock:
        while code in _rooms:
            code = _gen_code()
        _rooms[code] = {"code": code, "playing": False,
                        "title": None, "member_count": 0,
                        "created_at": time.time()}
        _room_queues[code] = []
    print(f"Room created: {code}", flush=True)
    return jsonify({"code": code})


@app.route("/api/room/<code>/state")
def api_room_state(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    with _room_lock:
        mc = len(_room_queues.get(code, []))
    r = dict(_rooms[code])
    r["member_count"] = mc
    return jsonify(r)


@app.route("/api/room/<code>/play", methods=["POST"])
def api_room_play(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    _room_stop_internal(code)

    if _is_stream and _stream_url:
        ffmpeg_input = ["-reconnect", "1", "-reconnect_streamed", "1",
                        "-reconnect_delay_max", "5", "-i", _stream_url]
        title = _stream_title or "YouTube stream"
    elif _uploaded_path and os.path.exists(_uploaded_path):
        ffmpeg_input = ["-i", _uploaded_path]
        title = os.path.basename(_uploaded_path)
    else:
        return jsonify({"error": "No song loaded — upload a file or load YouTube first."}), 400

    try:
        ffmpeg_exe = _get_ffmpeg_exe()
    except Exception as exc:
        return jsonify({"error": f"FFmpeg not available: {exc}"}), 500

    proc = subprocess.Popen(
        [ffmpeg_exe] + ffmpeg_input + [
            "-f", "mp3", "-ab", "128k", "-ar", "44100", "-ac", "2", "pipe:1"
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    _room_ffmpeg[code] = proc
    _rooms[code]["playing"] = True
    _rooms[code]["title"] = title

    def reader():
        try:
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                with _room_lock:
                    queues = list(_room_queues.get(code, []))
                for q in queues:
                    try: q.put_nowait(chunk)
                    except queue.Full: pass
        finally:
            with _room_lock:
                queues = list(_room_queues.get(code, []))
            for q in queues:
                try: q.put_nowait(None)
                except Exception: pass
            if code in _rooms:
                _rooms[code]["playing"] = False

    t = threading.Thread(target=reader, daemon=True)
    _room_reader[code] = t
    t.start()
    return jsonify({"message": f"Streaming '{title}' in room {code}."})


@app.route("/api/room/<code>/stop", methods=["POST"])
def api_room_stop(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    _room_stop_internal(code)
    return jsonify({"message": "Room stopped."})


@app.route("/room/<code>/stream")
def room_stream(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    client_q = queue.Queue(maxsize=300)
    with _room_lock:
        _room_queues.setdefault(code, []).append(client_q)

    def generate():
        try:
            while True:
                try:
                    chunk = client_q.get(timeout=30)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    break
        finally:
            with _room_lock:
                ql = _room_queues.get(code, [])
                if client_q in ql:
                    ql.remove(client_q)

    return Response(
        stream_with_context(generate()),
        mimetype="audio/mpeg",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/join/<code>")
def join_room(code):
    return render_template_string(JOIN_HTML, code=code)


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Speaker Player</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --sidebar-w: 180px;
  --accent: #2563eb;
  --bg: #f3f4f6;
  --card: #fff;
  --border: #e5e7eb;
  --text: #111827;
  --muted: #6b7280;
}
body { font-family: Arial,Helvetica,sans-serif; font-size:14px; color:var(--text);
       background:var(--bg); display:flex; min-height:100vh; }

/* ── Sidebar ── */
.sidebar {
  width: var(--sidebar-w);
  background: #1e293b;
  color: #cbd5e1;
  display: flex;
  flex-direction: column;
  padding: 24px 0;
  flex-shrink: 0;
}
.sidebar-logo {
  font-size: 15px; font-weight: 700; color: #f8fafc;
  padding: 0 20px 24px; border-bottom: 1px solid #334155; margin-bottom: 16px;
  line-height: 1.3;
}
.sidebar-logo small { display:block; font-size:11px; color:#94a3b8; font-weight:400; margin-top:2px; }
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 20px; cursor: pointer;
  font-size: 13px; font-weight: 500;
  border-left: 3px solid transparent;
  transition: background .15s, border-color .15s;
  user-select: none;
}
.nav-item:hover { background: #334155; }
.nav-item.active { background: #1d4ed8; color: #fff; border-left-color: #60a5fa; }
.nav-icon { font-size: 16px; }

/* ── Main content ── */
.main { flex: 1; padding: 28px 24px; overflow-y: auto; }
.tab-panel { display: none; max-width: 620px; }
.tab-panel.active { display: block; }
h2 { font-size: 18px; font-weight: 700; margin-bottom: 18px; }

/* ── Cards ── */
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; margin-bottom: 14px;
}
.card-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; color: var(--muted); margin-bottom: 12px;
}

/* ── Device list ── */
.device-row { display:flex; align-items:center; gap:8px; padding:6px 0;
              border-bottom:1px solid #f2f2f2; }
.device-row:last-child { border-bottom:none; }
.device-label { flex:1; cursor:pointer; }
.dev-sr { font-size:12px; color:#bbb; margin-left:4px; }
.delay-wrap { display:flex; align-items:center; gap:4px; white-space:nowrap; }
.delay-wrap span { font-size:12px; color:var(--muted); }
.delay-in { width:58px; padding:3px 5px; border:1px solid #ccc;
            border-radius:3px; text-align:right; font-size:13px; }

/* ── Inputs ── */
.file-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
#song-tag { font-size:13px; color:#16a34a; }
.text-in {
  flex:1; min-width:0; padding:7px 10px;
  border:1px solid #ccc; border-radius:5px; font-size:13px;
}
.vol-row { display:flex; align-items:center; gap:10px; }
#vol-range { width:220px; cursor:pointer; }
#vol-pct { width:36px; text-align:right; color:var(--muted); }

/* ── Buttons ── */
.btns { display:flex; gap:10px; flex-wrap:wrap; }
.btn { padding:9px 20px; border:none; border-radius:6px; cursor:pointer;
       font-size:14px; font-weight:600; }
.btn:hover { filter:brightness(90%); }
.btn:disabled { opacity:.5; cursor:not-allowed; }
.btn-blue  { background:#2563eb; color:#fff; }
.btn-green { background:#16a34a; color:#fff; }
.btn-red   { background:#dc2626; color:#fff; }
.btn-gray  { background:#6b7280; color:#fff; }
.btn-slate { background:#334155; color:#fff; }

/* ── Messages ── */
.msg { margin-top:14px; padding:10px 14px; border-radius:6px;
       font-size:13px; line-height:1.5; display:none; }
.msg.ok   { background:#dcfce7; color:#14532d; }
.msg.err  { background:#fee2e2; color:#7f1d1d; }
.msg.warn { background:#fef9c3; color:#713f12; }

/* ── Room tab specific ── */
.room-code-box {
  font-size: 36px; font-weight: 800; letter-spacing: .2em;
  color: var(--accent); background: #eff6ff; border: 2px dashed #bfdbfe;
  border-radius: 8px; padding: 16px 24px; text-align: center;
  margin: 10px 0;
}
.share-link { font-size: 12px; color: var(--muted); word-break: break-all;
              background: #f9fafb; border: 1px solid var(--border);
              border-radius: 4px; padding: 6px 10px; margin-top: 6px; }
.member-badge {
  display: inline-block; background: #dbeafe; color: #1e40af;
  border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600;
}
.hint { font-size: 12px; color: var(--muted); margin-top: 6px; }
.section-sep { border: none; border-top: 1px solid var(--border); margin: 14px 0; }

/* ── Responsive ── */
@media (max-width: 560px) {
  .sidebar { width: 56px; }
  .sidebar-logo, .nav-label { display: none; }
  .nav-item { justify-content: center; padding: 12px; }
  .main { padding: 16px 12px; }
}
</style>
</head>
<body>

<!-- Sidebar -->
<nav class="sidebar">
  <div class="sidebar-logo">
    Speaker<br>Player
    <small>Multi-output audio</small>
  </div>
  <div class="nav-item active" onclick="switchTab('host', this)">
    <span class="nav-icon">&#128266;</span>
    <span class="nav-label">Single Host</span>
  </div>
  <div class="nav-item" onclick="switchTab('room', this)">
    <span class="nav-icon">&#127968;</span>
    <span class="nav-label">Room</span>
  </div>
</nav>

<!-- Main -->
<main class="main">

  <!-- ══ Single Host tab ══ -->
  <div class="tab-panel active" id="tab-host">
    <h2>Single Host</h2>

    <div class="card">
      <div class="card-title">Speakers</div>
      <div style="margin-bottom:10px">
        <button class="btn btn-gray" onclick="loadDevices()">&#8635; Refresh</button>
      </div>
      <div id="device-list"><span style="color:#999">Loading&hellip;</span></div>
    </div>

    <div class="card">
      <div class="card-title">Song &nbsp;<small style="font-size:11px;color:#bbb;font-weight:400;text-transform:none;letter-spacing:0">MP3 &middot; WAV &middot; FLAC &middot; OGG</small></div>
      <div class="file-row">
        <input type="file" id="file-in" accept=".mp3,.wav,.flac,.ogg">
        <button class="btn btn-blue" onclick="uploadFile()">Upload</button>
        <span id="song-tag"></span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Or stream from YouTube</div>
      <div class="file-row">
        <input class="text-in" type="text" id="yt-url"
               placeholder="https://www.youtube.com/watch?v=...">
        <button class="btn btn-red" onclick="loadYoutube()" id="yt-btn">Load</button>
      </div>
      <div class="hint">Resolves stream URL (~2&ndash;5 s) &mdash; nothing downloaded.</div>
    </div>

    <div class="card">
      <div class="card-title">Volume</div>
      <div class="vol-row">
        <input type="range" id="vol-range" min="0" max="100" value="100"
               oninput="onVolume(this.value)">
        <span id="vol-pct">100%</span>
      </div>
    </div>

    <div class="btns">
      <button class="btn btn-green" onclick="play()">&#9654; Play</button>
      <button class="btn btn-red"   onclick="doStop()">&#9646;&#9646; Stop</button>
    </div>
    <div class="msg" id="msg-host"></div>
  </div>

  <!-- ══ Room tab ══ -->
  <div class="tab-panel" id="tab-room">
    <h2>Room</h2>

    <!-- Step 1: load a song (same as host tab) -->
    <div class="card">
      <div class="card-title">1 &mdash; Load a song</div>
      <div class="file-row" style="margin-bottom:8px">
        <input type="file" id="r-file-in" accept=".mp3,.wav,.flac,.ogg">
        <button class="btn btn-blue" onclick="rUploadFile()">Upload</button>
        <span id="r-song-tag" style="font-size:13px;color:#16a34a"></span>
      </div>
      <hr class="section-sep">
      <div class="card-title" style="margin-bottom:8px">Or YouTube</div>
      <div class="file-row">
        <input class="text-in" type="text" id="r-yt-url"
               placeholder="https://www.youtube.com/watch?v=...">
        <button class="btn btn-red" onclick="rLoadYoutube()" id="r-yt-btn">Load</button>
      </div>
      <div class="hint">Same source is used for both Room and Single Host.</div>
    </div>

    <!-- Step 2: create room -->
    <div class="card">
      <div class="card-title">2 &mdash; Create a room</div>
      <div id="room-create-area">
        <button class="btn btn-slate" onclick="createRoom()">&#43; Create Room</button>
        <div class="hint" style="margin-top:8px">
          A 4-letter code is generated. Share the link with anyone on the same network.
        </div>
      </div>

      <div id="room-active-area" style="display:none">
        <div class="room-code-box" id="room-code-display">????</div>
        <div class="share-link" id="room-share-link"></div>
        <div style="margin-top:10px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
          <span>Listeners: <span class="member-badge" id="room-members">0</span></span>
          <span id="room-status-badge" style="font-size:12px;color:var(--muted)">Stopped</span>
        </div>
      </div>
    </div>

    <!-- Step 3: controls -->
    <div class="card" id="room-controls-card" style="display:none">
      <div class="card-title">3 &mdash; Controls</div>
      <div class="btns">
        <button class="btn btn-green" onclick="roomPlay()">&#9654; Start Stream</button>
        <button class="btn btn-red"   onclick="roomStop()">&#9646;&#9646; Stop</button>
      </div>
    </div>

    <div class="msg" id="msg-room"></div>
  </div>

</main>

<script>
const $ = id => document.getElementById(id);
let _roomCode = null;
let _pollTimer = null;

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(name, el) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  $('tab-' + name).classList.add('active');
  el.classList.add('active');
}

// ── Generic helpers ────────────────────────────────────────────────────────
function showMsg(id, html, type) {
  const el = $(id);
  el.innerHTML = html; el.className = 'msg ' + type; el.style.display = 'block';
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
async function apiFetch(url, opts) {
  try {
    const r = await fetch(url, opts || {});
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('application/json')) return { ok: r.ok, data: await r.json() };
    return { ok: false, data: { error: `Server error ${r.status} — check terminal.` } };
  } catch(e) {
    return { ok: false, data: { error: 'Network error: ' + e } };
  }
}

// ── Single Host ────────────────────────────────────────────────────────────
async function loadDevices() {
  $('device-list').innerHTML = '<span style="color:#999">Loading&hellip;</span>';
  const { ok, data } = await apiFetch('/api/devices');
  const box = $('device-list');
  if (!ok || !Array.isArray(data) || !data.length) {
    box.innerHTML = '<span style="color:#999">No WASAPI devices found. Pair speakers then Refresh.</span>';
    return;
  }
  box.innerHTML = '';
  data.forEach(d => {
    const row = document.createElement('div');
    row.className = 'device-row';
    row.innerHTML =
      `<input type="checkbox" id="cb${d.id}" value="${d.id}">` +
      `<label class="device-label" for="cb${d.id}">${escHtml(d.name)}` +
        `<span class="dev-sr">${d.sample_rate} Hz</span></label>` +
      `<div class="delay-wrap"><span>Delay</span>` +
        `<input class="delay-in" type="number" id="dl${d.id}" value="0" min="0" max="10000">` +
        `<span>ms</span></div>`;
    box.appendChild(row);
  });
}
async function uploadFile() {
  const inp = $('file-in');
  if (!inp.files.length) { showMsg('msg-host','Choose a file first.','err'); return; }
  const fd = new FormData(); fd.append('file', inp.files[0]);
  showMsg('msg-host','Uploading&hellip;','ok');
  const { ok, data } = await apiFetch('/api/upload', { method:'POST', body:fd });
  if (ok) { $('song-tag').textContent = '&#10003; ' + data.message; showMsg('msg-host', escHtml(data.message),'ok'); }
  else { $('song-tag').textContent=''; showMsg('msg-host', escHtml(data.error||'Upload failed.'),'err'); }
}
async function loadYoutube() {
  const url = $('yt-url').value.trim();
  if (!url) { showMsg('msg-host','Paste a YouTube URL first.','err'); return; }
  $('yt-btn').disabled = true;
  showMsg('msg-host','Resolving stream&hellip; (~2&ndash;5 s)','ok');
  const { ok, data } = await apiFetch('/api/youtube', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ url })
  });
  $('yt-btn').disabled = false;
  if (ok) { $('song-tag').textContent = data.title || data.message; showMsg('msg-host', escHtml(data.message),'ok'); }
  else { showMsg('msg-host', escHtml(data.error||'YouTube load failed.'),'err'); }
}
async function play() {
  const checked = document.querySelectorAll('#device-list input[type=checkbox]:checked');
  if (!checked.length) { showMsg('msg-host','Select at least one speaker.','err'); return; }
  const devices = [...checked].map(c => ({ id:+c.value, delay_ms:+($('dl'+c.value).value)||0 }));
  showMsg('msg-host','Starting&hellip;','ok');
  const { ok, data } = await apiFetch('/api/play', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ devices })
  });
  if (ok) {
    const w = data.warnings;
    const extra = w&&w.length ? '<br><small>&#9888; '+w.map(escHtml).join('<br>')+'</small>' : '';
    showMsg('msg-host', escHtml(data.message)+extra, w&&w.length?'warn':'ok');
  } else { showMsg('msg-host', escHtml(data.error||'Playback failed.'),'err'); }
}
async function doStop() {
  const { ok, data } = await apiFetch('/api/stop', { method:'POST' });
  showMsg('msg-host', escHtml(data.message||(ok?'Stopped.':'Error.')), ok?'ok':'err');
}
let volTimer;
function onVolume(v) {
  $('vol-pct').textContent = v+'%';
  clearTimeout(volTimer);
  volTimer = setTimeout(() => apiFetch('/api/volume', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ volume:+v })
  }), 100);
}

// ── Room tab ───────────────────────────────────────────────────────────────
async function rUploadFile() {
  const inp = $('r-file-in');
  if (!inp.files.length) { showMsg('msg-room','Choose a file first.','err'); return; }
  const fd = new FormData(); fd.append('file', inp.files[0]);
  showMsg('msg-room','Uploading&hellip;','ok');
  const { ok, data } = await apiFetch('/api/upload', { method:'POST', body:fd });
  if (ok) { $('r-song-tag').textContent = data.message; showMsg('msg-room', escHtml(data.message),'ok'); }
  else { showMsg('msg-room', escHtml(data.error||'Upload failed.'),'err'); }
}
async function rLoadYoutube() {
  const url = $('r-yt-url').value.trim();
  if (!url) { showMsg('msg-room','Paste a YouTube URL first.','err'); return; }
  $('r-yt-btn').disabled = true;
  showMsg('msg-room','Resolving stream&hellip;','ok');
  const { ok, data } = await apiFetch('/api/youtube', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ url })
  });
  $('r-yt-btn').disabled = false;
  if (ok) { $('r-song-tag').textContent = data.title || data.message; showMsg('msg-room', escHtml(data.message),'ok'); }
  else { showMsg('msg-room', escHtml(data.error||'Failed.'),'err'); }
}
async function createRoom() {
  const { ok, data } = await apiFetch('/api/room/create', { method:'POST' });
  if (!ok) { showMsg('msg-room', escHtml(data.error||'Failed.'),'err'); return; }
  _roomCode = data.code;
  $('room-code-display').textContent = _roomCode;
  const link = location.protocol + '//' + location.hostname + ':' + location.port + '/join/' + _roomCode;
  $('room-share-link').textContent = link;
  $('room-create-area').style.display = 'none';
  $('room-active-area').style.display = 'block';
  $('room-controls-card').style.display = 'block';
  showMsg('msg-room', 'Room created! Share the link above with your listeners.', 'ok');
  startPolling();
}
async function roomPlay() {
  if (!_roomCode) return;
  showMsg('msg-room','Starting room stream&hellip;','ok');
  const { ok, data } = await apiFetch('/api/room/'+_roomCode+'/play', { method:'POST' });
  showMsg('msg-room', escHtml(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
}
async function roomStop() {
  if (!_roomCode) return;
  const { ok, data } = await apiFetch('/api/room/'+_roomCode+'/stop', { method:'POST' });
  showMsg('msg-room', escHtml(data.message||(ok?'Stopped.':'Error.')), ok?'ok':'err');
}
function startPolling() {
  clearInterval(_pollTimer);
  _pollTimer = setInterval(async () => {
    if (!_roomCode) return;
    const { ok, data } = await apiFetch('/api/room/'+_roomCode+'/state');
    if (!ok) return;
    $('room-members').textContent = data.member_count || 0;
    $('room-status-badge').textContent = data.playing
      ? ('Playing: ' + (data.title||'')) : 'Stopped';
    $('room-status-badge').style.color = data.playing ? '#16a34a' : '#6b7280';
  }, 2000);
}

loadDevices();
</script>
</body>
</html>"""


JOIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Room {{ code }}</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Arial,Helvetica,sans-serif; background: #0f172a; color: #f1f5f9;
       display: flex; align-items: center; justify-content: center;
       min-height: 100vh; padding: 24px; }
.card { background: #1e293b; border-radius: 16px; padding: 36px 32px;
        max-width: 420px; width: 100%; text-align: center; }
.badge { font-size: 48px; font-weight: 900; letter-spacing: .15em;
         color: #60a5fa; margin-bottom: 6px; }
.label { font-size: 13px; color: #94a3b8; margin-bottom: 24px; }
.now-playing { font-size: 15px; color: #f8fafc; margin-bottom: 20px;
               min-height: 22px; font-weight: 600; }
audio { width: 100%; border-radius: 8px; margin-bottom: 20px; }
.status { font-size: 12px; color: #64748b; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
       background: #ef4444; margin-right: 6px; }
.dot.live { background: #22c55e; animation: pulse 1.2s ease infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body>
<div class="card">
  <div class="badge">{{ code }}</div>
  <div class="label">Room &mdash; join &amp; listen</div>
  <div class="now-playing" id="now-playing">Waiting for host&hellip;</div>
  <audio id="player" controls autoplay></audio>
  <div class="status">
    <span class="dot" id="dot"></span>
    <span id="status-text">Connecting&hellip;</span>
  </div>
</div>
<script>
const code = "{{ code }}";
const player = document.getElementById('player');
const dot = document.getElementById('dot');
const statusEl = document.getElementById('status-text');
const nowPlaying = document.getElementById('now-playing');
let playing = false;

async function poll() {
  try {
    const r = await fetch('/api/room/' + code + '/state');
    if (!r.ok) { statusEl.textContent = 'Room not found.'; return; }
    const d = await r.json();
    if (d.playing && !playing) {
      playing = true;
      player.src = '/room/' + code + '/stream';
      player.play().catch(()=>{});
      dot.className = 'dot live';
      statusEl.textContent = 'Live';
      nowPlaying.textContent = d.title || 'Now playing';
    } else if (!d.playing && playing) {
      playing = false;
      player.src = '';
      dot.className = 'dot';
      statusEl.textContent = 'Stopped';
      nowPlaying.textContent = 'Waiting for host…';
    }
    if (!d.playing) statusEl.textContent = 'Waiting…';
  } catch(e) {
    statusEl.textContent = 'Error connecting.';
  }
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=== WASAPI Output Devices ===")
    devs = wasapi_devices()
    if devs:
        for d in devs:
            print(f"  [{d['id']:3}]  {d['name']}  ({d['channels']}ch @ {d['sample_rate']} Hz)")
    else:
        print("  (none found)")
    print()
    print("Open: http://localhost:5000")
    print("Press Ctrl+C to stop.")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
