"""
Multi-Speaker Player
Play one song on multiple Bluetooth speakers at the same time.
Run: python app.py
"""
import os
import queue
import subprocess
import tempfile
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

# ── Shared state ──────────────────────────────────────────────────────────────
_uploaded_path = None   # path to uploaded audio file (file mode)
_is_stream     = False  # True = source is a live YouTube stream
_stream_url    = None   # direct audio URL extracted by yt-dlp
_stream_title  = None   # video title (display only)
_ffmpeg_proc   = None   # running FFmpeg subprocess (stream mode)
_reader_thread = None   # thread that feeds speaker queues (stream mode)

_streams      = []      # active sd.OutputStream objects
_streams_lock = threading.Lock()
_volume       = 1.0     # 0.0–1.0; written by /api/volume, read by callbacks

# FFmpeg decodes to this fixed format; we resample per-speaker as needed
_BASE_SR = 48000
_BASE_CH = 2
_CHUNK   = 4096         # frames per reader iteration (~85 ms at 48 kHz)


# ── Device helpers ────────────────────────────────────────────────────────────

def wasapi_devices():
    """Return WASAPI output devices only (avoids MME duplicates)."""
    result = []
    try:
        apis = sd.query_hostapis()
        wasapi_idx = next(
            (i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None
        )
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
    """Read audio file → (float32 ndarray [samples × ch], sample_rate)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        try:
            import miniaudio
        except ImportError as e:
            raise RuntimeError("miniaudio required for MP3: pip install miniaudio") from e
        r = miniaudio.mp3_read_file_f32(path)
        data = np.array(r.samples, dtype=np.float32).reshape(-1, r.nchannels)
        return data, r.sample_rate
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data, sr


def prepare_audio(data, src_sr, dst_sr, src_ch, dst_ch):
    """Convert channel count then resample. Returns float32 [samples × dst_ch]."""
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
    """Return the MME output device ID whose name matches target_name, or None."""
    try:
        mme_idx = next(
            (i for i, a in enumerate(sd.query_hostapis()) if a["name"] == "MME"), None
        )
        if mme_idx is None:
            return None
        for i, dev in enumerate(sd.query_devices()):
            if dev["hostapi"] == mme_idx and dev["max_output_channels"] > 0:
                dn, tn = dev["name"], target_name
                # MME truncates names at ~31 chars; match if either is a prefix of the other
                if dn == tn or tn.startswith(dn) or dn.startswith(tn[:20]):
                    return i
    except Exception as e:
        print(f"  MME lookup error: {e}", flush=True)
    return None


def _open_and_start_stream(did, samplerate, channels, callback_fn, dev_name=""):
    """
    Try open+start as one atomic unit across multiple configurations.
    Order: WASAPI shared-mode configs → MME fallback.

    Bluetooth speakers fail with WdmSyncIoctl on WASAPI start() because the
    A2DP audio session isn't active yet. MME lets Windows activate it automatically.
    """
    # Build WASAPI candidates
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

    # MME fallback candidates (different device ID, same audio hardware)
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
            stream = sd.OutputStream(
                device=target_id,
                samplerate=sr,
                channels=ch,
                dtype="float32",
                callback=callback_fn,
                **kw,
            )
            stream.start()
            api = "MME" if target_id == mme_id else "WASAPI"
            print(f"  [{dev_name}] started via {api} device {target_id} {kw}", flush=True)
            return stream
        except Exception as exc:
            last_exc = exc
            print(f"  [{dev_name}] device {target_id} {kw} FAILED: {exc}", flush=True)
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass

    raise last_exc


def close_all_streams():
    """Kill any active FFmpeg process and stop all audio streams."""
    global _ffmpeg_proc
    if _ffmpeg_proc:
        try:
            _ffmpeg_proc.kill()
        except Exception:
            pass
        _ffmpeg_proc = None
    with _streams_lock:
        for s in _streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        _streams.clear()


# ── Streaming playback (YouTube) ──────────────────────────────────────────────

def play_stream(selected):
    """
    Open one sd.OutputStream per speaker, backed by a queue.
    A reader thread decodes the YouTube stream via FFmpeg and feeds all queues.
    Playback starts immediately; audio arrives within ~1 second.
    """
    global _streams, _ffmpeg_proc, _reader_thread

    if not _stream_url:
        return jsonify({"error": "No stream loaded."}), 400

    dev_map = {d["id"]: d for d in wasapi_devices()}
    errors, new_streams = [], []

    # Per-speaker state: queue + leftover buffer
    speaker_setup = {}   # did → {"q": Queue, "dst_sr": int, "dst_ch": int}

    for sel in selected:
        did = int(sel["id"])
        delay_ms = max(0, int(sel.get("delay_ms", 0) or 0))

        if did not in dev_map:
            errors.append(f"Device {did} not found (disconnected?).")
            continue

        dv   = dev_map[did]
        dst_sr = int(dv["sample_rate"])
        dst_ch = min(int(dv["channels"]), 2)

        spk_q = queue.Queue(maxsize=500)   # ~43 s buffer at 4096 frames

        # Pre-fill delay as silence so this speaker starts late
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
                        if chunk is None:          # end of stream sentinel
                            n = min(len(arr), frames)
                            outdata[:n] = arr[:n] * vol
                            outdata[n:] = 0
                            raise sd.CallbackStop()
                        arr = np.vstack([arr, chunk]) if len(arr) else chunk
                    except queue.Empty:
                        # buffer underrun: play what we have, rest is silence
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

    # Reader thread: FFmpeg → base PCM → per-speaker queues
    captured_url = _stream_url  # capture before thread starts

    def reader():
        global _ffmpeg_proc
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            for sp in speaker_setup.values():
                sp["q"].put(None)
            return

        _ffmpeg_proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-i", captured_url,
                "-f", "f32le",
                "-ar", str(_BASE_SR),
                "-ac", str(_BASE_CH),
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        bytes_per_chunk = _CHUNK * _BASE_CH * 4  # float32 = 4 bytes

        while True:
            raw = _ffmpeg_proc.stdout.read(bytes_per_chunk)
            if not raw:
                break
            n_frames = len(raw) // (_BASE_CH * 4)
            if n_frames == 0:
                break
            base = np.frombuffer(raw[: n_frames * _BASE_CH * 4],
                                 dtype=np.float32).reshape(-1, _BASE_CH)

            for sp in speaker_setup.values():
                dst_sr, dst_ch = sp["dst_sr"], sp["dst_ch"]
                chunk = (base.copy()
                         if dst_sr == _BASE_SR and dst_ch == _BASE_CH
                         else prepare_audio(base, _BASE_SR, dst_sr, _BASE_CH, dst_ch))
                try:
                    sp["q"].put(chunk, timeout=2)
                except queue.Full:
                    pass  # drop chunk; player too slow (shouldn't happen)

        for sp in speaker_setup.values():
            sp["q"].put(None)   # signal end of stream to each callback

    _reader_thread = threading.Thread(target=reader, daemon=True)
    _reader_thread.start()

    resp = {"message": f'Streaming "{_stream_title}" to {len(new_streams)} speaker(s).'}
    if errors:
        resp["warnings"] = errors
    return jsonify(resp)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    """Always return JSON, even for unhandled exceptions."""
    print(f"Unhandled exception: {e}", flush=True)
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template_string(HTML)


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
        return jsonify({
            "error": f'Format "{ext}" not supported. Use MP3, WAV, FLAC or OGG.'
        }), 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    tmp.close()
    if _uploaded_path and os.path.exists(_uploaded_path):
        try:
            os.unlink(_uploaded_path)
        except Exception:
            pass
    _uploaded_path = tmp.name
    _is_stream = False   # switch back to file mode
    return jsonify({"message": f"Loaded: {f.filename}"})


@app.route("/api/youtube", methods=["POST"])
def api_youtube():
    """
    Extract the direct audio stream URL from YouTube using yt-dlp.
    No download — just resolves the URL so Play can stream directly.
    Takes ~2–5 seconds.
    """
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
        return jsonify({"error": f"Missing package: {exc}. Run: pip install yt-dlp"}), 500

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown")

            # Prefer info['url'] (set when a single format is selected)
            stream_url = info.get("url")
            if not stream_url:
                fmts = [
                    f for f in info.get("formats", [])
                    if f.get("url") and f.get("acodec") not in (None, "none")
                ]
                if not fmts:
                    fmts = [f for f in info.get("formats", []) if f.get("url")]
                if not fmts:
                    raise ValueError("No playable stream URL found in video info.")
                fmts.sort(key=lambda x: x.get("abr") or x.get("tbr") or 0, reverse=True)
                stream_url = fmts[0]["url"]
    except Exception as exc:
        return jsonify({"error": f"Could not load video: {exc}"}), 500

    _is_stream    = True
    _stream_url   = stream_url
    _stream_title = title
    _uploaded_path = None   # clear file mode
    return jsonify({"message": f"Ready: {title}", "title": title})


@app.route("/api/play", methods=["POST"])
def api_play():
    global _streams
    close_all_streams()

    body = request.get_json(silent=True) or {}
    selected = body.get("devices", [])
    if not selected:
        return jsonify({"error": "No speakers selected — tick at least one."}), 400

    # ── Stream mode (YouTube) ─────────────────────────────────────────────────
    if _is_stream:
        return play_stream(selected)

    # ── File mode ─────────────────────────────────────────────────────────────
    if not _uploaded_path or not os.path.exists(_uploaded_path):
        return jsonify({
            "error": "No song loaded — upload a file or paste a YouTube URL first."
        }), 400

    try:
        audio, src_sr = load_audio(_uploaded_path)
    except Exception as exc:
        return jsonify({"error": f"Cannot read audio: {exc}"}), 500

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if len(audio) == 0:
        return jsonify({"error": "Audio file appears to be empty."}), 400

    src_ch  = audio.shape[1]
    dev_map = {d["id"]: d for d in wasapi_devices()}
    errors, new_streams = [], []

    for sel in selected:
        did      = int(sel["id"])
        delay_ms = max(0, int(sel.get("delay_ms", 0) or 0))

        if did not in dev_map:
            errors.append(f"Device {did} not found (maybe disconnected).")
            continue

        dv     = dev_map[did]
        dst_sr = int(dv["sample_rate"])
        dst_ch = min(int(dv["channels"]), 2)

        try:
            buf = prepare_audio(audio, src_sr, dst_sr, src_ch, dst_ch)

            if delay_ms > 0:
                n_pad = int(delay_ms * dst_sr / 1000)
                buf = np.vstack([np.zeros((n_pad, dst_ch), dtype=np.float32), buf])

            state = {"buf": buf, "pos": 0}
            lock  = threading.Lock()

            def make_cb(s, lk, dev_id=did):
                def cb(outdata, frames, _time, status):
                    if status:
                        print(f"[device {dev_id}] {status}", flush=True)
                    with lk:
                        pos    = s["pos"]
                        b      = s["buf"]
                        vol    = _volume
                        remain = len(b) - pos
                        if remain <= 0:
                            outdata[:] = 0
                            raise sd.CallbackStop()
                        n = min(frames, remain)
                        outdata[:n] = b[pos: pos + n] * vol
                        if n < frames:
                            outdata[n:] = 0
                        s["pos"] = pos + n
                        if n < frames:
                            raise sd.CallbackStop()
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
    if errors:
        resp["warnings"] = errors
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


# ── HTML page (embedded) ──────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi-Speaker Player</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
  color: #1a1a1a;
  background: #f0f0f0;
  padding: 28px 16px;
}
.wrap { max-width: 640px; margin: 0 auto; }
h1 { font-size: 19px; font-weight: 700; margin-bottom: 18px; }

.card {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 12px;
}
.card-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #888;
  margin-bottom: 12px;
}

.device-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid #f2f2f2;
}
.device-row:last-child { border-bottom: none; }
.device-label { flex: 1; cursor: pointer; }
.dev-sr { font-size: 12px; color: #bbb; margin-left: 4px; }
.delay-wrap { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
.delay-wrap span { font-size: 12px; color: #777; }
.delay-in {
  width: 58px;
  padding: 3px 5px;
  border: 1px solid #ccc;
  border-radius: 3px;
  text-align: right;
  font-size: 13px;
}
#no-dev { color: #999; font-size: 13px; }

.file-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
#song-tag { font-size: 13px; color: #16a34a; }

.vol-row { display: flex; align-items: center; gap: 10px; }
#vol-range { width: 220px; cursor: pointer; }
#vol-pct { width: 36px; text-align: right; color: #555; }

.btns { display: flex; gap: 10px; }
.btn {
  padding: 9px 22px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
}
.btn:hover { filter: brightness(90%); }
.btn-blue  { background: #2563eb; color: #fff; }
.btn-green { background: #16a34a; color: #fff; }
.btn-red   { background: #dc2626; color: #fff; }
.btn-gray  { background: #6b7280; color: #fff; }

#msg {
  margin-top: 14px;
  padding: 10px 14px;
  border-radius: 5px;
  font-size: 13px;
  line-height: 1.5;
  display: none;
}
#msg.ok   { background: #dcfce7; color: #14532d; }
#msg.err  { background: #fee2e2; color: #7f1d1d; }
#msg.warn { background: #fef9c3; color: #713f12; }

.yt-hint { font-size: 11px; color: #aaa; margin-top: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Multi-Speaker Player</h1>

  <!-- Speakers -->
  <div class="card">
    <div class="card-title">Speakers</div>
    <div style="margin-bottom:10px">
      <button class="btn btn-gray" onclick="loadDevices()">&#8635; Refresh</button>
    </div>
    <div id="device-list"><span id="no-dev">Loading&hellip;</span></div>
  </div>

  <!-- Song file -->
  <div class="card">
    <div class="card-title">
      Song &nbsp;<small style="font-size:11px;color:#bbb;font-weight:400;text-transform:none;letter-spacing:0">MP3 &middot; WAV &middot; FLAC &middot; OGG</small>
    </div>
    <div class="file-row">
      <input type="file" id="file-in" accept=".mp3,.wav,.flac,.ogg">
      <button class="btn btn-blue" onclick="uploadFile()">Upload</button>
      <span id="song-tag"></span>
    </div>
  </div>

  <!-- YouTube stream -->
  <div class="card">
    <div class="card-title">Or stream from YouTube</div>
    <div class="file-row">
      <input type="text" id="yt-url"
             placeholder="https://www.youtube.com/watch?v=..."
             style="flex:1;min-width:0;padding:6px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px;">
      <button class="btn btn-red" onclick="loadYoutube()" id="yt-btn">Load</button>
    </div>
    <div class="yt-hint">
      Resolves stream URL (~2&ndash;5 s), then plays live &mdash; nothing is downloaded to disk.
    </div>
  </div>

  <!-- Volume -->
  <div class="card">
    <div class="card-title">Volume</div>
    <div class="vol-row">
      <input type="range" id="vol-range" min="0" max="100" value="100"
             oninput="onVolume(this.value)">
      <span id="vol-pct">100%</span>
    </div>
  </div>

  <!-- Play / Stop -->
  <div class="btns">
    <button class="btn btn-green" onclick="play()">&#9654; Play</button>
    <button class="btn btn-red"   onclick="doStop()">&#9646;&#9646; Stop</button>
  </div>

  <div id="msg"></div>
</div>

<script>
const $ = id => document.getElementById(id);

function showMsg(html, type) {
  const el = $('msg');
  el.innerHTML = html;
  el.className = type;
  el.style.display = 'block';
}

async function apiFetch(url, opts) {
  try {
    const r = await fetch(url, opts || {});
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      return { ok: r.ok, data: await r.json() };
    }
    // Server sent HTML (unexpected crash) — give a readable message
    return { ok: false, data: { error: `Server error ${r.status} — check the terminal for the full traceback.` } };
  } catch(e) {
    return { ok: false, data: { error: 'Network error: ' + e } };
  }
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Speakers ──────────────────────────────────────────────────────────────────
async function loadDevices() {
  $('device-list').innerHTML = '<span id="no-dev">Loading&hellip;</span>';
  const { ok, data } = await apiFetch('/api/devices');
  const box = $('device-list');
  if (!ok || !Array.isArray(data) || !data.length) {
    box.innerHTML =
      '<span id="no-dev">No WASAPI output devices found. ' +
      'Pair your Bluetooth speakers in Windows Settings &rarr; Bluetooth, then click Refresh.</span>';
    return;
  }
  box.innerHTML = '';
  data.forEach(d => {
    const row = document.createElement('div');
    row.className = 'device-row';
    row.innerHTML =
      `<input type="checkbox" id="cb${d.id}" value="${d.id}">` +
      `<label class="device-label" for="cb${d.id}">${d.name}` +
        `<span class="dev-sr">${d.sample_rate} Hz</span></label>` +
      `<div class="delay-wrap"><span>Delay</span>` +
        `<input class="delay-in" type="number" id="dl${d.id}" value="0" min="0" max="10000">` +
        `<span>ms</span></div>`;
    box.appendChild(row);
  });
}

// ── Upload ────────────────────────────────────────────────────────────────────
async function uploadFile() {
  const inp = $('file-in');
  if (!inp.files.length) { showMsg('Choose a file first.', 'err'); return; }
  const fd = new FormData();
  fd.append('file', inp.files[0]);
  showMsg('Uploading&hellip;', 'ok');
  const { ok, data } = await apiFetch('/api/upload', { method: 'POST', body: fd });
  if (ok) {
    $('song-tag').textContent = '✓ ' + data.message;
    $('yt-url').value = '';
    showMsg(data.message, 'ok');
  } else {
    $('song-tag').textContent = '';
    showMsg(data.error || 'Upload failed.', 'err');
  }
}

// ── YouTube stream ────────────────────────────────────────────────────────────
async function loadYoutube() {
  const url = $('yt-url').value.trim();
  if (!url) { showMsg('Paste a YouTube URL first.', 'err'); return; }

  $('yt-btn').disabled = true;
  showMsg('Resolving YouTube stream&hellip; usually takes 2&ndash;5 seconds.', 'ok');

  const { ok, data } = await apiFetch('/api/youtube', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  });

  $('yt-btn').disabled = false;

  if (ok) {
    $('song-tag').textContent = '&#9654; ' + escHtml(data.title || data.message);
    $('file-in').value = '';
    showMsg(escHtml(data.message) + ' &mdash; press Play to start streaming.', 'ok');
  } else {
    showMsg(escHtml(data.error || 'YouTube load failed.'), 'err');
  }
}

// ── Play ──────────────────────────────────────────────────────────────────────
async function play() {
  const checked = document.querySelectorAll('#device-list input[type=checkbox]:checked');
  if (!checked.length) {
    showMsg('No speakers selected &mdash; tick at least one checkbox.', 'err');
    return;
  }
  const devices = [...checked].map(c => ({
    id: +c.value,
    delay_ms: +($('dl' + c.value).value) || 0
  }));

  showMsg('Starting&hellip;', 'ok');
  const { ok, data } = await apiFetch('/api/play', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ devices })
  });

  if (ok) {
    const w = data.warnings;
    const extra = w && w.length
      ? '<br><small>&#9888; ' + w.map(escHtml).join('<br>') + '</small>' : '';
    showMsg(escHtml(data.message) + extra, w && w.length ? 'warn' : 'ok');
  } else {
    showMsg(escHtml(data.error || 'Playback failed.'), 'err');
  }
}

// ── Stop ──────────────────────────────────────────────────────────────────────
async function doStop() {
  const { ok, data } = await apiFetch('/api/stop', { method: 'POST' });
  showMsg(escHtml(data.message || (ok ? 'Stopped.' : 'Error.')), ok ? 'ok' : 'err');
}

// ── Volume ────────────────────────────────────────────────────────────────────
let volTimer;
function onVolume(v) {
  $('vol-pct').textContent = v + '%';
  clearTimeout(volTimer);
  volTimer = setTimeout(() => apiFetch('/api/volume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ volume: +v })
  }), 100);
}

loadDevices();
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
        print("  (none found — pair your Bluetooth speakers in Windows Settings first)")
    print()
    print("Open your browser at:  http://localhost:5000")
    print("Press Ctrl+C to stop.")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
