"""
Multi-Speaker Player
Single Host : local Bluetooth speakers via sounddevice
Room        : shared queue + HTTP MP3 stream to any browser on the same WiFi
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
from flask import (Flask, Response, jsonify, render_template_string,
                   request, stream_with_context)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ── Single-host state ──────────────────────────────────────────────────────────
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

# ── Room state ─────────────────────────────────────────────────────────────────
# _rooms[code] = {
#   code, playing, current_idx,
#   queue: [{id, title, source("file"|"youtube"), path, url, added_by}],
#   created_at
# }
_rooms          = {}
_room_clients   = {}   # code -> [queue.Queue, ...]  one per HTTP listener
_room_ffmpeg    = {}   # code -> Popen
_room_readers   = {}   # code -> Thread
_room_lock      = threading.Lock()
_room_no_advance = set()   # codes where auto-advance is suppressed


def _new_id():
    return f"{int(time.time()*1000)}{random.randint(100,999)}"


def _gen_code():
    return "".join(random.choices(string.ascii_uppercase, k=4))


# ── Device helpers ─────────────────────────────────────────────────────────────

def wasapi_devices():
    result = []
    try:
        apis = sd.query_hostapis()
        wasapi_idx = next(
            (i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None)
        if wasapi_idx is None:
            return result
        for i, dev in enumerate(sd.query_devices()):
            if dev["hostapi"] == wasapi_idx and dev["max_output_channels"] > 0:
                result.append({"id": i, "name": dev["name"],
                                "channels": int(dev["max_output_channels"]),
                                "sample_rate": int(dev["default_samplerate"])})
    except Exception as exc:
        print(f"Device query error: {exc}")
    return result


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        try:
            import miniaudio
        except ImportError as e:
            raise RuntimeError("miniaudio required for MP3") from e
        r = miniaudio.mp3_read_file_f32(path)
        return np.array(r.samples, dtype=np.float32).reshape(-1, r.nchannels), r.sample_rate
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
            data = np.tile(data, (1, -(-dst_ch // src_ch)))[:, :dst_ch]
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
            (i for i, a in enumerate(sd.query_hostapis()) if a["name"] == "MME"), None)
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
    wasapi_cfgs += [(did, dict(latency=0.5, blocksize=0)),
                    (did, dict(latency="high", blocksize=0))]
    mme_id = _find_mme_device(dev_name) if dev_name else None
    mme_cfgs = []
    if mme_id is not None:
        mme_sr = int(sd.query_devices(mme_id)["default_samplerate"])
        mme_ch = min(int(sd.query_devices(mme_id)["max_output_channels"]), channels)
        mme_cfgs = [(mme_id, dict(latency=0.5, blocksize=0)),
                    (mme_id, dict(latency="high", blocksize=0)),
                    (mme_id, dict(latency="low",  blocksize=0))]
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
            print(f"  [{dev_name}] started via {api} device {target_id}", flush=True)
            return stream
        except Exception as exc:
            last_exc = exc
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


# ── Single-host YouTube playback ───────────────────────────────────────────────

def play_stream(selected):
    global _streams, _ffmpeg_proc, _reader_thread
    if not _stream_url:
        return jsonify({"error": "No stream loaded."}), 400
    dev_map = {d["id"]: d for d in wasapi_devices()}
    errors, new_streams, speaker_setup = [], [], {}
    for sel in selected:
        did = int(sel["id"])
        delay_ms = max(0, int(sel.get("delay_ms", 0) or 0))
        if did not in dev_map:
            errors.append(f"Device {did} not found."); continue
        dv = dev_map[did]
        dst_sr, dst_ch = int(dv["sample_rate"]), min(int(dv["channels"]), 2)
        spk_q = queue.Queue(maxsize=500)
        if delay_ms > 0:
            spk_q.put(np.zeros((int(delay_ms * dst_sr / 1000), dst_ch), dtype=np.float32))

        def make_stream_cb(q, ch):
            leftover = [np.empty((0, ch), dtype=np.float32)]
            def cb(outdata, frames, _t, status):
                vol = _volume
                arr = leftover[0]; leftover[0] = np.empty((0, ch), dtype=np.float32)
                while len(arr) < frames:
                    try:
                        chunk = q.get_nowait()
                        if chunk is None:
                            n = min(len(arr), frames)
                            outdata[:n] = arr[:n] * vol; outdata[n:] = 0
                            raise sd.CallbackStop()
                        arr = np.vstack([arr, chunk]) if len(arr) else chunk
                    except queue.Empty:
                        n = len(arr); outdata[:n] = arr * vol; outdata[n:] = 0; return
                outdata[:] = arr[:frames] * vol; leftover[0] = arr[frames:]
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
            import imageio_ffmpeg; ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            for sp in speaker_setup.values(): sp["q"].put(None); return
        _ffmpeg_proc = subprocess.Popen(
            [ffmpeg_exe, "-reconnect", "1", "-reconnect_streamed", "1",
             "-reconnect_delay_max", "5", "-i", captured_url,
             "-f", "f32le", "-ar", str(_BASE_SR), "-ac", str(_BASE_CH), "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        bytes_per_chunk = _CHUNK * _BASE_CH * 4
        while True:
            raw = _ffmpeg_proc.stdout.read(bytes_per_chunk)
            if not raw: break
            n_frames = len(raw) // (_BASE_CH * 4)
            if n_frames == 0: break
            base = np.frombuffer(raw[:n_frames * _BASE_CH * 4],
                                 dtype=np.float32).reshape(-1, _BASE_CH)
            for sp in speaker_setup.values():
                chunk = (base.copy() if sp["dst_sr"] == _BASE_SR and sp["dst_ch"] == _BASE_CH
                         else prepare_audio(base, _BASE_SR, sp["dst_sr"], _BASE_CH, sp["dst_ch"]))
                try: sp["q"].put(chunk, timeout=2)
                except queue.Full: pass
        for sp in speaker_setup.values(): sp["q"].put(None)
    _reader_thread = threading.Thread(target=reader, daemon=True)
    _reader_thread.start()
    resp = {"message": f'Streaming "{_stream_title}" to {len(new_streams)} speaker(s).'}
    if errors: resp["warnings"] = errors
    return jsonify(resp)


# ── Room: core ─────────────────────────────────────────────────────────────────

def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _room_broadcast_stop(code):
    """Send None sentinel to all connected HTTP clients for a room."""
    with _room_lock:
        for q in _room_clients.get(code, []):
            try: q.put_nowait(None)
            except Exception: pass


def _room_stop_internal(code):
    """Kill FFmpeg + notify clients. Does NOT auto-advance."""
    _room_no_advance.add(code)
    proc = _room_ffmpeg.pop(code, None)
    if proc:
        try: proc.kill()
        except Exception: pass
    _room_broadcast_stop(code)
    if code in _rooms:
        _rooms[code]["playing"] = False


def _room_start_item(code, idx):
    """Start playing queue item at idx. Called by play route and auto-advance."""
    room = _rooms.get(code)
    if not room:
        return
    q = room["queue"]
    if idx >= len(q):
        room["playing"] = False
        return

    item = q[idx]
    room["current_idx"] = idx
    room["playing"] = True

    # Build FFmpeg input args
    if item["source"] == "youtube":
        # Re-resolve URL fresh at play time — YouTube stream URLs expire quickly
        fresh_url = item["url"]
        yt_page = item.get("yt_page_url")
        if yt_page:
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL({"format": "bestaudio/best", "quiet": True,
                                        "no_warnings": True, "noplaylist": True}) as ydl:
                    info = ydl.extract_info(yt_page, download=False)
                    resolved = info.get("url")
                    if not resolved:
                        fmts = [f for f in info.get("formats", [])
                                if f.get("url") and f.get("acodec") not in (None, "none")]
                        if not fmts:
                            fmts = [f for f in info.get("formats", []) if f.get("url")]
                        if fmts:
                            fmts.sort(key=lambda x: x.get("abr") or x.get("tbr") or 0, reverse=True)
                            resolved = fmts[0]["url"]
                    if resolved:
                        fresh_url = resolved
                        item["url"] = resolved  # update cache
                print(f"[Room {code}] YouTube URL refreshed for: {item['title']}", flush=True)
            except Exception as exc:
                print(f"[Room {code}] URL re-resolve failed, using cached: {exc}", flush=True)
        ffmpeg_in = ["-reconnect", "1", "-reconnect_streamed", "1",
                     "-reconnect_delay_max", "5", "-i", fresh_url]
    else:
        ffmpeg_in = ["-i", item["path"]]

    try:
        exe = _ffmpeg_exe()
    except Exception as exc:
        print(f"[Room {code}] ffmpeg unavailable: {exc}", flush=True)
        room["playing"] = False
        return

    proc = subprocess.Popen(
        [exe] + ffmpeg_in + ["-f", "mp3", "-ab", "128k", "-ar", "44100", "-ac", "2", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    _room_ffmpeg[code] = proc
    print(f"[Room {code}] playing #{idx}: {item['title']}", flush=True)

    def reader():
        try:
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                with _room_lock:
                    clients = list(_room_clients.get(code, []))
                for cq in clients:
                    try: cq.put_nowait(chunk)
                    except queue.Full: pass
        finally:
            # Notify clients this stream ended
            with _room_lock:
                clients = list(_room_clients.get(code, []))
            for cq in clients:
                try: cq.put_nowait(None)
                except Exception: pass
            # Auto-advance unless we were intentionally stopped
            if code not in _room_no_advance:
                _room_advance(code)
            _room_no_advance.discard(code)

    t = threading.Thread(target=reader, daemon=True)
    _room_readers[code] = t
    t.start()


def _room_advance(code):
    """Move to next queue item and start playing."""
    room = _rooms.get(code)
    if not room or not room.get("playing"):
        return
    next_idx = room["current_idx"] + 1
    if next_idx >= len(room["queue"]):
        room["playing"] = False
        print(f"[Room {code}] queue finished.", flush=True)
        return
    _room_start_item(code, next_idx)


# ── Flask error handler ────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {e}", flush=True)
    return jsonify({"error": str(e)}), 500


# ── Main page ──────────────────────────────────────────────────────────────────

@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
def index():
    return render_template_string(HTML)


# ── Single Host API ────────────────────────────────────────────────────────────

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
    try:
        with yt_dlp.YoutubeDL({"format": "bestaudio/best", "quiet": True,
                                "no_warnings": True, "noplaylist": True}) as ydl:
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
    # Store original page URL so play_stream can re-resolve if needed
    app.config["_yt_page_url"] = url
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
        did = int(sel["id"]); delay_ms = max(0, int(sel.get("delay_ms", 0) or 0))
        if did not in dev_map:
            errors.append(f"Device {did} not found."); continue
        dv = dev_map[did]
        dst_sr, dst_ch = int(dv["sample_rate"]), min(int(dv["channels"]), 2)
        try:
            buf = prepare_audio(audio, src_sr, dst_sr, src_ch, dst_ch)
            if delay_ms > 0:
                buf = np.vstack([np.zeros((int(delay_ms * dst_sr / 1000), dst_ch),
                                          dtype=np.float32), buf])
            state = {"buf": buf, "pos": 0}; lock = threading.Lock()
            def make_cb(s, lk, dev_id=did):
                def cb(outdata, frames, _time, status):
                    with lk:
                        pos = s["pos"]; b = s["buf"]; vol = _volume
                        remain = len(b) - pos
                        if remain <= 0: outdata[:] = 0; raise sd.CallbackStop()
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
        return jsonify({"error": "Could not start. " + " ".join(errors)}), 500
    with _streams_lock: _streams.extend(new_streams)
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


# ── Room API ───────────────────────────────────────────────────────────────────

@app.route("/api/room/create", methods=["POST"])
def api_room_create():
    body = request.get_json(silent=True) or {}
    requested = (body.get("code") or "").strip().upper()
    if requested and len(requested) == 4 and requested.isalpha():
        code = requested
    else:
        code = _gen_code()
    with _room_lock:
        if code in _rooms:
            return jsonify({"error": f'Room "{code}" already exists. Choose a different code.'}), 400
        _rooms[code] = {"code": code, "playing": False,
                        "current_idx": 0, "queue": [], "created_at": time.time()}
        _room_clients[code] = []
    print(f"Room created: {code}", flush=True)
    return jsonify({"code": code})


@app.route("/api/room/<code>/state")
def api_room_state(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    with _room_lock:
        mc = len(_room_clients.get(code, []))
    idx = room["current_idx"]
    q = room["queue"]
    current_title = q[idx]["title"] if idx < len(q) else None
    return jsonify({
        "code": code,
        "playing": room["playing"],
        "paused": room.get("paused", False),
        "current_idx": idx,
        "current_title": current_title,
        "member_count": mc,
        "queue": [{"id": it["id"], "title": it["title"],
                   "source": it["source"], "added_by": it["added_by"]} for it in q],
    })


@app.route("/api/room/<code>/queue/add", methods=["POST"])
def api_room_queue_add(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404

    is_host = request.remote_addr in ("127.0.0.1", "::1")

    # File upload
    if "file" in request.files:
        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "No file."}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".mp3", ".wav", ".flac", ".ogg"):
            return jsonify({"error": f'Format "{ext}" not supported.'}), 400
        added_by = "Host" if is_host else (request.form.get("name") or "").strip() or request.remote_addr
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        f.save(tmp.name); tmp.close()
        item = {"id": _new_id(), "title": f.filename,
                "source": "file", "path": tmp.name, "url": None,
                "added_by": added_by}
        _rooms[code]["queue"].append(item)
        return jsonify({"message": f'Added: {f.filename}', "id": item["id"]})

    # YouTube URL
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Provide a file or YouTube URL."}), 400
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Not a valid YouTube URL."}), 400
    added_by = "Host" if is_host else (body.get("name") or "").strip() or request.remote_addr
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"format": "bestaudio/best", "quiet": True,
                                "no_warnings": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "YouTube track")
            stream_url = info.get("url")
            if not stream_url:
                fmts = [f for f in info.get("formats", [])
                        if f.get("url") and f.get("acodec") not in (None, "none")]
                if not fmts:
                    fmts = [f for f in info.get("formats", []) if f.get("url")]
                fmts.sort(key=lambda x: x.get("abr") or x.get("tbr") or 0, reverse=True)
                stream_url = fmts[0]["url"]
    except Exception as exc:
        return jsonify({"error": f"Could not load YouTube: {exc}"}), 500
    item = {"id": _new_id(), "title": title,
            "source": "youtube", "path": None,
            "url": stream_url,        # cached stream URL (may expire)
            "yt_page_url": url,       # original YouTube page URL for re-resolving
            "added_by": added_by}
    _rooms[code]["queue"].append(item)
    return jsonify({"message": f'Added: {title}', "id": item["id"]})


@app.route("/api/room/<code>/queue/remove/<item_id>", methods=["POST"])
def api_room_queue_remove(code, item_id):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    before = len(room["queue"])
    new_q = [it for it in room["queue"] if it["id"] != item_id]
    removed = before - len(new_q)
    if removed == 0:
        return jsonify({"error": "Item not found"}), 404
    # Adjust current_idx if needed
    removed_idx = next((i for i, it in enumerate(room["queue"]) if it["id"] == item_id), -1)
    room["queue"] = new_q
    if removed_idx < room["current_idx"]:
        room["current_idx"] = max(0, room["current_idx"] - 1)
    return jsonify({"message": "Removed."})


@app.route("/api/room/<code>/play", methods=["POST"])
def api_room_play(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    if not room["queue"]:
        return jsonify({"error": "Queue is empty — add a song first."}), 400
    _room_stop_internal(code)
    _room_no_advance.discard(code)
    idx = room["current_idx"]
    if idx >= len(room["queue"]):
        idx = 0; room["current_idx"] = 0
    _room_start_item(code, idx)
    title = room["queue"][idx]["title"]
    return jsonify({"message": f'Playing: {title}'})


@app.route("/api/room/<code>/jump/<item_id>", methods=["POST"])
def api_room_jump(code, item_id):
    """Host forces everyone onto a specific song immediately."""
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    idx = next((i for i, it in enumerate(room["queue"]) if it["id"] == item_id), None)
    if idx is None:
        return jsonify({"error": "Song not found in playlist"}), 404
    _room_stop_internal(code)
    _room_no_advance.discard(code)
    room["playing"] = True
    _room_start_item(code, idx)
    return jsonify({"message": f'Now playing for everyone: {room["queue"][idx]["title"]}'})


@app.route("/api/room/<code>/sync", methods=["POST"])
def api_room_sync(code):
    """Restart current song so all members reconnect in sync."""
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    if not room["queue"]:
        return jsonify({"error": "Playlist is empty"}), 400
    _room_stop_internal(code)
    _room_no_advance.discard(code)
    room["playing"] = True
    _room_start_item(code, room["current_idx"])
    return jsonify({"message": "Synced — all members restarting current song."})


@app.route("/api/room/<code>/skip", methods=["POST"])
def api_room_skip(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    next_idx = room["current_idx"] + 1
    if next_idx >= len(room["queue"]):
        _room_stop_internal(code)
        return jsonify({"message": "Queue finished."})
    _room_stop_internal(code)
    _room_no_advance.discard(code)
    room["playing"] = True
    _room_start_item(code, next_idx)
    return jsonify({"message": f'Skipped to: {room["queue"][next_idx]["title"]}'})


@app.route("/api/room/<code>/stop", methods=["POST"])
def api_room_stop(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    _room_stop_internal(code)
    _rooms[code]["paused"] = False
    return jsonify({"message": "Stopped."})


@app.route("/api/room/<code>/pause", methods=["POST"])
def api_room_pause(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    _room_stop_internal(code)
    _rooms[code]["paused"] = True
    return jsonify({"message": "Room paused — meeting mode on."})


@app.route("/api/room/<code>/resume", methods=["POST"])
def api_room_resume(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    room = _rooms[code]
    if not room["queue"]:
        return jsonify({"error": "Queue is empty."}), 400
    room["paused"] = False
    _room_no_advance.discard(code)
    _room_start_item(code, room["current_idx"])
    return jsonify({"message": "Resumed."})



# ── Room HTTP stream ───────────────────────────────────────────────────────────

@app.route("/room/<code>/stream")
def room_stream(code):
    if code not in _rooms:
        return jsonify({"error": "Room not found"}), 404
    client_q = queue.Queue(maxsize=400)
    with _room_lock:
        _room_clients.setdefault(code, []).append(client_q)

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
                ql = _room_clients.get(code, [])
                if client_q in ql:
                    ql.remove(client_q)

    return Response(stream_with_context(generate()), mimetype="audio/mpeg",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Join page ──────────────────────────────────────────────────────────────────

@app.route("/join/<code>")
def join_room(code):
    return render_template_string(JOIN_HTML, code=code)


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speaker Player</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--sb:180px;--accent:#2563eb;--bg:#f3f4f6;--card:#fff;--border:#e5e7eb;--text:#111827;--muted:#6b7280}
body{font-family:Arial,Helvetica,sans-serif;font-size:14px;color:var(--text);background:var(--bg);display:flex;min-height:100vh}

/* sidebar */
.sb{width:var(--sb);background:#1e293b;color:#cbd5e1;display:flex;flex-direction:column;padding:24px 0;flex-shrink:0}
.sb-logo{font-size:15px;font-weight:700;color:#f8fafc;padding:0 20px 24px;border-bottom:1px solid #334155;margin-bottom:16px;line-height:1.4}
.sb-logo small{display:block;font-size:11px;color:#94a3b8;font-weight:400;margin-top:2px}
.nav{display:flex;align-items:center;gap:10px;padding:10px 20px;cursor:pointer;font-size:13px;font-weight:500;border-left:3px solid transparent;transition:background .15s,border-color .15s;user-select:none}
.nav:hover{background:#334155}
.nav.active{background:#1d4ed8;color:#fff;border-left-color:#60a5fa}
.nav-icon{font-size:17px}

/* main */
.main{flex:1;padding:28px 24px;overflow-y:auto}
.panel{display:none;max-width:640px}
.panel.active{display:block}
h2{font-size:18px;font-weight:700;margin-bottom:18px}

/* cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:14px}
.ct{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:12px}

/* device list */
.drow{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #f2f2f2}
.drow:last-child{border-bottom:none}
.dlbl{flex:1;cursor:pointer}
.dsr{font-size:12px;color:#bbb;margin-left:4px}
.dly{display:flex;align-items:center;gap:4px;white-space:nowrap}
.dly span{font-size:12px;color:var(--muted)}
.dly input{width:58px;padding:3px 5px;border:1px solid #ccc;border-radius:3px;text-align:right;font-size:13px}

/* inputs */
.frow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.txtin{flex:1;min-width:0;padding:7px 10px;border:1px solid #ccc;border-radius:5px;font-size:13px}
.vrow{display:flex;align-items:center;gap:10px}
#vol-range{width:220px;cursor:pointer}
#vol-pct{width:36px;text-align:right;color:var(--muted)}

/* buttons */
.btns{display:flex;gap:10px;flex-wrap:wrap}
.btn{padding:8px 18px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600}
.btn:hover{filter:brightness(90%)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.bg{background:#6b7280;color:#fff}
.bb{background:#2563eb;color:#fff}
.bgr{background:#16a34a;color:#fff}
.br{background:#dc2626;color:#fff}
.bsl{background:#334155;color:#fff}
.bam{background:#7c3aed;color:#fff}

/* messages */
.msg{margin-top:12px;padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.5;display:none}
.msg.ok{background:#dcfce7;color:#14532d}
.msg.err{background:#fee2e2;color:#7f1d1d}
.msg.warn{background:#fef9c3;color:#713f12}

/* room */
.rcode{font-size:40px;font-weight:900;letter-spacing:.2em;color:var(--accent);background:#eff6ff;border:2px dashed #bfdbfe;border-radius:8px;padding:14px 20px;text-align:center;margin:8px 0}
.slink{font-size:12px;color:var(--muted);word-break:break-all;background:#f9fafb;border:1px solid var(--border);border-radius:4px;padding:6px 10px;margin-top:4px}
.badge{display:inline-block;background:#dbeafe;color:#1e40af;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600}
.hint{font-size:12px;color:var(--muted);margin-top:6px}
.sep{border:none;border-top:1px solid var(--border);margin:12px 0}

/* queue list */
.qi{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:6px;margin-bottom:4px;background:#f9fafb;border:1px solid var(--border)}
.qi.current{background:#eff6ff;border-color:#bfdbfe}
.qi-num{font-size:12px;color:var(--muted);width:20px;text-align:right;flex-shrink:0}
.qi-title{flex:1;font-size:13px;font-weight:500}
.qi-by{font-size:11px;color:#9ca3af;margin-left:4px}
.qi-now{font-size:11px;font-weight:700;color:var(--accent);margin-left:4px}
.qi-by-badge{font-size:11px;background:#e0e7ff;color:#3730a3;border-radius:999px;
             padding:2px 8px;white-space:nowrap;flex-shrink:0}
.qi.current .qi-by-badge{background:#bfdbfe;color:#1e40af}
.qi.past{opacity:.45}
.qi-rm{background:none;border:none;color:#ef4444;cursor:pointer;font-size:16px;padding:0 4px;line-height:1;flex-shrink:0}
.qi-rm:hover{color:#b91c1c}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:6px}
.status-dot.live{background:#22c55e;animation:pulse 1.2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

@media(max-width:560px){
  .sb{width:52px}
  .sb-logo,.nav-label{display:none}
  .nav{justify-content:center;padding:12px}
  .main{padding:16px 10px}
}
</style>
</head>
<body>

<nav class="sb">
  <div class="sb-logo">Speaker<br>Player<small>Multi-output audio</small></div>
  <div class="nav active" onclick="switchTab('host',this)">
    <span class="nav-icon">&#128266;</span><span class="nav-label">Single Host</span>
  </div>
  <div class="nav" onclick="switchTab('room',this)">
    <span class="nav-icon">&#127968;</span><span class="nav-label">Room</span>
  </div>
</nav>

<main class="main">

<!-- ═══ Single Host ═══ -->
<div class="panel active" id="panel-host">
  <h2>Single Host</h2>

  <div class="card">
    <div class="ct">Speakers</div>
    <div style="margin-bottom:10px">
      <button class="btn bg" onclick="loadDevices()">&#8635; Refresh</button>
    </div>
    <div id="dev-list"><span style="color:#999">Loading&hellip;</span></div>
  </div>

  <div class="card">
    <div class="ct">Song <small style="font-size:11px;color:#bbb;font-weight:400;text-transform:none;letter-spacing:0">MP3 &middot; WAV &middot; FLAC &middot; OGG</small></div>
    <div class="frow">
      <input type="file" id="h-file" accept=".mp3,.wav,.flac,.ogg">
      <button class="btn bb" onclick="hUpload()">Upload</button>
      <span id="h-song" style="font-size:13px;color:#16a34a"></span>
    </div>
  </div>

  <div class="card">
    <div class="ct">Or stream from YouTube</div>
    <div class="frow">
      <input class="txtin" type="text" id="h-yt" placeholder="https://www.youtube.com/watch?v=...">
      <button class="btn br" id="h-yt-btn" onclick="hLoadYT()">Load</button>
    </div>
    <div class="hint">Resolves stream URL (~2&ndash;5 s) &mdash; nothing downloaded.</div>
  </div>

  <div class="card">
    <div class="ct">Volume</div>
    <div class="vrow">
      <input type="range" id="vol-range" min="0" max="100" value="100" oninput="onVol(this.value)">
      <span id="vol-pct">100%</span>
    </div>
  </div>

  <div class="btns">
    <button class="btn bgr" onclick="hPlay()">&#9654; Play</button>
    <button class="btn br" onclick="hStop()">&#9646;&#9646; Stop</button>
  </div>
  <div class="msg" id="msg-host"></div>
</div>


<!-- ═══ Room ═══ -->
<div class="panel" id="panel-room">
  <h2>Room</h2>

  <!-- Create or join room -->
  <div class="card" id="r-setup">
    <div class="ct">Create a room</div>
    <p style="font-size:13px;color:var(--muted);margin-bottom:12px">
      Pick a 4-letter room code. Anyone on the same WiFi joins with this code.
    </p>
    <div class="frow" style="margin-bottom:6px">
      <input class="txtin" type="text" id="create-code" maxlength="4" value="PLAY"
             style="font-size:20px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;max-width:140px;text-align:center"
             oninput="this.value=this.value.toUpperCase().replace(/[^A-Z]/g,'')"
             onkeydown="if(event.key==='Enter')createRoom()">
      <button class="btn bsl" onclick="createRoom()">&#43; Create Room</button>
    </div>
    <div class="hint">Default is PLAY &mdash; change it to anything you like (4 letters).</div>

    <hr class="sep" style="margin:16px 0">

    <div class="ct">Join a room</div>
    <div class="frow">
      <input class="txtin" type="text" id="join-code" maxlength="4"
             placeholder="Enter 4-letter code (e.g. PLAY)"
             oninput="this.value=this.value.toUpperCase().replace(/[^A-Z]/g,'')"
             onkeydown="if(event.key==='Enter')joinRoom()">
      <button class="btn bb" onclick="joinRoom()">Join &rarr;</button>
    </div>
  </div>

  <!-- Room dashboard (hidden until created) -->
  <div id="r-dash" style="display:none">

    <!-- Code + share -->
    <div class="card">
      <div class="ct">Room Code</div>
      <div class="rcode" id="r-code">????</div>
      <div class="slink" id="r-link"></div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <span>Listeners: <span class="badge" id="r-members">0</span></span>
        <span><span class="status-dot" id="r-dot"></span><span id="r-status">Stopped</span></span>
      </div>
    </div>

    <!-- Now playing + controls -->
    <div class="card">
      <div class="ct">Controls</div>
      <div id="r-now" style="font-size:14px;font-weight:600;color:#1e293b;margin-bottom:12px;min-height:20px">—</div>
      <div class="btns" style="margin-bottom:10px">
        <button class="btn bgr" onclick="rPlay()">&#9654; Play</button>
        <button class="btn bam" onclick="rSkip()">&#9197; Skip</button>
        <button class="btn br"  onclick="rStopRoom()">&#9646;&#9646; Stop Room</button>
        <button class="btn" style="background:#0891b2;color:#fff" onclick="rSync()" title="Restart current song for all members">&#8635; Sync All</button>
      </div>
      <div style="border-top:1px solid var(--border);padding-top:10px;margin-top:4px">
        <button class="btn" id="r-meeting-btn" onclick="rToggleMeeting()"
          style="background:#f59e0b;color:#fff;width:100%;font-size:13px">
          &#127908; Meeting Mode &mdash; Off
        </button>
        <div class="hint">Pauses audio for everyone in the room. Press again to resume.</div>
      </div>
    </div>

    <!-- Playlist / Queue -->
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <span class="ct" style="margin-bottom:0">Playlist</span>
        <span id="r-q-count" style="font-size:12px;color:var(--muted)">0 songs</span>
      </div>
      <div id="r-q-list"><span style="color:#999;font-size:13px">Playlist is empty &mdash; add songs below.</span></div>
    </div>

    <!-- Add to queue -->
    <div class="card">
      <div class="ct">Add to Playlist</div>
      <div class="frow" style="margin-bottom:10px">
        <input type="file" id="r-file" accept=".mp3,.wav,.flac,.ogg">
        <button class="btn bb" onclick="rAddFile()">&#43; Add File</button>
      </div>
      <hr class="sep">
      <div class="frow">
        <input class="txtin" type="text" id="r-yt" placeholder="YouTube URL">
        <button class="btn br" id="r-yt-btn" onclick="rAddYT()">&#43; Add YouTube</button>
      </div>
      <div class="hint">Anyone in the room can add songs from their device too.</div>
    </div>

  </div><!-- /r-dash -->

  <div class="msg" id="msg-room"></div>
</div><!-- /panel-room -->

</main>

<script>
const $ = id => document.getElementById(id);
let _rCode = null, _pollTimer = null;

function switchTab(name, el) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav').forEach(n => n.classList.remove('active'));
  $('panel-' + name).classList.add('active');
  el.classList.add('active');
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function showMsg(id, html, type) {
  const el = $(id); el.innerHTML = html;
  el.className = 'msg ' + type; el.style.display = 'block';
}
async function api(url, opts) {
  try {
    const r = await fetch(url, opts || {});
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('application/json')) return { ok: r.ok, data: await r.json() };
    return { ok: false, data: { error: `Server error ${r.status}` } };
  } catch(e) { return { ok: false, data: { error: 'Network error: ' + e } }; }
}

/* ── Single Host ── */
async function loadDevices() {
  $('dev-list').innerHTML = '<span style="color:#999">Loading&hellip;</span>';
  const { ok, data } = await api('/api/devices');
  const box = $('dev-list');
  if (!ok || !Array.isArray(data) || !data.length) {
    box.innerHTML = '<span style="color:#999">No WASAPI devices found. Connect speakers then Refresh.</span>'; return;
  }
  box.innerHTML = '';
  data.forEach(d => {
    const row = document.createElement('div'); row.className = 'drow';
    row.innerHTML =
      `<input type="checkbox" id="cb${d.id}" value="${d.id}">` +
      `<label class="dlbl" for="cb${d.id}">${esc(d.name)}<span class="dsr">${d.sample_rate} Hz</span></label>` +
      `<div class="dly"><span>Delay</span><input type="number" id="dl${d.id}" value="0" min="0" max="10000"><span>ms</span></div>`;
    box.appendChild(row);
  });
}
async function hUpload() {
  const inp = $('h-file');
  if (!inp.files.length) { showMsg('msg-host','Choose a file first.','err'); return; }
  const fd = new FormData(); fd.append('file', inp.files[0]);
  showMsg('msg-host','Uploading&hellip;','ok');
  const { ok, data } = await api('/api/upload', { method:'POST', body:fd });
  if (ok) { $('h-song').textContent = data.message; showMsg('msg-host', esc(data.message), 'ok'); }
  else { showMsg('msg-host', esc(data.error || 'Upload failed.'), 'err'); }
}
async function hLoadYT() {
  const url = $('h-yt').value.trim();
  if (!url) { showMsg('msg-host','Paste a YouTube URL.','err'); return; }
  $('h-yt-btn').disabled = true;
  showMsg('msg-host','Resolving&hellip; (~2&ndash;5 s)','ok');
  const { ok, data } = await api('/api/youtube', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})
  });
  $('h-yt-btn').disabled = false;
  if (ok) { $('h-song').textContent = data.title || data.message; showMsg('msg-host', esc(data.message), 'ok'); }
  else showMsg('msg-host', esc(data.error || 'Failed.'), 'err');
}
async function hPlay() {
  const checked = document.querySelectorAll('#dev-list input[type=checkbox]:checked');
  if (!checked.length) { showMsg('msg-host','Select at least one speaker.','err'); return; }
  const devices = [...checked].map(c => ({ id:+c.value, delay_ms:+($('dl'+c.value).value)||0 }));
  showMsg('msg-host','Starting&hellip;','ok');
  const { ok, data } = await api('/api/play', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({devices})
  });
  if (ok) {
    const w = data.warnings;
    showMsg('msg-host', esc(data.message) + (w&&w.length ? '<br><small>&#9888; '+w.map(esc).join('<br>')+'</small>' : ''), w&&w.length?'warn':'ok');
  } else showMsg('msg-host', esc(data.error||'Failed.'), 'err');
}
async function hStop() {
  const { ok, data } = await api('/api/stop', {method:'POST'});
  showMsg('msg-host', esc(data.message||(ok?'Stopped.':'Error.')), ok?'ok':'err');
}
let volTimer;
function onVol(v) {
  $('vol-pct').textContent = v + '%';
  clearTimeout(volTimer);
  volTimer = setTimeout(() => api('/api/volume', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({volume:+v})
  }), 100);
}

/* ── Room ── */
function joinRoom() {
  const code = ($('join-code').value || '').trim().toUpperCase();
  if (code.length !== 4) { showMsg('msg-room', 'Enter the 4-letter room code.', 'err'); return; }
  window.location.href = '/join/' + code;
}
async function createRoom() {
  const code = ($('create-code').value || '').trim().toUpperCase();
  if (code.length !== 4) { showMsg('msg-room', 'Room code must be exactly 4 letters.', 'err'); return; }
  const { ok, data } = await api('/api/room/create', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ code })
  });
  if (!ok) { showMsg('msg-room', esc(data.error||'Failed.'), 'err'); return; }
  _rCode = data.code;
  $('r-code').textContent = _rCode;
  const link = location.protocol+'//'+location.hostname+(location.port?':'+location.port:'')+'/join/'+_rCode;
  $('r-link').textContent = link;
  $('r-setup').style.display = 'none';
  $('r-dash').style.display = 'block';
  showMsg('msg-room', 'Room created! Share the link above with your listeners.', 'ok');
  startPoll();
}
async function rPlay() {
  if (!_rCode) return;
  showMsg('msg-room','Starting stream&hellip;','ok');
  const { ok, data } = await api('/api/room/'+_rCode+'/play', {method:'POST'});
  showMsg('msg-room', esc(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
}
async function rSkip() {
  if (!_rCode) return;
  const { ok, data } = await api('/api/room/'+_rCode+'/skip', {method:'POST'});
  showMsg('msg-room', esc(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
}
async function rSync() {
  if (!_rCode) return;
  showMsg('msg-room', 'Syncing all members&hellip;', 'ok');
  const { ok, data } = await api('/api/room/'+_rCode+'/sync', {method:'POST'});
  showMsg('msg-room', esc(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
}
async function rJump(itemId) {
  if (!_rCode) return;
  showMsg('msg-room', 'Switching song for everyone&hellip;', 'ok');
  const { ok, data } = await api('/api/room/'+_rCode+'/jump/'+itemId, {method:'POST'});
  showMsg('msg-room', esc(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
}
async function rStopRoom() {
  if (!_rCode) return;
  const { ok, data } = await api('/api/room/'+_rCode+'/stop', {method:'POST'});
  showMsg('msg-room', esc(data.message||(ok?'Stopped.':'Error.')), ok?'ok':'err');
  setMeetingBtn(false);
}
let _meetingOn = false;
async function rToggleMeeting() {
  if (!_rCode) return;
  if (!_meetingOn) {
    const { ok, data } = await api('/api/room/'+_rCode+'/pause', {method:'POST'});
    if (ok) { _meetingOn = true; setMeetingBtn(true); showMsg('msg-room', esc(data.message), 'warn'); }
    else showMsg('msg-room', esc(data.error||'Failed.'), 'err');
  } else {
    const { ok, data } = await api('/api/room/'+_rCode+'/resume', {method:'POST'});
    if (ok) { _meetingOn = false; setMeetingBtn(false); showMsg('msg-room', esc(data.message), 'ok'); }
    else showMsg('msg-room', esc(data.error||'Failed.'), 'err');
  }
}
function setMeetingBtn(on) {
  const btn = $('r-meeting-btn');
  if (!btn) return;
  btn.style.background = on ? '#dc2626' : '#f59e0b';
  btn.innerHTML = on ? '&#127908; Meeting Mode &mdash; ON (tap to resume)' : '&#127908; Meeting Mode &mdash; Off';
}
async function rAddFile() {
  if (!_rCode) return;
  const inp = $('r-file');
  if (!inp.files.length) { showMsg('msg-room','Choose a file first.','err'); return; }
  const fd = new FormData(); fd.append('file', inp.files[0]);
  showMsg('msg-room','Adding to queue&hellip;','ok');
  const { ok, data } = await api('/api/room/'+_rCode+'/queue/add', {method:'POST', body:fd});
  showMsg('msg-room', esc(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
  if (ok) { inp.value = ''; pollState(); }
}
async function rAddYT() {
  if (!_rCode) return;
  const url = $('r-yt').value.trim();
  if (!url) { showMsg('msg-room','Paste a YouTube URL.','err'); return; }
  $('r-yt-btn').disabled = true;
  showMsg('msg-room','Resolving YouTube&hellip; (~2&ndash;5 s)','ok');
  const { ok, data } = await api('/api/room/'+_rCode+'/queue/add', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})
  });
  $('r-yt-btn').disabled = false;
  showMsg('msg-room', esc(ok ? data.message : (data.error||'Failed.')), ok?'ok':'err');
  if (ok) { $('r-yt').value = ''; pollState(); }
}
async function rRemove(id) {
  if (!_rCode) return;
  const { ok, data } = await api('/api/room/'+_rCode+'/queue/remove/'+id, {method:'POST'});
  if (ok) pollState();
  else showMsg('msg-room', esc(data.error||'Failed.'), 'err');
}

function srcIcon(src) {
  return src === 'youtube' ? '<span title="YouTube" style="color:#dc2626;font-size:12px">&#9654; YT</span>'
                           : '<span title="File" style="color:#6b7280;font-size:12px">&#127925; File</span>';
}
function renderQueue(q, currentIdx, playing) {
  const box = $('r-q-list');
  $('r-q-count').textContent = q.length + ' song' + (q.length===1?'':'s');
  if (!q.length) {
    box.innerHTML = '<span style="color:#999;font-size:13px">Playlist is empty &mdash; add songs below.</span>'; return;
  }
  box.innerHTML = '';
  q.forEach((it, i) => {
    const isCurrent = (i === currentIdx);
    const isPast    = (i < currentIdx);
    const div = document.createElement('div');
    div.className = 'qi' + (isCurrent ? ' current' : '') + (isPast ? ' past' : '');
    div.innerHTML =
      `<span class="qi-num">${i+1}</span>` +
      `<span style="margin-right:6px">${srcIcon(it.source)}</span>` +
      `<span class="qi-title">${esc(it.title)}</span>` +
      `<span class="qi-by-badge">${esc(it.added_by)}</span>` +
      (isCurrent && playing ? '<span class="qi-now">&#9654; now</span>' : isCurrent ? '<span class="qi-now">next</span>' : '') +
      (!isCurrent ? `<button class="btn" style="background:#059669;color:#fff;padding:3px 8px;font-size:11px;flex-shrink:0" onclick="rJump('${it.id}')" title="Play this for everyone">&#9654; All</button>` : '') +
      (!isPast ? `<button class="qi-rm" onclick="rRemove('${it.id}')" title="Remove">&#215;</button>` : '');
    box.appendChild(div);
  });
}

async function pollState() {
  if (!_rCode) return;
  const { ok, data } = await api('/api/room/'+_rCode+'/state');
  if (!ok) return;
  $('r-members').textContent = data.member_count || 0;
  const dot = $('r-dot'), st = $('r-status');
  const label = data.paused ? 'Meeting mode' : data.playing ? ('Playing: ' + (data.current_title||'')) : 'Stopped';
  dot.className = 'status-dot' + (data.playing ? ' live' : '');
  st.textContent = label;
  $('r-now').textContent = data.current_title || '—';
  if (data.paused !== _meetingOn) { _meetingOn = data.paused; setMeetingBtn(data.paused); }
  renderQueue(data.queue, data.current_idx, data.playing);
}
function startPoll() {
  clearInterval(_pollTimer);
  _pollTimer = setInterval(pollState, 2000);
  pollState();
}

loadDevices();
</script>
</body>
</html>"""


JOIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Room {{ code }}</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,Helvetica,sans-serif;background:#0f172a;color:#f1f5f9;
     min-height:100vh;padding:24px 16px;display:flex;justify-content:center}
.wrap{max-width:460px;width:100%}
.room-badge{font-size:13px;color:#64748b;margin-bottom:4px;letter-spacing:.05em;text-transform:uppercase}
h1{font-size:28px;font-weight:900;letter-spacing:.15em;color:#60a5fa;margin-bottom:20px}
.card{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:14px;border:1px solid #334155}
.ct{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#64748b;margin-bottom:12px}
.now{font-size:15px;font-weight:600;color:#f8fafc;margin-bottom:4px;min-height:22px}
.added{font-size:12px;color:#64748b;margin-bottom:14px}
audio{width:100%;border-radius:8px;margin-bottom:10px}
.status{display:flex;align-items:center;gap:8px;font-size:12px;color:#64748b}
.dot{width:8px;height:8px;border-radius:50%;background:#ef4444;flex-shrink:0}
.dot.live{background:#22c55e;animation:pulse 1.2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.frow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.txtin{flex:1;min-width:0;padding:7px 10px;border:1px solid #334155;border-radius:5px;
       font-size:13px;background:#0f172a;color:#f1f5f9}
.txtin::placeholder{color:#475569}
.btn{padding:8px 14px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600}
.btn:disabled{opacity:.5;cursor:not-allowed}
.bb{background:#2563eb;color:#fff}
.br{background:#dc2626;color:#fff}
.btn:hover{filter:brightness(90%)}
.msg{padding:8px 12px;border-radius:6px;font-size:12px;margin-top:8px;display:none}
.msg.ok{background:#14532d;color:#bbf7d0}
.msg.err{background:#7f1d1d;color:#fecaca}

.qi{padding:7px 10px;border-radius:6px;font-size:13px;margin-bottom:4px;
    background:#0f172a;border:1px solid #334155;display:flex;gap:8px;align-items:center}
.qi.current{border-color:#3b82f6;background:#1e3a5f}
.qi-n{font-size:11px;color:#64748b;width:18px;text-align:right;flex-shrink:0}
.qi-t{flex:1;font-weight:500}
.qi-now{font-size:11px;color:#60a5fa;margin-left:6px}
.hint{font-size:12px;color:#475569;margin-top:6px}
</style>
</head>
<body>

<!-- Meeting mode overlay -->
<div id="meeting-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);
     z-index:999;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px">
  <div style="font-size:56px;margin-bottom:16px">&#127908;</div>
  <div style="font-size:22px;font-weight:800;color:#fbbf24;margin-bottom:8px">Meeting in Progress</div>
  <div style="font-size:14px;color:#94a3b8">Music paused by the host. It will resume when the meeting ends.</div>
</div>

<div class="wrap">
  <div class="room-badge">Room</div>
  <h1>{{ code }}</h1>

  <div class="card">
    <div class="ct">Now Playing</div>
    <div class="now" id="now-title">Waiting for host&hellip;</div>

    <!-- hidden real audio element - controlled by JS only -->
    <audio id="player" style="display:none"></audio>

    <!-- Tap to listen button (satisfies mobile autoplay policy) -->
    <div id="tap-area" style="text-align:center;padding:20px 0">
      <button id="tap-btn" onclick="tapToListen()"
        style="background:#2563eb;color:#fff;border:none;border-radius:50%;
               width:72px;height:72px;font-size:28px;cursor:pointer;
               box-shadow:0 4px 14px rgba(37,99,235,.5)">&#9654;</button>
      <div style="font-size:12px;color:#64748b;margin-top:8px">Tap to connect audio</div>
    </div>

    <!-- Volume control (shown after tap) -->
    <div id="vol-area" style="display:none;margin-top:8px">
      <div style="display:flex;align-items:center;gap:10px">
        <span style="font-size:14px">&#128266;</span>
        <input type="range" id="j-vol" min="0" max="100" value="80"
               oninput="document.getElementById('player').volume=this.value/100"
               style="flex:1;cursor:pointer">
        <span id="j-vol-pct" style="font-size:12px;color:#64748b;width:34px">80%</span>
      </div>
    </div>

    <div class="status" style="margin-top:12px">
      <span class="dot" id="dot"></span>
      <span id="status-txt">Tap the button above to connect</span>
    </div>
  </div>

  <div class="card">
    <div class="ct">Your Name</div>
    <input class="txtin" type="text" id="j-name" placeholder="Enter your name (shown on playlist)"
           oninput="saveName()" style="width:100%">
    <div class="hint">Shows next to songs you add.</div>
  </div>

  <div class="card">
    <div class="ct">Add to Playlist</div>
    <div class="frow">
      <input type="file" id="j-file" accept=".mp3,.wav,.flac,.ogg" style="flex:1;min-width:0;color:#94a3b8">
      <button class="btn bb" onclick="jAddFile()">&#43; Add</button>
    </div>
    <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
    <div class="frow">
      <input class="txtin" type="text" id="j-yt" placeholder="YouTube URL">
      <button class="btn br" id="j-yt-btn" onclick="jAddYT()">&#43; Add</button>
    </div>
    <div class="hint">Your song plays after the current queue.</div>
    <div class="msg" id="j-msg"></div>
  </div>

  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <span class="ct" style="margin-bottom:0">Playlist</span>
      <span id="j-q-count" style="font-size:12px;color:#64748b"></span>
    </div>
    <div id="j-q-list"><span style="color:#475569;font-size:13px">Empty</span></div>
  </div>
</div>

<script>
const CODE = "{{ code }}";
const player = document.getElementById('player');
let _playing = false, _streamTs = 0, _tapped = false;

function tapToListen() {
  _tapped = true;
  document.getElementById('tap-btn').style.background = '#16a34a';
  document.getElementById('tap-btn').innerHTML = '&#10003;';
  document.getElementById('tap-area').querySelector('div').textContent = 'Connected — waiting for host';
  document.getElementById('vol-area').style.display = 'block';
  document.getElementById('j-vol').addEventListener('input', function(){
    document.getElementById('j-vol-pct').textContent = this.value + '%';
  });
  // if host is already playing, start immediately
  pollState();
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function showMsg(html,type){const el=document.getElementById('j-msg');el.innerHTML=html;el.className='msg '+type;el.style.display='block'}
function myName(){return (document.getElementById('j-name').value||'').trim()||'Guest'}
function saveName(){try{localStorage.setItem('room_name',document.getElementById('j-name').value);}catch(e){}}

async function api(url,opts){
  try{
    const r=await fetch(url,opts||{});
    const ct=r.headers.get('content-type')||'';
    if(ct.includes('application/json'))return{ok:r.ok,data:await r.json()};
    return{ok:false,data:{error:'Server error '+r.status}};
  }catch(e){return{ok:false,data:{error:'Network error: '+e}};}
}

async function jAddFile(){
  const inp=document.getElementById('j-file');
  if(!inp.files.length){showMsg('Choose a file first.','err');return;}
  const fd=new FormData();fd.append('file',inp.files[0]);fd.append('name',myName());
  showMsg('Adding&hellip;','ok');
  const{ok,data}=await api('/api/room/'+CODE+'/queue/add',{method:'POST',body:fd});
  showMsg(esc(ok?data.message:(data.error||'Failed.')),ok?'ok':'err');
  if(ok){inp.value='';pollState();}
}
async function jAddYT(){
  const url=document.getElementById('j-yt').value.trim();
  if(!url){showMsg('Paste a YouTube URL.','err');return;}
  document.getElementById('j-yt-btn').disabled=true;
  showMsg('Resolving&hellip; (~2&ndash;5 s)','ok');
  const{ok,data}=await api('/api/room/'+CODE+'/queue/add',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url,name:myName()})
  });
  document.getElementById('j-yt-btn').disabled=false;
  showMsg(esc(ok?data.message:(data.error||'Failed.')),ok?'ok':'err');
  if(ok){document.getElementById('j-yt').value='';pollState();}
}

function jSrcIcon(src){
  return src==='youtube'
    ? '<span style="color:#ef4444;font-size:11px;margin-right:4px">&#9654; YT</span>'
    : '<span style="color:#6b7280;font-size:11px;margin-right:4px">&#127925;</span>';
}
function renderQueue(q,idx,playing){
  const box=document.getElementById('j-q-list');
  document.getElementById('j-q-count').textContent=q.length+' song'+(q.length===1?'':'s');
  if(!q.length){box.innerHTML='<span style="color:#475569;font-size:13px">Empty</span>';return;}
  box.innerHTML='';
  q.forEach((it,i)=>{
    const isCur=(i===idx);const isPast=(i<idx);
    const div=document.createElement('div');
    div.className='qi'+(isCur?' current':'')+(isPast?' past':'');
    div.innerHTML=
      `<span class="qi-n">${i+1}</span>`+
      jSrcIcon(it.source)+
      `<span class="qi-t">${esc(it.title)}</span>`+
      `<span style="font-size:11px;background:#1e3a5f;color:#93c5fd;border-radius:999px;padding:2px 8px;white-space:nowrap;flex-shrink:0">${esc(it.added_by)}</span>`+
      (isCur&&playing?'<span class="qi-now" style="margin-left:6px">&#9654; now</span>':'');
    box.appendChild(div);
  });
}

async function pollState(){
  const{ok,data}=await api('/api/room/'+CODE+'/state');
  if(!ok){document.getElementById('status-txt').textContent='Room not found.';return;}
  const dot=document.getElementById('dot');
  const st=document.getElementById('status-txt');
  const nowTitle=document.getElementById('now-title');

  renderQueue(data.queue,data.current_idx,data.playing);

  if(data.paused){
    if(_playing){
      _playing=false;
      player.pause();   // stop immediately, don't wait for buffer to drain
      player.src='';
    }
    showMeetingOverlay(true);
    dot.className='dot';dot.style.background='#f59e0b';
    st.textContent='Meeting in progress';
    nowTitle.textContent='Meeting mode on';
  } else if(data.playing&&!_playing&&_tapped){
    _playing=true;_streamTs=Date.now();
    document.getElementById('tap-area').querySelector('div').textContent='Now playing';
    player.src='/room/'+CODE+'/stream?t='+_streamTs;
    player.play().catch(()=>{});
    dot.className='dot live';dot.style.background='';
    st.textContent='Live';
    nowTitle.textContent=data.current_title||'Now playing';
  } else if(!data.playing&&_playing){
    _playing=false;player.pause();player.src='';
    showMeetingOverlay(false);
    dot.className='dot';dot.style.background='';
    st.textContent='Waiting for host…';
    nowTitle.textContent='Waiting for host…';
  } else if(data.playing){
    showMeetingOverlay(false);
    nowTitle.textContent=data.current_title||'Now playing';
    dot.style.background='';st.textContent='Live';
    if(player.paused&&!player.ended){player.play().catch(()=>{});}
  }
}

function showMeetingOverlay(show){
  const ov=document.getElementById('meeting-overlay');
  if(ov) ov.style.display = show ? 'flex' : 'none';
}

player.addEventListener('ended',()=>{
  _playing=false;
  document.getElementById('dot').className='dot';
  document.getElementById('status-txt').textContent='Waiting for next song&hellip;';
});

try{const n=localStorage.getItem('room_name');if(n)document.getElementById('j-name').value=n;}catch(e){}
pollState();
setInterval(pollState, 1000);
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

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
