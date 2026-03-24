import json
import os
import tempfile
import time
import uuid
from pathlib import Path

def _patch_whisper_ffmpeg():
    """
    Monkey-patch whisper.audio.load_audio to use the imageio-ffmpeg binary
    directly instead of looking for 'ffmpeg' on PATH.  This makes transcription
    work on Railway (and any environment) without a system ffmpeg install.
    """
    try:
        import imageio_ffmpeg
        import numpy as np
        from subprocess import run, CalledProcessError
        import whisper.audio as _wa

        _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        _SAMPLE_RATE = _wa.SAMPLE_RATE

        def _load_audio(file: str, sr: int = _SAMPLE_RATE):
            cmd = [
                _ffmpeg_exe,
                "-nostdin", "-threads", "0",
                "-i", file,
                "-f", "s16le", "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-",
            ]
            try:
                out = run(cmd, capture_output=True, check=True).stdout
            except CalledProcessError as e:
                raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        _wa.load_audio = _load_audio

    except Exception as e:
        print(f"Warning: could not patch whisper ffmpeg path: {e}")

_patch_whisper_ffmpeg()

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from analyzer import synthesize

app = Flask(__name__, static_folder="public")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

DATA_FILE = Path(__file__).parent / "transcripts.json"
TEMP_DIR = Path(tempfile.gettempdir()) / "interview_synthesizer"
TEMP_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac"}

# Cache the Whisper model in memory after first load
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def load_transcripts():
    if not DATA_FILE.exists():
        return []
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception:
        return []


def save_transcripts(transcripts):
    DATA_FILE.write_text(json.dumps(transcripts, indent=2))


def sse(status, message="", **extra):
    payload = {"status": status, "message": message, **extra}
    return f"data: {json.dumps(payload)}\n\n"


# ── Static ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("public", "index.html")


# ── Transcripts CRUD ─────────────────────────────────────────────────────────

@app.get("/api/transcripts")
def get_transcripts():
    transcripts = load_transcripts()
    return jsonify({"transcripts": transcripts, "count": len(transcripts)})


@app.post("/api/transcripts")
def add_transcript():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Transcript text is required."}), 400
    transcripts = load_transcripts()
    label = (body.get("label") or "").strip() or f"Interview {len(transcripts) + 1}"
    entry = {
        "id": int(time.time() * 1000),
        "label": label,
        "text": text,
        "addedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    transcripts.append(entry)
    save_transcripts(transcripts)
    return jsonify({"success": True, "transcript": entry, "count": len(transcripts)})


@app.delete("/api/transcripts/<int:tid>")
def delete_transcript(tid):
    transcripts = load_transcripts()
    filtered = [t for t in transcripts if t["id"] != tid]
    if len(filtered) == len(transcripts):
        return jsonify({"error": "Transcript not found."}), 404
    save_transcripts(filtered)
    return jsonify({"success": True, "count": len(filtered)})


@app.delete("/api/transcripts")
def clear_transcripts():
    save_transcripts([])
    return jsonify({"success": True, "count": 0})


# ── File upload ──────────────────────────────────────────────────────────────

@app.post("/api/upload")
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    file_id = str(uuid.uuid4())
    dest = TEMP_DIR / f"{file_id}{ext}"
    f.save(str(dest))
    return jsonify({"file_id": file_id, "filename": f.filename})


# ── Whisper transcription (SSE) ──────────────────────────────────────────────

@app.get("/api/transcribe/<file_id>")
def transcribe_file(file_id):
    # Validate file_id is a UUID to prevent path traversal
    try:
        uuid.UUID(file_id)
    except ValueError:
        return jsonify({"error": "Invalid file ID."}), 400

    matches = list(TEMP_DIR.glob(f"{file_id}.*"))
    if not matches:
        return jsonify({"error": "Upload not found. Please re-upload the file."}), 404

    audio_path = matches[0]

    @stream_with_context
    def generate():
        try:
            yield sse("loading", "Loading Whisper model — this may take a moment on first run…")

            model = get_whisper_model()

            yield sse("transcribing", "Transcribing audio… longer recordings may take a few minutes.")

            result = model.transcribe(str(audio_path), fp16=False)
            transcript = result["text"].strip()

            audio_path.unlink(missing_ok=True)
            yield sse("done", "Transcription complete.", transcript=transcript)

        except Exception as e:
            audio_path.unlink(missing_ok=True)
            yield sse("error", str(e))

    return Response(generate(), content_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no"})


# ── Synthesis (SSE) ──────────────────────────────────────────────────────────

@app.get("/api/synthesize")
def synthesize_route():
    transcripts = load_transcripts()
    if not transcripts:
        return jsonify({"error": "No transcripts to synthesize."}), 400

    @stream_with_context
    def generate():
        try:
            report = synthesize(transcripts)
            for line in report.split("\n"):
                yield f"data: {json.dumps({'text': line + chr(10)})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), content_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Interview Synthesizer running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
