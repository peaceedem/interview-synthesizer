import json
import os
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from analyzer import synthesize

app = Flask(__name__, static_folder="public")
DATA_FILE = Path(__file__).parent / "transcripts.json"


def load_transcripts():
    if not DATA_FILE.exists():
        return []
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception:
        return []


def save_transcripts(transcripts):
    DATA_FILE.write_text(json.dumps(transcripts, indent=2))


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


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


@app.get("/api/synthesize")
def synthesize_route():
    transcripts = load_transcripts()
    if not transcripts:
        return jsonify({"error": "No transcripts to synthesize."}), 400

    def generate():
        try:
            report = synthesize(transcripts)
            # Stream section by section so the UI renders progressively
            for line in report.split("\n"):
                chunk = line + "\n"
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Interview Synthesizer running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
