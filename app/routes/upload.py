import os
from flask import Blueprint, request, jsonify, session
from app.pdf_utils import extract_text_from_pdf, clean_text
from app.rag import ingest_document

upload_bp = Blueprint("upload", __name__)

MAX_TEXT_CHARS = 200_000  # ~50k tokens guard


@upload_bp.route("/", methods=["POST"])
def upload():
    text = ""

    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400
        try:
            text = extract_text_from_pdf(file.read())
        except Exception as e:
            return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500

    elif request.json and request.json.get("text"):
        text = request.json["text"]

    else:
        return jsonify({"error": "No file or text provided."}), 400

    text = clean_text(text)
    if len(text) < 100:
        return jsonify({"error": "Document too short or empty."}), 400

    text = text[:MAX_TEXT_CHARS]

    try:
        doc_id, chunk_count = ingest_document(text)
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500

    # Store doc_id and first 3000 chars (for title extraction) in session
    session["doc_id"] = doc_id
    session["doc_preview"] = text[:3000]
    session["doc_text"] = text[:50000]  # store partial text for summary

    return jsonify({
        "doc_id": doc_id,
        "chunk_count": chunk_count,
        "preview": text[:300],
    })
