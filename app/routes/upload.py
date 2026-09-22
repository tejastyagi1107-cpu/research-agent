import os
from flask import Blueprint, request, jsonify, session
from app.pdf_utils import extract_pages_from_pdf, extract_text_from_pdf, clean_text
from app.rag import ingest_document, ingest_document_pages

upload_bp = Blueprint("upload", __name__)

MAX_TEXT_CHARS = 200_000  # ~50k tokens guard


@upload_bp.route("/", methods=["POST"])
def upload():
    text = ""
    pages = None  # list of {"page": int, "text": str}

    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400
        try:
            file_bytes = file.read()
            # Page-aware extraction for PDFs — preserves page numbers
            pages = extract_pages_from_pdf(file_bytes)
            if not pages:
                return jsonify({"error": "Could not extract text from PDF."}), 400
            # Clean each page's text
            for p in pages:
                p["text"] = clean_text(p["text"])
            pages = [p for p in pages if len(p["text"]) > 20]
            text = " ".join(p["text"] for p in pages)
        except Exception as e:
            return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500

    elif request.json and request.json.get("text"):
        text = request.json["text"]
        # Plain text paste — no page metadata available
        pages = None

    else:
        return jsonify({"error": "No file or text provided."}), 400

    text = clean_text(text)
    if len(text) < 100:
        return jsonify({"error": "Document too short or empty."}), 400

    try:
        if pages:
            # Trim pages so total text stays within token guard
            trimmed_pages = []
            total_chars = 0
            for p in pages:
                if total_chars + len(p["text"]) > MAX_TEXT_CHARS:
                    break
                trimmed_pages.append(p)
                total_chars += len(p["text"])
            doc_id, chunk_count = ingest_document_pages(trimmed_pages)
        else:
            # Paste-text fallback — no page metadata
            text = text[:MAX_TEXT_CHARS]
            doc_id, chunk_count = ingest_document(text)
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500

    # Store doc_id and preview in session
    session["doc_id"] = doc_id
    session["doc_preview"] = text[:3000]
    session["doc_text"] = text[:50000]  # partial text for summary fallback

    return jsonify({
        "doc_id": doc_id,
        "chunk_count": chunk_count,
        "preview": text[:300],
    })
