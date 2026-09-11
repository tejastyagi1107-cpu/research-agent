from flask import Blueprint, request, jsonify, session
from app.rag import VectorStore
from app.prompts import generate_summary

summary_bp = Blueprint("summary", __name__)


@summary_bp.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    doc_id = data.get("doc_id") or session.get("doc_id")

    if not doc_id:
        return jsonify({"error": "No document loaded. Please upload a paper first."}), 400

    try:
        store = VectorStore(doc_id)
        if not store.load():
            return jsonify({"error": "Document index not found. Please re-upload."}), 404

        # Use the first 10 chunks as context (covers abstract + intro + methods)
        context_chunks = store.chunks[:10]
        summary = generate_summary(context_chunks)
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

    return jsonify({"summary": summary, "doc_id": doc_id})
