from flask import Blueprint, request, jsonify, session
from app.rag import VectorStore, _chunk_text
from app.prompts import generate_summary

summary_bp = Blueprint("summary", __name__)


def _select_representative_chunks(chunks: list, target: int = 12) -> list:
    """
    Select up to `target` chunks spread across the entire document.
    This prevents the summary from being biased toward the beginning.

    Strategy:
    - Always include the first 2 chunks (abstract / intro region).
    - Always include the last 2 chunks (conclusion / future work region).
    - Fill remaining slots evenly from the middle.
    """
    n = len(chunks)
    if n <= target:
        return chunks

    selected_indices = set()

    # Always take first 2 and last 2
    for i in range(min(2, n)):
        selected_indices.add(i)
    for i in range(max(0, n - 2), n):
        selected_indices.add(i)

    # Fill remaining slots from evenly spaced middle positions
    remaining = target - len(selected_indices)
    if remaining > 0:
        step = n / (remaining + 1)
        for k in range(1, remaining + 1):
            idx = int(k * step)
            idx = max(0, min(idx, n - 1))
            selected_indices.add(idx)

    # Return in original document order
    return [chunks[i] for i in sorted(selected_indices)]


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

        # Select representative chunks spread across the whole document
        selected = _select_representative_chunks(store.chunks, target=12)
        context_chunks = [_chunk_text(c) for c in selected]

        summary = generate_summary(context_chunks)
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

    return jsonify({"summary": summary, "doc_id": doc_id})
