from flask import Blueprint, request, jsonify, session
from app.rag import retrieve
from app.prompts import explain_concept

explain_bp = Blueprint("explain", __name__)


@explain_bp.route("/concept", methods=["POST"])
def concept():
    data = request.get_json(force=True) or {}
    concept = (data.get("concept") or "").strip()
    doc_id = data.get("doc_id") or session.get("doc_id")

    if not concept:
        return jsonify({"error": "Concept cannot be empty."}), 400

    context_chunks = []
    if doc_id:
        try:
            context_chunks = retrieve(doc_id, concept, top_k=4)
        except Exception:
            pass  # Explain without document context

    try:
        explanation = explain_concept(concept, context_chunks)
    except Exception as e:
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500

    return jsonify({"explanation": explanation})
