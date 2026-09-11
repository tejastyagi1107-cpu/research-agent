from flask import Blueprint, request, jsonify, session
from app.rag import retrieve
from app.prompts import answer_question

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    doc_id = data.get("doc_id") or session.get("doc_id")
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if not doc_id:
        return jsonify({"error": "No document loaded. Please upload a paper first."}), 400

    try:
        chunks = retrieve(doc_id, question, top_k=6)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    if not chunks:
        return jsonify({"answer": "I couldn't find relevant sections in the paper to answer that question."})

    try:
        answer = answer_question(question, chunks, history=history)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    return jsonify({"answer": answer, "doc_id": doc_id})
