from flask import Blueprint, request, jsonify, session
from app.rag import retrieve_with_metadata
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
        chunk_results = retrieve_with_metadata(doc_id, question, top_k=6)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    if not chunk_results:
        return jsonify({"answer": "I couldn't find relevant sections in the paper to answer that question.", "sources": []})

    # Extract plain texts for LLM context
    context_chunks = [c["text"] for c in chunk_results]

    try:
        answer = answer_question(question, context_chunks, history=history)
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    # Build source metadata from retrieval — NOT from LLM
    sources = []
    seen_pages = set()
    for i, chunk in enumerate(chunk_results, start=1):
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        if page_start is not None:
            # Deduplicate by page range
            page_key = (page_start, page_end)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)

            if page_start == page_end:
                page_label = f"Page {page_start}"
            else:
                page_label = f"Pages {page_start}–{page_end}"
            sources.append({
                "label": f"Source {len(sources) + 1}",
                "page": page_label,
                "snippet": chunk["text"][:120].strip() + "…",
            })
        # If no page metadata (plain-text upload), skip source entry

    return jsonify({
        "answer": answer,
        "sources": sources,
        "doc_id": doc_id,
    })
