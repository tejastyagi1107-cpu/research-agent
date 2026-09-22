import os
from flask import Blueprint, request, jsonify
from app.pdf_utils import extract_pages_from_pdf, clean_text
from app.rag import ingest_document_pages, retrieve_with_metadata, get_all_chunks
from app.prompts import generate_comparison

compare_bp = Blueprint("compare", __name__)

MAX_TEXT_CHARS = 200_000

# Category-specific retrieval queries — one per comparison axis
COMPARISON_QUERIES = {
    "research_problem": "problem motivation research gap limitation of existing approaches abstract introduction why proposed",
    "contribution": "main contributions proposed novel key contribution introduce we propose",
    "method": "What methodology or approach does the paper propose?",
    "architecture": "What model architecture or system design does the paper describe?",
    "dataset": "datasets benchmarks experimental setup training data evaluation corpus",
    "metrics": "evaluation metrics benchmark task accuracy F1 BLEU ROUGE score error rate",
    "results": "results evaluation benchmark scores table performance accuracy numerical findings experimental comparison percentage improvement",
    "limitations": "explicitly stated limitation drawback constraint weakness trade-off future work",
}


def _get_early_chunks(doc_id: str, max_chunks: int = 2):
    """
    Retrieve the first 2-3 meaningful chunks from the document
    (typically containing Abstract / Introduction / Motivation).
    Skips chunks that are too short (< 40 words) or contain reference-only text.
    """
    all_chunks = get_all_chunks(doc_id)
    early = []
    for c in all_chunks[:6]:
        if isinstance(c, str):
            c = {"text": c, "page_start": None, "page_end": None, "chunk_id": None}
        elif not isinstance(c, dict):
            continue
        text = c.get("text", "").strip()
        words = text.split()
        if len(words) < 40:
            continue
        lower = text.lower()
        if lower.startswith("references") or lower.startswith("bibliography"):
            continue
        early.append(c)
        if len(early) >= max_chunks:
            break
    return early


def _dedup_chunks(chunks):
    """Deduplicate chunk dicts preserving order, using chunk_id or text prefix."""
    seen_ids = set()
    seen_prefixes = set()
    unique = []
    for c in chunks:
        cid = c.get("chunk_id")
        text = c.get("text", "").strip()
        prefix = text[:100].lower()
        if cid and cid in seen_ids:
            continue
        if prefix and prefix in seen_prefixes:
            continue
        if cid:
            seen_ids.add(cid)
        if prefix:
            seen_prefixes.add(prefix)
        unique.append(c)
    return unique


def _ingest_pdf(file_storage, label):
    """Extract, clean, chunk, embed, and index a single PDF. Returns (doc_id, chunk_count)."""
    if not file_storage.filename.lower().endswith(".pdf"):
        raise ValueError(f"{label} must be a PDF file.")

    pages = extract_pages_from_pdf(file_storage.read())
    if not pages:
        raise ValueError(f"Could not extract text from {label}.")

    for p in pages:
        p["text"] = clean_text(p["text"])
    pages = [p for p in pages if len(p["text"]) > 20]

    if not pages:
        raise ValueError(f"{label} appears empty after processing.")

    # Trim to max chars
    trimmed = []
    total_chars = 0
    for p in pages:
        if total_chars + len(p["text"]) > MAX_TEXT_CHARS:
            break
        trimmed.append(p)
        total_chars += len(p["text"])

    return ingest_document_pages(trimmed)


def _page_label(chunk):
    """Human-readable page label from a retrieved chunk's metadata."""
    ps, pe = chunk.get("page_start"), chunk.get("page_end")
    if ps is None:
        return None
    return f"Page {ps}" if ps == pe else f"Pages {ps}\u2013{pe}"


@compare_bp.route("/analyze", methods=["POST"])
def analyze():
    # ── Validate both files ──
    if "paper_a" not in request.files or not request.files["paper_a"].filename:
        return jsonify({"error": "Please upload Paper A."}), 400
    if "paper_b" not in request.files or not request.files["paper_b"].filename:
        return jsonify({"error": "Please upload Paper B."}), 400

    # ── Ingest Paper A ──
    try:
        doc_id_a, count_a = _ingest_pdf(request.files["paper_a"], "Paper A")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process Paper A: {str(e)}"}), 500

    # ── Ingest Paper B ──
    try:
        doc_id_b, count_b = _ingest_pdf(request.files["paper_b"], "Paper B")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process Paper B: {str(e)}"}), 500

    # ── Category-specific retrieval from both papers ──
    evidence_a = {}   # category -> [text, ...]
    evidence_b = {}
    cat_sources_a = {}  # category -> [{page, snippet}, ...]
    cat_sources_b = {}

    early_chunks_a = _get_early_chunks(doc_id_a, max_chunks=2)
    early_chunks_b = _get_early_chunks(doc_id_b, max_chunks=2)

    for category, query in COMPARISON_QUERIES.items():
        k = 3 if category == "results" else 2

        # Paper A
        chunks_a = retrieve_with_metadata(doc_id_a, query, top_k=k)
        if category in ("research_problem", "contribution"):
            chunks_a = _dedup_chunks(early_chunks_a + chunks_a)
        evidence_a[category] = [c["text"] for c in chunks_a]
        srcs_a = []
        seen = set()
        for c in chunks_a:
            lbl = _page_label(c)
            if lbl and lbl not in seen:
                seen.add(lbl)
                srcs_a.append({"page": lbl, "snippet": c["text"][:100].strip() + "\u2026"})
        cat_sources_a[category] = srcs_a

        # Paper B
        chunks_b = retrieve_with_metadata(doc_id_b, query, top_k=k)
        if category in ("research_problem", "contribution"):
            chunks_b = _dedup_chunks(early_chunks_b + chunks_b)
        evidence_b[category] = [c["text"] for c in chunks_b]
        srcs_b = []
        seen = set()
        for c in chunks_b:
            lbl = _page_label(c)
            if lbl and lbl not in seen:
                seen.add(lbl)
                srcs_b.append({"page": lbl, "snippet": c["text"][:100].strip() + "\u2026"})
        cat_sources_b[category] = srcs_b

    # ── LLM comparison ──
    try:
        comparison_data = generate_comparison(evidence_a, evidence_b)
    except Exception as e:
        return jsonify({"error": f"Comparison generation failed: {str(e)}"}), 500

    # ── Assemble response with backend-derived sources ──
    result = {}
    for category in COMPARISON_QUERIES:
        cat = comparison_data.get(category, {})
        result[category] = {
            "paper_a": cat.get("paper_a", "Not specified in the paper.") if isinstance(cat, dict) else "Not specified in the paper.",
            "paper_b": cat.get("paper_b", "Not specified in the paper.") if isinstance(cat, dict) else "Not specified in the paper.",
            "sources_a": cat_sources_a.get(category, []),
            "sources_b": cat_sources_b.get(category, []),
        }

    return jsonify({
        "comparison": result,
        "key_differences": comparison_data.get("key_differences", []),
        "meta": {
            "paper_a_chunks": count_a,
            "paper_b_chunks": count_b,
        },
    })
