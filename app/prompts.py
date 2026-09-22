"""Prompt templates and LLM-powered generation functions."""

from typing import List
from app.watsonx_client import get_watsonx_client


# ──────────────────────────────────────────────
# Summary generation
# ──────────────────────────────────────────────

SUMMARY_PROMPT = """\
You are an expert computer science researcher and educator. Given the following excerpts from a research paper, produce a structured summary.

IMPORTANT:
- Only use information that is explicitly present in the excerpts below.
- Do not invent results, numbers, or claims not mentioned in the text.
- If a section (e.g., experiments) is not covered by the excerpts, omit that section rather than guessing.

Paper Excerpts:
{context}

Write a comprehensive summary with the following sections (omit any section where the excerpts contain no relevant information):

## 📌 Core Contributions
List the 3-5 main contributions of the paper in bullet points.

## 🧠 Key Algorithms & Methods
Describe the primary algorithms, techniques, or methods proposed. Be specific.

## ⏱ Complexity Analysis
Summarize any time/space complexity claims, Big-O notation, or performance benchmarks mentioned.

## 🔬 Experimental Results
Highlight key experimental findings, datasets used, and comparisons with baselines.

## 💡 Takeaway
One paragraph plain-English summary suitable for a CS undergraduate student.

Be concise but technically accurate. Use markdown formatting.
"""


def generate_summary(context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = SUMMARY_PROMPT.format(context=context)
    client = get_watsonx_client()
    return client.generate(prompt, max_tokens=1500)


# ──────────────────────────────────────────────
# Q&A
# ──────────────────────────────────────────────

QA_PROMPT = """\
You are a helpful CS tutor explaining a research paper to a student.

Use ONLY the context excerpts below to answer the question. Follow these rules strictly:
- If the answer is clearly in the context, answer directly and cite the relevant part.
- If the context is partially relevant, answer what you can and note the limitation.
- If the answer is not in the context, say exactly: "I couldn't find that in the provided sections of the paper."
- Preserve exact numbers, percentages, and technical terms from the context — do not paraphrase them.
- Do NOT invent page numbers, section names, or experimental results.
- Do NOT make claims that go beyond what the context states.
- The paper may describe multiple distinct components, stages, modules, or sections. Treat each one separately. Never transfer a property, value, or behaviour described for one component to a different component, or generalise it to the whole system, unless the context explicitly says so.
- Preserve the exact scope of every technical claim: if the context says something applies to a specific part, your answer must say the same specific part — not a broader category.
- If the retrieved context is insufficient to determine a particular detail, say so explicitly rather than filling the gap with general background knowledge.

Context from the paper:
{context}

Question: {question}

Answer in simple, student-friendly language. If the answer involves an algorithm or concept, give a brief step-by-step explanation.
Keep your answer under 350 words unless a step-by-step explanation genuinely requires more.
"""


def answer_question(question: str, context_chunks: List[str], history: List[dict] = None) -> str:
    context = "\n\n---\n\n".join(context_chunks[:6])
    history_text = ""
    if history:
        recent = history[-4:]  # last 2 turns
        history_lines = []
        for turn in recent:
            role = "Student" if turn["role"] == "user" else "Tutor"
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines) + "\n\n"

    prompt = history_text + QA_PROMPT.format(context=context, question=question)
    client = get_watsonx_client()
    return client.generate(prompt, max_tokens=800)


# ──────────────────────────────────────────────
# Concept explainer
# ──────────────────────────────────────────────

CONCEPT_PROMPT_WITH_CONTEXT = """\
You are an expert CS educator. The student is reading a research paper and wants to understand the concept below.

Concept: {concept}

Relevant excerpts from the paper the student is reading:
{context}

Explain the concept using the paper's own treatment of it where possible. Your explanation should include:
1. **What it is** — a simple one-sentence definition.
2. **How it works** — step-by-step breakdown (use numbered steps).
3. **Analogy** — a real-world analogy to make it intuitive.
4. **In this paper** — how the paper specifically uses or discusses this concept (based on the excerpts above).
5. **Example** — a small concrete example or pseudocode snippet if relevant.

Format your response with clear markdown headers.
Preserve exact terminology and values from the paper excerpts. Do not invent paper-specific details.
"""

CONCEPT_PROMPT_GENERAL = """\
You are an expert CS educator. Explain the following concept in a way that a first-year CS student can understand.

Concept: {concept}

> ℹ️ **Note:** No uploaded paper context was found for this concept. This is a general explanation.

Your explanation should include:
1. **What it is** — a simple one-sentence definition.
2. **How it works** — step-by-step breakdown (use numbered steps).
3. **Analogy** — a real-world analogy to make it intuitive.
4. **Why it matters** — its importance in CS/AI/ML.
5. **Example** — a small concrete example or pseudocode snippet if relevant.

Format your response with clear markdown headers.
"""


def explain_concept(concept: str, context_chunks: List[str]) -> str:
    client = get_watsonx_client()

    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks[:4])
        prompt = CONCEPT_PROMPT_WITH_CONTEXT.format(concept=concept, context=context)
    else:
        prompt = CONCEPT_PROMPT_GENERAL.format(concept=concept)

    return client.generate(prompt, max_tokens=1000)


# ──────────────────────────────────────────────
# Paper comparison
# ──────────────────────────────────────────────

COMPARE_PROMPT = """\
You are a neutral research paper comparison assistant.

Below are retrieved excerpts from two research papers (Paper A and Paper B), organized by comparison category. Compare the two papers based ONLY on the provided excerpts.

STRICT RULES:
- Use ONLY information from the provided excerpts. Do not use external or general knowledge.
- Treat Paper A and Paper B as completely separate sources. Never mix or transfer facts between them (metrics, results, architecture, datasets, limitations, or contributions).
- Never attribute a property of one component to a different component, or generalise it to the whole paper, unless the excerpt explicitly supports that.
- Preserve exact numbers, percentages, metric names, model names, and technical terms from the excerpts.
- If information for a category is not present in a paper's excerpts, write exactly: "Not specified in the paper."
- Do not invent datasets, metrics, architectures, results, or limitations.
- Do not generate page numbers or source references — those are handled separately.
- Do not rank, score, or judge which paper is better, superior, or more reliable.
- Use neutral language: "Paper A uses...", "Paper B reports...", "Paper A proposes..."
- Keep each response concise (2-4 sentences per paper per category).
- If the two papers use different evaluation metrics, state that explicitly — do not pretend they are directly comparable.

CATEGORY-SPECIFIC INSTRUCTIONS:

RESEARCH PROBLEM — Look for the problem statement, motivation, or research gap in the excerpts, especially in abstract or introduction text. State clearly what problem the authors set out to solve and why existing approaches were insufficient, using only words from the excerpts.

MAIN CONTRIBUTION — State only novelties or contributions explicitly claimed and supported by the excerpts (typically from abstract/intro). If the contribution is not found in the provided evidence, write: "Not specified in the retrieved evidence." Do NOT invent contributions.

ARCHITECTURE / MODEL — Clearly distinguish between model architecture, pre-training procedure, training procedure, and task-specific adaptation. Do NOT describe a paper as modifying model architecture when the excerpts only describe modifications to training or pre-training procedures.

LIMITATIONS / CONSTRAINTS — Only report a limitation when the retrieved excerpts clearly show the authors explicitly presenting it as a limitation, drawback, constraint, trade-off, or weakness of their own work. Do NOT infer a limitation merely because something is expensive, large, slow, resource-intensive, or requires more data — unless the paper itself explicitly frames it as a limitation. If no explicit limitation is stated in the excerpts, write: "Not explicitly stated in the paper."

KEY RESULTS — Strongly prioritize numerical evidence. When the excerpts contain specific numerical values (accuracy, F1, EM, BLEU, ROUGE, percentage improvement, benchmark scores, dataset size, parameter count, or any other quantitative result), report those exact numbers. Do not round, combine, or calculate new numbers. Do not substitute generic phrases like "achieves state-of-the-art" when specific numbers are available. Do NOT rank papers or declare a winner.

EVALUATION METRICS — When the excerpts mention which benchmark or task a metric applies to, include that association (e.g., "accuracy on GLUE", "F1 and EM on SQuAD"). Only include associations explicitly stated in the excerpts.

KEY DIFFERENCES — Derive differences strictly from the verified category results above. Do NOT introduce new factual claims not established in the category evidence. If papers differ in pre-training, training, architecture, datasets, or evaluation benchmarks, state those differences accurately. Never claim a paper modifies architecture if evidence only supports pre-training changes. Do NOT rank papers or say one is better.
"""


COMPARE_JSON_INSTRUCTIONS = '''
Respond with ONLY a valid JSON object. No markdown fences, no explanation, no text before or after the JSON. Use this exact structure:

{"research_problem": {"paper_a": "...", "paper_b": "..."}, "contribution": {"paper_a": "...", "paper_b": "..."}, "method": {"paper_a": "...", "paper_b": "..."}, "architecture": {"paper_a": "...", "paper_b": "..."}, "dataset": {"paper_a": "...", "paper_b": "..."}, "metrics": {"paper_a": "...", "paper_b": "..."}, "results": {"paper_a": "...", "paper_b": "..."}, "limitations": {"paper_a": "...", "paper_b": "..."}, "key_differences": ["difference 1", "difference 2", "difference 3"]}
'''

_CATEGORY_LABELS = {
    "research_problem": "RESEARCH PROBLEM",
    "contribution": "MAIN CONTRIBUTION",
    "method": "METHOD / APPROACH",
    "architecture": "ARCHITECTURE / MODEL",
    "dataset": "DATASET / EXPERIMENTAL SETUP",
    "metrics": "EVALUATION METRICS",
    "results": "KEY RESULTS",
    "limitations": "LIMITATIONS / CONSTRAINTS",
}


def _trim_words(text: str, max_words: int = 250) -> str:
    """Trim text to approximately max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def _build_evidence_block(evidence_a: dict, evidence_b: dict) -> str:
    """Format retrieved evidence from both papers into a structured text block."""
    sections = []
    for category, label in _CATEGORY_LABELS.items():
        lines = [f"=== {label} ==="]

        texts_a = evidence_a.get(category, [])
        if texts_a:
            lines.append("PAPER A excerpts:")
            for t in texts_a:
                lines.append(_trim_words(t, 250))
                lines.append("---")
        else:
            lines.append("PAPER A excerpts: (none retrieved)")

        texts_b = evidence_b.get(category, [])
        if texts_b:
            lines.append("PAPER B excerpts:")
            for t in texts_b:
                lines.append(_trim_words(t, 250))
                lines.append("---")
        else:
            lines.append("PAPER B excerpts: (none retrieved)")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _parse_comparison_json(raw_text: str) -> dict:
    """Parse the LLM's JSON output with robust fallback."""
    import json
    import re

    # Try extracting JSON from code fences
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try direct JSON extraction (first { to last })
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text in the first category so the user sees something
    return {
        "research_problem": {
            "paper_a": raw_text,
            "paper_b": "Structured comparison could not be parsed.",
        },
        "key_differences": [],
    }


def generate_comparison(evidence_a: dict, evidence_b: dict) -> dict:
    """
    Generate a structured paper comparison from category-specific evidence.
    evidence_a / evidence_b: {category: [text, ...]}
    Returns parsed dict with per-category paper_a/paper_b strings + key_differences list.
    """
    evidence_block = _build_evidence_block(evidence_a, evidence_b)
    # Concatenate (not .format()) to avoid escaping issues with JSON braces
    prompt = COMPARE_PROMPT + evidence_block + "\n\n" + COMPARE_JSON_INSTRUCTIONS

    client = get_watsonx_client()
    raw = client.generate(prompt, max_tokens=2500)
    return _parse_comparison_json(raw)

