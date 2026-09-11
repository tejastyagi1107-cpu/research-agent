"""Prompt templates and LLM-powered generation functions."""

from typing import List
from app.watsonx_client import get_watsonx_client


# ──────────────────────────────────────────────
# Summary generation
# ──────────────────────────────────────────────

SUMMARY_PROMPT = """\
You are an expert computer science researcher and educator. Given the following excerpts from a research paper, produce a structured summary.

Paper Excerpts:
{context}

Write a comprehensive summary with the following sections:

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
    context = "\n\n---\n\n".join(context_chunks[:10])
    prompt = SUMMARY_PROMPT.format(context=context)
    client = get_watsonx_client()
    return client.generate(prompt, max_tokens=1500)


# ──────────────────────────────────────────────
# Q&A
# ──────────────────────────────────────────────

QA_PROMPT = """\
You are a friendly CS tutor helping a student understand a research paper. Answer the question below using ONLY the provided context excerpts. 
If the answer is not in the context, say "I couldn't find that in the paper."

Context from the paper:
{context}

Question: {question}

Instructions:
- Answer in simple, beginner-friendly language.
- If the answer involves an algorithm or concept, give a brief step-by-step explanation.
- At the end of your answer, add a "📍 Source" line citing the relevant section/excerpt (first 15 words of the chunk used).
- Keep your answer under 300 words unless a step-by-step explanation requires more.
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

CONCEPT_PROMPT = """\
You are an expert CS educator. Explain the following concept from a research paper in a way that a first-year CS student can understand.

Concept: {concept}

Context from the paper (if available):
{context}

Your explanation should include:
1. **What it is** — a simple one-sentence definition.
2. **How it works** — step-by-step breakdown (use numbered steps).
3. **Analogy** — a real-world analogy to make it intuitive.
4. **Why it matters** — its importance in CS/AI/ML.
5. **Example** — a small concrete example or pseudocode snippet if relevant.

Format your response with clear markdown headers.
"""


def explain_concept(concept: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks[:4]) if context_chunks else "No additional context provided."
    prompt = CONCEPT_PROMPT.format(concept=concept, context=context)
    client = get_watsonx_client()
    return client.generate(prompt, max_tokens=1000)
