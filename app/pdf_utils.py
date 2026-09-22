"""PDF and plain-text extraction utilities."""

import io
import re
from typing import List, Dict


import pdfplumber


def extract_pages_from_pdf(file_bytes: bytes) -> List[Dict]:
    """
    Extract text from each page of a PDF, returning a list of dicts:
      [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Page numbers are 1-indexed to match human-readable page numbers.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append({"page": i, "text": page_text.strip()})
    return pages


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF as a single string.
    Kept for backward-compatibility; page metadata is lost here.
    Use extract_pages_from_pdf() when page metadata is needed.
    """
    pages = extract_pages_from_pdf(file_bytes)
    return "\n\n".join(p["text"] for p in pages)


def clean_text(text: str) -> str:
    """Basic text normalisation: collapse whitespace, remove null bytes."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
