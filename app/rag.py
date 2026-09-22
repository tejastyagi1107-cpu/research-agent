import os
import re
import uuid
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = Path(os.environ.get("VECTOR_STORE_DIR", "data/vector_stores"))
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ──────────────────────────────────────────────
# Text chunking — plain text (no page metadata)
# ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks of ~chunk_size words."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────
# Page-aware chunking
# ──────────────────────────────────────────────

def chunk_pages(pages: List[Dict], chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Chunk a list of page dicts into overlapping word-level chunks,
    preserving page_start and page_end for each chunk.

    Each page dict must have: {"page": int, "text": str}
    Returns list of dicts:
      {"chunk_id": str, "text": str, "page_start": int, "page_end": int}
    """
    # Build a flat word list with per-word page annotations
    word_page_pairs: List[Tuple[str, int]] = []
    for p in pages:
        page_num = p["page"]
        words = p["text"].split()
        for w in words:
            word_page_pairs.append((w, page_num))

    chunks: List[Dict] = []
    start = 0
    chunk_idx = 0
    total = len(word_page_pairs)

    while start < total:
        end = min(start + chunk_size, total)
        slice_ = word_page_pairs[start:end]
        text = " ".join(w for w, _ in slice_)
        page_start = slice_[0][1]
        page_end = slice_[-1][1]

        if text.strip():
            chunks.append({
                "chunk_id": f"chunk_{chunk_idx:04d}",
                "text": text,
                "page_start": page_start,
                "page_end": page_end,
            })
            chunk_idx += 1

        if end == total:
            break
        start += chunk_size - overlap

    return chunks


# ──────────────────────────────────────────────
# Vector store management
# ──────────────────────────────────────────────

class VectorStore:
    """FAISS-backed vector store for a single document."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.store_path = VECTOR_STORE_DIR / doc_id
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.store_path / "index.faiss"
        self.chunks_file = self.store_path / "chunks.pkl"
        self.index: Optional[faiss.IndexFlatL2] = None
        # Each element is either a plain str (legacy) or a dict with metadata
        self.chunks: List = []

    def build_from_chunk_dicts(self, chunk_dicts: List[Dict]) -> int:
        """Build index from page-aware chunk dicts. Preferred path."""
        embedder = _get_embedder()
        self.chunks = chunk_dicts  # list of dicts
        if not self.chunks:
            return 0
        texts = [c["text"] for c in self.chunks]
        vectors = embedder.encode(texts, show_progress_bar=False).astype("float32")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        return len(self.chunks)

    def build(self, text: str) -> int:
        """Fallback: chunk plain text and build index (no page metadata)."""
        embedder = _get_embedder()
        plain_chunks = chunk_text(text)
        self.chunks = plain_chunks
        if not self.chunks:
            return 0
        vectors = embedder.encode(self.chunks, show_progress_bar=False).astype("float32")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        return len(self.chunks)

    def load(self) -> bool:
        if not self.index_file.exists() or not self.chunks_file.exists():
            return False
        self.index = faiss.read_index(str(self.index_file))
        with open(self.chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        return True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Returns list of (chunk, distance) tuples.
        chunk is either a dict (with metadata) or a str (legacy).
        """
        if self.index is None and not self.load():
            return []
        embedder = _get_embedder()
        q_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
        distances, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(dist)))
        return results


# ──────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────

def create_doc_id(text: str) -> str:
    """Deterministic doc ID based on content hash."""
    return hashlib.md5(text[:2000].encode()).hexdigest()[:12]


def ingest_document(text: str) -> Tuple[str, int]:
    """Ingest plain text into vector store (no page metadata). Returns (doc_id, chunk_count)."""
    doc_id = create_doc_id(text)
    store = VectorStore(doc_id)
    if not store.load():
        chunk_count = store.build(text)
    else:
        chunk_count = len(store.chunks)
    return doc_id, chunk_count


def ingest_document_pages(pages: List[Dict]) -> Tuple[str, int]:
    """
    Ingest page-aware content into vector store. Preferred over ingest_document().
    pages: [{"page": int, "text": str}, ...]
    Returns (doc_id, chunk_count).
    """
    combined_text = " ".join(p["text"] for p in pages)
    doc_id = create_doc_id(combined_text)
    store = VectorStore(doc_id)
    if not store.load():
        chunk_dicts = chunk_pages(pages)
        chunk_count = store.build_from_chunk_dicts(chunk_dicts)
    else:
        chunk_count = len(store.chunks)
    return doc_id, chunk_count


def retrieve(doc_id: str, query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve top-k relevant chunk texts for a query.
    Legacy interface — returns only text strings.
    """
    store = VectorStore(doc_id)
    results = store.search(query, top_k=top_k)
    return [_chunk_text(chunk) for chunk, _ in results]


def retrieve_with_metadata(doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k relevant chunks with full metadata.
    Returns list of dicts: {"text": str, "page_start": int, "page_end": int, "chunk_id": str}
    For legacy str chunks, page info will be None.
    """
    store = VectorStore(doc_id)
    results = store.search(query, top_k=top_k)
    output = []
    for chunk, dist in results:
        if isinstance(chunk, dict):
            output.append({
                "text": chunk["text"],
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "chunk_id": chunk.get("chunk_id"),
            })
        else:
            # Legacy plain string chunk — no page metadata available
            output.append({
                "text": chunk,
                "page_start": None,
                "page_end": None,
                "chunk_id": None,
            })
    return output


def _chunk_text(chunk) -> str:
    """Extract text string from either a dict chunk or a plain str chunk."""
    if isinstance(chunk, dict):
        return chunk["text"]
    return chunk


def get_all_chunks(doc_id: str) -> List:
    """Load and return all chunks for a document."""
    store = VectorStore(doc_id)
    if not store.load():
        return []
    return store.chunks
