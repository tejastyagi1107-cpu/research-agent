import os
import re
import uuid
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

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
# Text chunking
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
        self.chunks: List[str] = []

    def build(self, text: str) -> int:
        """Chunk text, embed, and persist."""
        embedder = _get_embedder()
        self.chunks = chunk_text(text)
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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
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
    """Ingest text into vector store. Returns (doc_id, chunk_count)."""
    doc_id = create_doc_id(text)
    store = VectorStore(doc_id)
    # Re-build only if not already present
    if not store.load():
        chunk_count = store.build(text)
    else:
        chunk_count = len(store.chunks)
    return doc_id, chunk_count


def retrieve(doc_id: str, query: str, top_k: int = 5) -> List[str]:
    """Retrieve top-k relevant chunks for a query."""
    store = VectorStore(doc_id)
    results = store.search(query, top_k=top_k)
    return [chunk for chunk, _ in results]
