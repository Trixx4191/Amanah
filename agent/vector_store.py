"""
Vector Store (v1: from-scratch, numpy-based)
---------------------------------------------
A flat vector index: every chunk's embedding is kept in memory as a row
in a matrix, and search is a single matrix-vector dot product (since
embeddings are L2-normalized, dot product == cosine similarity).

This is intentionally simple. For a knowledge base of thousands to low
tens-of-thousands of chunks, a brute-force scan on CPU is fast (numpy's
BLAS backend handles this in milliseconds). If the knowledge base grows
much larger than that, swap this module for a FAISS-backed index behind
the same `VectorStore` interface — nothing else in the codebase would
need to change.

Persistence: vectors go in a .npy file, metadata in a .json file
alongside it, so the knowledge base survives restarts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int | None
    text: str
    score: float  # cosine similarity, higher is more relevant


class VectorStore:
    def __init__(self):
        self._vectors: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self._metadata: list[dict] = []  # parallel to _vectors rows

    # -- building the index -------------------------------------------------

    def add(self, chunks: list, vectors: np.ndarray) -> None:
        """chunks: list of agent.chunking.Chunk objects.
        vectors: (N, 384) array aligned 1:1 with chunks."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must be the same length")
        if vectors.shape[0] == 0:
            return

        self._vectors = np.vstack([self._vectors, vectors])
        for c in chunks:
            self._metadata.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "text": c.text,
            })

    # -- querying -------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self._vectors.shape[0] == 0:
            return []

        scores = self._vectors @ query_vector  # cosine sim, vectors are normalized
        top_k = min(top_k, len(scores))
        # argpartition for O(n) top-k selection, then sort just those k
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for i in top_idx:
            meta = self._metadata[i]
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                source_filename=meta["source_filename"],
                page_number=meta["page_number"],
                text=meta["text"],
                score=float(scores[i]),
            ))
        return results

    # -- knowledge base management --------------------------------------------

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to `doc_id`. Returns count removed."""
        keep_mask = np.array(
            [m["doc_id"] != doc_id for m in self._metadata], dtype=bool
        )
        removed = int((~keep_mask).sum())
        self._vectors = self._vectors[keep_mask]
        self._metadata = [m for m, keep in zip(self._metadata, keep_mask) if keep]
        return removed

    def list_documents(self) -> list[dict]:
        """Summary of what's in the knowledge base, grouped by source document."""
        seen: dict[str, dict] = {}
        for m in self._metadata:
            doc_id = m["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source_filename": m["source_filename"],
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1
        return list(seen.values())

    def __len__(self) -> int:
        return self._vectors.shape[0]

    # -- persistence ------------------------------------------------------------

    def save(self, path: str) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.save(base.with_suffix(".vectors.npy"), self._vectors)
        with open(base.with_suffix(".meta.json"), "w") as f:
            json.dump(self._metadata, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        base = Path(path)
        store = cls()
        vec_path = base.with_suffix(".vectors.npy")
        meta_path = base.with_suffix(".meta.json")
        if vec_path.exists() and meta_path.exists():
            store._vectors = np.load(vec_path)
            with open(meta_path) as f:
                store._metadata = json.load(f)
        return store
