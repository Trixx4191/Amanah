"""
Embedding Layer
---------------
Wraps a local sentence-embedding model. This is the only place in the
codebase that knows about `sentence-transformers` — everything else
just calls `embed_texts()` and gets back vectors.

Model: all-MiniLM-L6-v2
- 384-dimensional embeddings, ~90MB, runs fast on CPU
- Good general-purpose semantic similarity for English text
- Downloaded once from Hugging Face on first run, then cached locally
  under ~/.cache/huggingface — after that, fully offline.

IMPORTANT (see docs/setup.md): install the CPU-only build of PyTorch
before installing sentence-transformers, or pip will pull the full
CUDA-enabled build (multiple GB, unnecessary on a CPU-only machine):

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # lazy-loaded singleton, loading the model is the slow part


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns an (N, 384) float32 array,
    L2-normalized so dot product == cosine similarity."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so we can use dot product as cosine sim
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a (384,) vector."""
    return embed_texts([query])[0]
