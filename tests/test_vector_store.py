"""
Tests for the from-scratch vector store: add, search ranking, deletion,
and save/load persistence. Uses synthetic embeddings (no model needed)
so this suite runs fast and offline.
"""
import numpy as np
import pytest

from agent.chunking import Chunk
from agent.vector_store import VectorStore


def make_chunk(doc_id, idx, text, source="test.pdf"):
    return Chunk(chunk_id=f"{doc_id}-{idx}", doc_id=doc_id, source_filename=source,
                 page_number=idx + 1, chunk_index=idx, text=text)


def make_clustered_vectors(base, n, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    vecs = np.array([base + noise * rng.standard_normal(384) for _ in range(n)],
                     dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def two_doc_store():
    rng = np.random.default_rng(42)
    base_a = rng.standard_normal(384).astype(np.float32)
    base_a /= np.linalg.norm(base_a)
    base_b = rng.standard_normal(384).astype(np.float32)
    base_b -= base_b.dot(base_a) * base_a
    base_b /= np.linalg.norm(base_b)

    store = VectorStore()
    chunks_a = [make_chunk("docA", i, f"Chunk {i} about topic A") for i in range(3)]
    store.add(chunks_a, make_clustered_vectors(base_a, 3, seed=1))

    chunks_b = [make_chunk("docB", i, f"Chunk {i} about topic B", source="other.pdf")
                for i in range(2)]
    store.add(chunks_b, make_clustered_vectors(base_b, 2, seed=2))

    return store, base_a, base_b


def test_add_and_count(two_doc_store):
    store, _, _ = two_doc_store
    assert len(store) == 5
    docs = {d["doc_id"]: d["chunk_count"] for d in store.list_documents()}
    assert docs == {"docA": 3, "docB": 2}


def test_search_ranks_relevant_document_first(two_doc_store):
    store, base_a, _ = two_doc_store
    query = base_a + 0.02 * np.random.default_rng(3).standard_normal(384).astype(np.float32)
    query /= np.linalg.norm(query)

    results = store.search(query, top_k=3)
    assert len(results) == 3
    assert all(r.doc_id == "docA" for r in results)
    # scores should be sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_delete_document_removes_only_that_document(two_doc_store):
    store, _, _ = two_doc_store
    removed = store.delete_document("docA")
    assert removed == 3
    assert len(store) == 2
    remaining_docs = {d["doc_id"] for d in store.list_documents()}
    assert remaining_docs == {"docB"}


def test_save_and_load_roundtrip(two_doc_store, tmp_path):
    store, base_a, _ = two_doc_store
    index_path = str(tmp_path / "index")
    store.save(index_path)

    reloaded = VectorStore.load(index_path)
    assert len(reloaded) == len(store)

    query = base_a.copy()
    original_results = store.search(query, top_k=2)
    reloaded_results = reloaded.search(query, top_k=2)
    assert [r.chunk_id for r in original_results] == [r.chunk_id for r in reloaded_results]


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore()
    results = store.search(np.zeros(384, dtype=np.float32), top_k=5)
    assert results == []


def test_add_mismatched_lengths_raises():
    store = VectorStore()
    chunks = [make_chunk("docA", 0, "text")]
    vectors = np.zeros((2, 384), dtype=np.float32)  # wrong length on purpose
    with pytest.raises(ValueError):
        store.add(chunks, vectors)
