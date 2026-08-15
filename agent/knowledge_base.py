"""
Knowledge Base Manager
-----------------------
Glue layer that ties ingestion -> chunking -> embedding -> vector store
into simple add_document() / remove_document() operations, and handles
persistence to disk so the CLI/UI don't need to know the internals.
"""

from __future__ import annotations

from pathlib import Path

from agent.chunking import chunk_document
from agent.embeddings import embed_texts
from agent.ingestion import ingest_file
from agent.vector_store import VectorStore

DEFAULT_INDEX_PATH = "data/knowledge_base/index"


class KnowledgeBase:
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        self.store = VectorStore.load(index_path)

    def add_document(self, filepath: str, chunk_size: int = 800, overlap: int = 150) -> dict:
        result = ingest_file(filepath)
        chunks = chunk_document(result, chunk_size=chunk_size, overlap=overlap)
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        self.store.add(chunks, vectors)
        self.save()
        return {
            "doc_id": result.doc_id,
            "source_filename": result.source_filename,
            "chunks_added": len(chunks),
        }

    def remove_document(self, doc_id: str) -> int:
        removed = self.store.delete_document(doc_id)
        self.save()
        return removed

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def save(self) -> None:
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.store.save(self.index_path)

    def __len__(self) -> int:
        return len(self.store)
