"""
Chunking Engine
---------------
Splits raw page/section text into overlapping, semantically coherent
chunks ready for embedding.

Design notes:
- We split on paragraph boundaries first, then sentence boundaries if a
  paragraph is still too big, and only fall back to a hard character
  cut as a last resort. This avoids severing a sentence mid-thought,
  which hurts retrieval quality more than people expect.
- Overlap between consecutive chunks preserves context across the seam,
  so a fact split across a chunk boundary is still findable.
- Every chunk keeps a reference back to its source document, page, and
  a chunk-local index — this is what powers citations and deletion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.ingestion import IngestResult, PageRecord

DEFAULT_CHUNK_SIZE = 800       # characters, not tokens (simple + good enough to start)
DEFAULT_CHUNK_OVERLAP = 150

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str          # doc_id + local index, globally unique
    doc_id: str
    source_filename: str
    page_number: int | None
    chunk_index: int       # position within the document
    text: str


def _split_into_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _pack_units(units: list[str], max_size: int, overlap: int) -> list[str]:
    """Greedily pack small text units (sentences/paragraphs) into chunks
    close to max_size, carrying `overlap` characters of trailing context
    forward into the next chunk."""
    chunks: list[str] = []
    current = ""

    for unit in units:
        # A single unit longer than max_size: hard-split it as a last resort.
        if len(unit) > max_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(unit), max_size - overlap):
                chunks.append(unit[i:i + max_size].strip())
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= max_size:
            current = candidate
        else:
            chunks.append(current.strip())
            # start next chunk with overlap tail of the previous one
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {unit}".strip()

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def chunk_page(page: PageRecord, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    paragraphs = _split_into_paragraphs(page.text)
    if not paragraphs:
        return []

    # If paragraphs are already small, pack them directly.
    # If any paragraph alone exceeds chunk_size, break it into sentences first.
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    return _pack_units(units, chunk_size, overlap)


def chunk_document(result: IngestResult, chunk_size: int = DEFAULT_CHUNK_SIZE,
                    overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Chunk]:
    """Turn an IngestResult (one or more pages of raw text) into a flat
    list of Chunk objects ready for embedding."""
    chunks: list[Chunk] = []
    running_index = 0

    for page in result.pages:
        page_chunks = chunk_page(page, chunk_size, overlap)
        for text in page_chunks:
            chunks.append(Chunk(
                chunk_id=f"{result.doc_id}-{running_index}",
                doc_id=result.doc_id,
                source_filename=result.source_filename,
                page_number=page.page_number,
                chunk_index=running_index,
                text=text,
            ))
            running_index += 1

    return chunks
