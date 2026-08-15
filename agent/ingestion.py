"""
Ingestion Layer
---------------
Parses PDF, DOCX, TXT, and MD files into a normalized list of
(text, metadata) page/section records. This is the entry point for
anything that goes into the knowledge base.

Design notes:
- We keep parsing separate from chunking. This module's job is only to
  get clean text OUT of a file format, with enough metadata (source
  filename, page number) to support citations and later deletion.
- Each source document gets a stable `doc_id` (derived from filename +
  content hash) so it can be found and removed as a unit later.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageRecord:
    """One page or section of raw extracted text, before chunking."""
    doc_id: str
    source_filename: str
    page_number: int | None  # None for formats without page concept (e.g. .md)
    text: str


@dataclass
class IngestResult:
    doc_id: str
    source_filename: str
    pages: list[PageRecord] = field(default_factory=list)
    total_chars: int = 0


def _make_doc_id(filepath: str, content_sample: str) -> str:
    """Stable ID from filename + a hash of content, so re-ingesting the
    same file twice is detectable, and the doc can be targeted for deletion."""
    h = hashlib.sha1()
    h.update(os.path.basename(filepath).encode("utf-8"))
    h.update(content_sample[:2000].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:12]


def ingest_pdf(filepath: str) -> IngestResult:
    import pymupdf  # PyMuPDF, current import name (fitz is the deprecated alias)

    doc = pymupdf.open(filepath)
    pages: list[PageRecord] = []
    sample = ""
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not sample:
            sample = text
        if text:
            pages.append(PageRecord(
                doc_id="",  # filled in after we compute doc_id
                source_filename=os.path.basename(filepath),
                page_number=i + 1,
                text=text,
            ))
    doc.close()

    doc_id = _make_doc_id(filepath, sample)
    for p in pages:
        p.doc_id = doc_id

    return IngestResult(
        doc_id=doc_id,
        source_filename=os.path.basename(filepath),
        pages=pages,
        total_chars=sum(len(p.text) for p in pages),
    )


def ingest_docx(filepath: str) -> IngestResult:
    import docx

    d = docx.Document(filepath)
    # DOCX has no native "page" concept without rendering, so we treat
    # the whole document as one logical unit and let the chunker split it.
    full_text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    doc_id = _make_doc_id(filepath, full_text)

    page = PageRecord(
        doc_id=doc_id,
        source_filename=os.path.basename(filepath),
        page_number=None,
        text=full_text,
    )
    return IngestResult(
        doc_id=doc_id,
        source_filename=os.path.basename(filepath),
        pages=[page],
        total_chars=len(full_text),
    )


def ingest_text(filepath: str) -> IngestResult:
    text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    doc_id = _make_doc_id(filepath, text)

    page = PageRecord(
        doc_id=doc_id,
        source_filename=os.path.basename(filepath),
        page_number=None,
        text=text,
    )
    return IngestResult(
        doc_id=doc_id,
        source_filename=os.path.basename(filepath),
        pages=[page],
        total_chars=len(text),
    )


_DISPATCH = {
    ".pdf": ingest_pdf,
    ".docx": ingest_docx,
    ".txt": ingest_text,
    ".md": ingest_text,
}


def ingest_file(filepath: str) -> IngestResult:
    """Single entry point: detect file type by extension and dispatch."""
    ext = Path(filepath).suffix.lower()
    if ext not in _DISPATCH:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(_DISPATCH.keys())}"
        )
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    result = _DISPATCH[ext](filepath)
    if result.total_chars == 0:
        raise ValueError(
            f"No extractable text found in {filepath}. "
            f"If this is a scanned/image-only PDF, OCR support isn't "
            f"in scope yet — see docs/ingestion.md."
        )
    return result
