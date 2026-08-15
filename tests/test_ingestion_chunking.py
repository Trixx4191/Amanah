import pymupdf
import pytest

from agent.chunking import chunk_document, chunk_page
from agent.ingestion import PageRecord, ingest_file

SAMPLE_TEXT = """Paragraph one has a few sentences. It talks about local agents. They run without the cloud.

Paragraph two is about retrieval. It explains chunking and embeddings. These are stored in a vector index.

Paragraph three wraps up. It mentions reasoning loops. And memory management too.
"""


@pytest.fixture
def sample_txt_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT)
    return str(p)


@pytest.fixture
def sample_pdf_file(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT, fontsize=11)
    p = tmp_path / "sample.pdf"
    doc.save(str(p))
    return str(p)


def test_ingest_txt_extracts_all_text(sample_txt_file):
    result = ingest_file(sample_txt_file)
    assert result.total_chars > 0
    assert len(result.pages) == 1
    assert "Paragraph three" in result.pages[0].text
    assert result.pages[0].page_number is None  # txt has no page concept


def test_ingest_pdf_extracts_text_with_page_numbers(sample_pdf_file):
    result = ingest_file(sample_pdf_file)
    assert result.total_chars > 0
    assert result.pages[0].page_number == 1


def test_ingest_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "file.xyz"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        ingest_file(str(bad))


def test_ingest_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ingest_file("/tmp/does_not_exist_12345.txt")


def test_chunking_preserves_all_content_roughly(sample_txt_file):
    result = ingest_file(sample_txt_file)
    chunks = chunk_document(result, chunk_size=120, overlap=20)
    assert len(chunks) > 1  # small chunk_size should force multiple chunks

    # every chunk traces back to the source doc
    for c in chunks:
        assert c.doc_id == result.doc_id
        assert c.source_filename == "sample.txt"

    # key content survives chunking somewhere
    joined = " ".join(c.text for c in chunks)
    assert "reasoning loops" in joined
    assert "vector index" in joined


def test_chunk_ids_are_unique_and_sequential(sample_txt_file):
    result = ingest_file(sample_txt_file)
    chunks = chunk_document(result, chunk_size=100, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # all unique
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_page_handles_oversized_paragraph():
    # A single paragraph with no breaks, longer than chunk_size,
    # must still get split (via sentence fallback / hard cut).
    long_text = "This is one sentence. " * 50  # ~1150 chars, no paragraph breaks
    page = PageRecord(doc_id="d1", source_filename="f.txt", page_number=None, text=long_text)
    chunks = chunk_page(page, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 30 for c in chunks)  # rough bound, overlap included
