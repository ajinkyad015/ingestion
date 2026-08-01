from __future__ import annotations

from pathlib import Path

import pytest

from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.entities.document import Document, NormalizedDocument, ParsedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document() -> Document:
    return Document(
        source_path=Path("/data/report.pdf"),
        extension=".pdf",
        source_hash="abc123" * 8 + "ab",  # 50-char hex-like string
        size_bytes=10_240,
    )


def _make_parsed(doc: Document) -> ParsedDocument:
    return ParsedDocument(
        source=doc,
        text="Introduction\n\nThis is the body.",
        title="Annual Report",
        num_pages=3,
    )


def _make_normalized(parsed: ParsedDocument) -> NormalizedDocument:
    return NormalizedDocument(
        source=parsed,
        text="Introduction\n\nThis is the body.",
        sections=(("Introduction", "This is the body."),),
    )


def _make_chunk_metadata(doc: Document, *, index: int = 0, total: int = 1) -> ChunkMetadata:
    return ChunkMetadata(
        document_hash=doc.source_hash,
        source_path=doc.source_path,
        chunk_index=index,
        total_chunks=total,
        heading="Introduction",
        page_number=1,
        char_start=0,
        char_end=32,
    )


def _make_chunk(doc: Document, *, index: int = 0, total: int = 1) -> Chunk:
    meta = _make_chunk_metadata(doc, index=index, total=total)
    return Chunk(
        chunk_id=f"{doc.source_hash}_{index:06d}",
        text="Introduction\n\nThis is the body.",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_fields_are_accessible(self) -> None:
        doc = _make_document()
        assert doc.source_path == Path("/data/report.pdf")
        assert doc.extension == ".pdf"
        assert doc.size_bytes == 10_240

    def test_is_frozen(self) -> None:
        doc = _make_document()
        with pytest.raises(AttributeError):
            doc.extension = ".txt"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        doc_a = _make_document()
        doc_b = _make_document()
        assert doc_a == doc_b

    def test_inequality_on_different_hash(self) -> None:
        doc_a = _make_document()
        doc_b = Document(
            source_path=doc_a.source_path,
            extension=doc_a.extension,
            source_hash="deadbeef" * 8,
            size_bytes=doc_a.size_bytes,
        )
        assert doc_a != doc_b


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------


class TestParsedDocument:
    def test_preserves_source_reference(self) -> None:
        doc = _make_document()
        parsed = _make_parsed(doc)
        assert parsed.source is doc

    def test_num_pages_zero_for_unpaginated(self) -> None:
        doc = _make_document()
        parsed = ParsedDocument(source=doc, text="hello", title="", num_pages=0)
        assert parsed.num_pages == 0

    def test_is_frozen(self) -> None:
        doc = _make_document()
        parsed = _make_parsed(doc)
        with pytest.raises(AttributeError):
            parsed.title = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NormalizedDocument
# ---------------------------------------------------------------------------


class TestNormalizedDocument:
    def test_preserves_source_reference(self) -> None:
        doc = _make_document()
        parsed = _make_parsed(doc)
        normalized = _make_normalized(parsed)
        assert normalized.source is parsed

    def test_sections_default_to_empty_tuple(self) -> None:
        doc = _make_document()
        parsed = _make_parsed(doc)
        normalized = NormalizedDocument(source=parsed, text="hello")
        assert normalized.sections == ()

    def test_sections_preserved(self) -> None:
        doc = _make_document()
        parsed = _make_parsed(doc)
        sections = (("Intro", "body text"), ("Conclusion", "end"))
        normalized = NormalizedDocument(source=parsed, text="body", sections=sections)
        assert normalized.sections == sections

    def test_is_frozen(self) -> None:
        doc = _make_document()
        parsed = _make_parsed(doc)
        normalized = _make_normalized(parsed)
        with pytest.raises(AttributeError):
            normalized.text = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_fields_accessible(self) -> None:
        doc = _make_document()
        meta = _make_chunk_metadata(doc, index=2, total=10)
        assert meta.chunk_index == 2
        assert meta.total_chunks == 10
        assert meta.heading == "Introduction"
        assert meta.page_number == 1
        assert meta.char_start == 0
        assert meta.char_end == 32

    def test_source_path_matches_document(self) -> None:
        doc = _make_document()
        meta = _make_chunk_metadata(doc)
        assert meta.source_path == doc.source_path

    def test_document_hash_matches_document(self) -> None:
        doc = _make_document()
        meta = _make_chunk_metadata(doc)
        assert meta.document_hash == doc.source_hash

    def test_is_frozen(self) -> None:
        doc = _make_document()
        meta = _make_chunk_metadata(doc)
        with pytest.raises(AttributeError):
            meta.chunk_index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_chunk_id_format(self) -> None:
        doc = _make_document()
        chunk = _make_chunk(doc, index=7, total=20)
        assert chunk.chunk_id == f"{doc.source_hash}_000007"

    def test_text_preserved(self) -> None:
        doc = _make_document()
        chunk = _make_chunk(doc)
        assert chunk.text == "Introduction\n\nThis is the body."

    def test_is_frozen(self) -> None:
        doc = _make_document()
        chunk = _make_chunk(doc)
        with pytest.raises(AttributeError):
            chunk.text = "modified"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        doc = _make_document()
        chunk_a = _make_chunk(doc)
        chunk_b = _make_chunk(doc)
        assert chunk_a == chunk_b


# ---------------------------------------------------------------------------
# EmbeddedChunk
# ---------------------------------------------------------------------------


class TestEmbeddedChunk:
    def test_embedding_preserved(self) -> None:
        doc = _make_document()
        chunk = _make_chunk(doc)
        embedding = tuple([0.1, 0.2, 0.3])
        ec = EmbeddedChunk(chunk=chunk, embedding=embedding)
        assert ec.embedding == (0.1, 0.2, 0.3)

    def test_chunk_reference(self) -> None:
        doc = _make_document()
        chunk = _make_chunk(doc)
        ec = EmbeddedChunk(chunk=chunk, embedding=(0.0,))
        assert ec.chunk is chunk

    def test_is_frozen(self) -> None:
        doc = _make_document()
        chunk = _make_chunk(doc)
        ec = EmbeddedChunk(chunk=chunk, embedding=(0.0,))
        with pytest.raises(AttributeError):
            ec.embedding = (1.0,)  # type: ignore[misc]
