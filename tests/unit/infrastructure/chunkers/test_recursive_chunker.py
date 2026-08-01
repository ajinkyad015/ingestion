from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.document import (
    Document,
    NormalizedDocument,
    ParsedDocument,
)
from rag_ingestion.domain.protocols.chunker import Chunker
from rag_ingestion.infrastructure.chunkers.recursive import RecursiveChunker


@pytest.fixture
def sample_parsed_document() -> ParsedDocument:
    doc = Document(
        source_path=Path("/mock/path/test.txt"),
        extension=".txt",
        source_hash="abcdef123456",
        size_bytes=1024,
    )
    return ParsedDocument(
        source=doc,
        text="Mock text",
        title="Mock Title",
        num_pages=1,
    )


def create_normalized_document(parsed: ParsedDocument, text: str) -> NormalizedDocument:
    return NormalizedDocument(
        source=parsed,
        text=text,
        sections=(),
    )


def test_protocol_conformance():
    settings = Settings(CHUNK_SIZE=512, CHUNK_OVERLAP=64)
    chunker = RecursiveChunker(settings)
    assert isinstance(chunker, Chunker)


def test_empty_document(sample_parsed_document):
    settings = Settings(CHUNK_SIZE=512, CHUNK_OVERLAP=64)
    chunker = RecursiveChunker(settings)
    
    doc = create_normalized_document(sample_parsed_document, "")
    chunks = chunker.chunk(doc)
    
    assert len(chunks) == 0


def test_document_smaller_than_chunk_size(sample_parsed_document):
    settings = Settings(CHUNK_SIZE=512, CHUNK_OVERLAP=64)
    chunker = RecursiveChunker(settings)
    
    text = "This is a short document."
    doc = create_normalized_document(sample_parsed_document, text)
    chunks = chunker.chunk(doc)
    
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "abcdef123456_000000"
    assert chunks[0].metadata.document_hash == "abcdef123456"
    assert chunks[0].metadata.source_path == Path("/mock/path/test.txt")
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.total_chunks == 1
    assert chunks[0].metadata.char_start == 0
    assert chunks[0].metadata.char_end == len(text)
    assert chunks[0].metadata.heading == ""


def test_section_hierarchy_preservation_and_deterministic_ids(sample_parsed_document):
    settings = Settings(CHUNK_SIZE=64, CHUNK_OVERLAP=16)
    chunker = RecursiveChunker(settings)
    
    text = (
        "Preamble text here.\n"
        "## Section 1\n"
        "This is the first section.\n"
        "It is very nice.\n"
        "## Section 2\n"
        "This is the second section.\n"
        "It has some more text."
    )
    doc = create_normalized_document(sample_parsed_document, text)
    chunks = chunker.chunk(doc)
    
    # Check that chunks are ordered and IDs are deterministic
    assert len(chunks) > 0
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"abcdef123456_{i:06d}"
        assert chunk.metadata.chunk_index == i
        assert chunk.metadata.total_chunks == len(chunks)
        # Verify text offset is correct
        start = chunk.metadata.char_start
        end = chunk.metadata.char_end
        assert chunk.text == text[start:end]

    # First section should have empty heading
    assert chunks[0].metadata.heading == ""
    
    # We should have chunks for Section 1 and Section 2 with corresponding headings
    headings = [c.metadata.heading for c in chunks]
    assert "Section 1" in headings
    assert "Section 2" in headings


def test_recursive_split_oversized_section(sample_parsed_document):
    settings = Settings(CHUNK_SIZE=64, CHUNK_OVERLAP=16)
    chunker = RecursiveChunker(settings)
    
    # Section block is larger than 64 characters
    text = "## Big Section\n" + "Word " * 20
    doc = create_normalized_document(sample_parsed_document, text)
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 1
    
    # Reconstruct text by ensuring every character is covered
    reconstructed_text = ""
    for chunk in chunks:
        assert len(chunk.text) <= 64
        assert chunk.metadata.heading == "Big Section"
        # Make sure the chunk text is exactly in the original document
        start = chunk.metadata.char_start
        end = chunk.metadata.char_end
        assert chunk.text == text[start:end]


def test_deterministic_chunk_boundaries(sample_parsed_document):
    settings = Settings(CHUNK_SIZE=70, CHUNK_OVERLAP=15)
    chunker1 = RecursiveChunker(settings)
    chunker2 = RecursiveChunker(settings)
    
    text = (
        "## Architecture\n"
        "The system has many components.\n"
        "They all work together nicely.\n"
        "Very good design indeed. Let's make it a bit longer to exceed 70 characters so it splits."
    )
    doc1 = create_normalized_document(sample_parsed_document, text)
    doc2 = create_normalized_document(sample_parsed_document, text)
    
    chunks1 = chunker1.chunk(doc1)
    chunks2 = chunker2.chunk(doc2)
    
    assert len(chunks1) > 1
    assert [c.text for c in chunks1] == [c.text for c in chunks2]
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_configurable_overlap(sample_parsed_document):
    # Test with no overlap
    settings_no_overlap = Settings(CHUNK_SIZE=64, CHUNK_OVERLAP=0)
    chunker_no_overlap = RecursiveChunker(settings_no_overlap)
    
    # Test with overlap
    settings_overlap = Settings(CHUNK_SIZE=64, CHUNK_OVERLAP=20)
    chunker_overlap = RecursiveChunker(settings_overlap)
    
    text = ("1234567890 " * 20).strip()
    doc = create_normalized_document(sample_parsed_document, text)
    
    chunks_no = chunker_no_overlap.chunk(doc)
    chunks_yes = chunker_overlap.chunk(doc)
    
    total_len_no = sum(len(c.text) for c in chunks_no)
    total_len_yes = sum(len(c.text) for c in chunks_yes)
    
    assert total_len_yes > total_len_no
