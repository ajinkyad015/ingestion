"""Integration tests for the RAG ingestion pipeline."""

from __future__ import annotations

import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from typing import Any
import pytest
from sentence_transformers import SentenceTransformer
from torch import embedding

from rag_ingestion.config import Settings
from rag_ingestion.application.orchestrators.ingestion import IngestionOrchestrator
from rag_ingestion.infrastructure.loaders.filesystem import FileSystemLoader
from rag_ingestion.infrastructure.parsers.docling_parser import DoclingParser
from rag_ingestion.infrastructure.normalizers.default import DefaultDocumentNormalizer
from rag_ingestion.infrastructure.chunkers.recursive import RecursiveChunker
from rag_ingestion.infrastructure.metadata_enrichers.default import DefaultMetadataEnricher
from rag_ingestion.infrastructure.embedders.sentence_transformer import SentenceTransformerEmbedder
from rag_ingestion.infrastructure.storage.chroma import ChromaVectorStore
from rag_ingestion.domain.entities.document import Document
from structlog.testing import capture_logs
from docx import Document as WordDocument
from reportlab.pdfgen import canvas
# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def shared_sentence_transformer() -> SentenceTransformer:
    """
    Session/module scoped SentenceTransformer instance.
    Prevents reloading/re-downloading the model weights between individual tests.
    """
    try:
        # Using the default model BAAI/bge-small-en-v1.5 on CPU
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        pytest.xfail(f"Embedding model is unavailable: {e}")


@pytest.fixture(autouse=True)
def patch_embedder_model(shared_sentence_transformer: SentenceTransformer, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically injects the shared_sentence_transformer into the SentenceTransformerEmbedder
    to bypass loading weights from scratch for every test.
    """
    monkeypatch.setattr(
        SentenceTransformerEmbedder,
        "_build_model",
        lambda self: shared_sentence_transformer,
    )


def create_test_documents(tmp_path: Path) -> Path:
    """Create representative input documents under a temp path."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 1. Markdown Document
    md_content = (
        "# Introduction\n"
        "This is the first paragraph of a markdown document.\n\n"
        "## Features\n"
        "- High-performance pipeline\n"
        "- Clean architecture\n\n"
        "# Section Two\n"
        "Here is some more content about integration testing."
    )
    (input_dir / "doc.md").write_text(md_content, encoding="utf-8")

    # 2. Text Document
    txt_content = (
        "Section 1: Overview\n"
        "This is a plain text file containing some representative test sentences.\n\n"
        "Section 2: Installation\n"
        "You can run the ingestion pipeline via the command line interface."
    )
    (input_dir / "doc.txt").write_text(txt_content, encoding="utf-8")

    # 3. HTML Document
    html_content = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>HTML Integration Document</title></head>\n"
        "<body>\n"
        "<h1>Main Heading</h1>\n"
        "<p>This paragraph contains HTML content to be loaded and parsed by Docling.</p>\n"
        "<h2>Subsection</h2>\n"
        "<ul>\n"
        "  <li>Item One</li>\n"
        "  <li>Item Two</li>\n"
        "</ul>\n"
        "</body>\n"
        "</html>"
    )
    (input_dir / "doc.html").write_text(html_content, encoding="utf-8")


     # 4. PDF Document
    pdf_path = input_dir / "doc.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(100, 750, "PDF Integration Test")
    pdf.drawString(
        100,
        730,
        "This PDF verifies Docling PDF parsing."
    )
    pdf.drawString(
        100,
        710,
        "The ingestion pipeline should create embeddings."
    )
    pdf.save()


    return input_dir


def test_end_to_end_pipeline_success(
    tmp_path: Path,
    shared_sentence_transformer: SentenceTransformer,
) -> None:
    """
    Verify the complete RAG ingestion pipeline executes successfully end-to-end
    using real implementations, and verify correct persistence and query validation.
    """
    input_dir = create_test_documents(tmp_path)
    chroma_dir = tmp_path / "chroma"
    collection_name = f"test_coll_{uuid.uuid4().hex}"

    # 1. Instantiate settings with temporary directories
    settings = Settings(
        CHROMA_PERSIST_DIRECTORY=chroma_dir,
        CHROMA_COLLECTION=collection_name,
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        CHUNK_SIZE=256,
        CHUNK_OVERLAP=32,
    )

    # 2. Wire up the production pipeline components manually
    loader = FileSystemLoader(settings)
    parser = DoclingParser(settings)
    normalizer = DefaultDocumentNormalizer()
    chunker = RecursiveChunker(settings)
    metadata_enricher = DefaultMetadataEnricher(settings)
    embedder = SentenceTransformerEmbedder(settings)
    vector_store = ChromaVectorStore(settings)

    orchestrator = IngestionOrchestrator(
        loader=loader,
        parser=parser,
        normalizer=normalizer,
        chunker=chunker,
        metadata_enricher=metadata_enricher,
        embedder=embedder,
        vector_store=vector_store,
    )

    # 3. Run the pipeline and capture structured logs
    with capture_logs() as captured:
        result = orchestrator.ingest(str(input_dir))
        print(result)

        for failure in result.failures:
            print("=" * 80)
            print("Document:", failure.document_path)
            print("Stage:", failure.stage)
            print("Error:")
            print(failure.error)

    # 4. Verify pipeline output metrics
    assert result.documents_processed == 4
    assert result.chunks_created > 0
    assert result.embeddings_created == result.chunks_created
    assert result.vectors_written == result.chunks_created
    assert len(result.failures) == 0

    # 5. Verify the number of vectors stored equals the number of embedded chunks
    assert vector_store.count() == result.vectors_written

    # 6. Query Validation: Retrieve and inspect from Chroma
    collection = vector_store._get_collection()
    data = collection.get(include=["embeddings", "documents", "metadatas"])

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    embeddings = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])

    pdf_found = False

    for metadata in metadatas:
        if metadata["source_path"].endswith("doc.pdf"):
            pdf_found = True
            break

    assert pdf_found


    pdf_text_found = False

    for text, metadata in zip(documents, metadatas, strict=True):
        if metadata["source_path"].endswith("doc.pdf"):
            assert "PDF Integration Test" in text
            pdf_text_found = True

    assert pdf_text_found

    assert len(ids) == result.vectors_written
    assert len(documents) == result.vectors_written
    assert len(embeddings) == result.vectors_written
    assert len(metadatas) == result.vectors_written

    for doc_id, text, embedding, meta in zip(ids, documents, embeddings, metadatas, strict=True):
        # Stored document IDs exist and are strings
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

        # Stored text exists
        assert isinstance(text, str)
        assert len(text) > 0

        # Embedding dimension
        expected_dim = shared_sentence_transformer.get_sentence_embedding_dimension()
        assert len(embedding) == expected_dim

        # Stored metadata exists and contains required fields from MetadataEnricher
        assert "document_hash" in meta
        assert "chunk_hash" in meta
        assert "page_number" in meta
        assert "section_hierarchy" in meta
        assert "source_path" in meta

        # Verify type of fields
        assert isinstance(meta["document_hash"], str)
        assert isinstance(meta["chunk_hash"], str)
        assert isinstance(meta["page_number"], int)

        # section_hierarchy is persisted as a JSON-serialized string in Chroma
        hierarchy = json.loads(meta["section_hierarchy"])
        assert isinstance(hierarchy, list)

    # 7. Logging Validation: Check structlog output events
    events = {log["event"] for log in captured if "event" in log}
    expected_events = {
        "pipeline_started",
        "document_loaded",
        "document_parsed",
        "document_normalized",
        "document_chunked",
        "metadata_enriched",
        "embeddings_created",
        "vectors_written",
        "pipeline_completed",
    }
    assert expected_events.issubset(events)


def test_pipeline_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify pipeline execution is deterministic:
    - Run ingestion twice on identical input.
    - Verify identical hashes, chunk boundaries, metadata, and vector count.
    - Verify that the pipeline does not duplicate vectors (upsert-based).
    """
    input_dir = create_test_documents(tmp_path)
    chroma_dir = tmp_path / "chroma"
    collection_name = f"test_coll_{uuid.uuid4().hex}"

    settings = Settings(
        CHROMA_PERSIST_DIRECTORY=chroma_dir,
        CHROMA_COLLECTION=collection_name,
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        CHUNK_SIZE=256,
        CHUNK_OVERLAP=32,
    )

    # Patch DefaultMetadataEnricher._resolve_timestamps to return fixed timestamps.
    # This prevents st_atime changes on filesystem read from breaking hash determinism.
    from datetime import UTC, datetime
    fixed_time = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        DefaultMetadataEnricher,
        "_resolve_timestamps",
        lambda *args, **kwargs: (fixed_time, fixed_time, fixed_time),
    )

    # Instantiate first orchestrator
    loader1 = FileSystemLoader(settings)
    parser1 = DoclingParser(settings)
    normalizer1 = DefaultDocumentNormalizer()
    chunker1 = RecursiveChunker(settings)
    metadata_enricher1 = DefaultMetadataEnricher(settings)
    embedder1 = SentenceTransformerEmbedder(settings)
    vector_store1 = ChromaVectorStore(settings)
    orchestrator1 = IngestionOrchestrator(
        loader=loader1,
        parser=parser1,
        normalizer=normalizer1,
        chunker=chunker1,
        metadata_enricher=metadata_enricher1,
        embedder=embedder1,
        vector_store=vector_store1,
    )

    # Run 1
    result1 = orchestrator1.ingest(str(input_dir))
    assert result1.documents_processed == 4

    # Query Chroma after run 1
    collection1 = vector_store1._get_collection()
    data1 = collection1.get(include=["embeddings", "documents", "metadatas"])

    # Run 2 on identical input using same collection
    loader2 = FileSystemLoader(settings)
    parser2 = DoclingParser(settings)
    normalizer2 = DefaultDocumentNormalizer()
    chunker2 = RecursiveChunker(settings)
    metadata_enricher2 = DefaultMetadataEnricher(settings)
    embedder2 = SentenceTransformerEmbedder(settings)
    vector_store2 = ChromaVectorStore(settings)
    orchestrator2 = IngestionOrchestrator(
        loader=loader2,
        parser=parser2,
        normalizer=normalizer2,
        chunker=chunker2,
        metadata_enricher=metadata_enricher2,
        embedder=embedder2,
        vector_store=vector_store2,
    )

    result2 = orchestrator2.ingest(str(input_dir))
    assert result2.documents_processed == 4

    # Query Chroma after run 2
    collection2 = vector_store2._get_collection()
    data2 = collection2.get(include=["embeddings", "documents", "metadatas"])

    # 1. Verify vector count hasn't increased (no duplicate vectors)
    assert vector_store2.count() == result1.vectors_written
    assert len(data2["ids"]) == len(data1["ids"])

    # Map records by ID for comparison
    records1 = {data1["ids"][i]: (data1["documents"][i], data1["metadatas"][i], data1["embeddings"][i]) for i in range(len(data1["ids"]))}
    records2 = {data2["ids"][i]: (data2["documents"][i], data2["metadatas"][i], data2["embeddings"][i]) for i in range(len(data2["ids"]))}

    # 2. Verify identical document hash, chunk hash, chunk boundaries, and metadata
    assert set(records1.keys()) == set(records2.keys())
    for chunk_id in records1:
        doc1, meta1, emb1 = records1[chunk_id]
        doc2, meta2, emb2 = records2[chunk_id]

        assert doc1 == doc2  # identical chunk boundaries & text
        assert meta1["document_hash"] == meta2["document_hash"]
        assert meta1["chunk_hash"] == meta2["chunk_hash"]
        assert meta1["char_start"] == meta2["char_start"]
        assert meta1["char_end"] == meta2["char_end"]
        assert meta1["section_hierarchy"] == meta2["section_hierarchy"]

        # Embeddings should match (within small float precision limits)
        assert len(emb1) == len(emb2)
        for val1, val2 in zip(emb1, emb2, strict=True):
            assert abs(val1 - val2) < 1e-6


def test_pipeline_failures_graceful_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify graceful handling of various failures:
    - Unsupported extension (ignored/skipped by loader, or parsed with error if forced)
    - Empty document (processed with 0 chunks, no failure)
    - Malformed document (corrupt pdf fails parse, recorded as failure, rest of pipeline proceeds)
    - Parser failure (gracefully recorded, rest continues)
    - Embedding failure (gracefully recorded, rest continues)
    - Storage failure (gracefully recorded, rest continues)
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid document
    (input_dir / "valid.txt").write_text("This is valid content.", encoding="utf-8")

    # 2. Empty document
    (input_dir / "empty.txt").write_text("", encoding="utf-8")

    # 3. Malformed document (corrupt PDF)
    (input_dir / "malformed.pdf").write_bytes(b"Not a valid PDF file content.")

    # 4. Document that will trigger simulated parser failure
    (input_dir / "fail_parse.txt").write_text("Trigger a parser error please.", encoding="utf-8")

    # 5. Document that will trigger simulated embedding failure
    (input_dir / "fail_embed.txt").write_text("Trigger an embedding error please.", encoding="utf-8")

    # 6. Document that will trigger simulated storage failure
    (input_dir / "fail_store.txt").write_text("Trigger a storage error please.", encoding="utf-8")

    # 7. Unsupported extension (should be ignored by FileSystemLoader)
    (input_dir / "unsupported.xyz").write_text("Some text.", encoding="utf-8")

    chroma_dir = tmp_path / "chroma"
    collection_name = f"test_coll_{uuid.uuid4().hex}"
    settings = Settings(
        CHROMA_PERSIST_DIRECTORY=chroma_dir,
        CHROMA_COLLECTION=collection_name,
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
    )

    loader = FileSystemLoader(settings)
    parser = DoclingParser(settings)
    normalizer = DefaultDocumentNormalizer()
    chunker = RecursiveChunker(settings)
    metadata_enricher = DefaultMetadataEnricher(settings)
    embedder = SentenceTransformerEmbedder(settings)
    vector_store = ChromaVectorStore(settings)

    # Patch Parser to simulate failure for "fail_parse"
    original_parse = parser.parse
    def patch_parse(document: Document) -> Any:
        if "fail_parse" in str(document.source_path):
            raise RuntimeError("Simulated parsing error")
        return original_parse(document)
    monkeypatch.setattr(parser, "parse", patch_parse)

    # Patch Embedder to simulate failure for "fail_embed"
    original_embed = embedder.embed
    def patch_embed(chunks: list[Any]) -> list[Any]:
        if chunks and any("fail_embed" in str(chunk.metadata.source_path) for chunk in chunks):
            raise RuntimeError("Simulated embedding error")
        return original_embed(chunks)
    monkeypatch.setattr(embedder, "embed", patch_embed)

    # Patch Storage to simulate failure for "fail_store"
    original_upsert = vector_store.upsert
    def patch_upsert(chunks: list[Any]) -> None:
        if chunks and any("fail_store" in str(chunk.chunk.metadata.source_path) for chunk in chunks):
            raise RuntimeError("Simulated storage error")
        original_upsert(chunks)
    monkeypatch.setattr(vector_store, "upsert", patch_upsert)

    orchestrator = IngestionOrchestrator(
        loader=loader,
        parser=parser,
        normalizer=normalizer,
        chunker=chunker,
        metadata_enricher=metadata_enricher,
        embedder=embedder,
        vector_store=vector_store,
    )

    result = orchestrator.ingest(str(input_dir))

    # Verify that:
    # - Valid document succeeded (1 processed)
    # - Empty document succeeded (1 processed, since 0 chunks is not a failure)
    # Total processed: 2
    assert result.documents_processed == 2

    # Verify that the failures list has recorded the expected failures
    failures_by_doc = {Path(f.document_path).name: f for f in result.failures}
    
    # 1. Malformed document (corrupt PDF parser failure)
    assert "malformed.pdf" in failures_by_doc
    assert failures_by_doc["malformed.pdf"].stage == "parse"

    # 2. Simulated parser failure
    assert "fail_parse.txt" in failures_by_doc
    assert failures_by_doc["fail_parse.txt"].stage == "parse"
    assert "Simulated parsing error" in failures_by_doc["fail_parse.txt"].error

    # 3. Simulated embedding failure
    assert "fail_embed.txt" in failures_by_doc
    assert failures_by_doc["fail_embed.txt"].stage == "embed"
    assert "Simulated embedding error" in failures_by_doc["fail_embed.txt"].error

    # 4. Simulated storage failure
    assert "fail_store.txt" in failures_by_doc
    assert failures_by_doc["fail_store.txt"].stage == "vector_store"
    assert "Simulated storage error" in failures_by_doc["fail_store.txt"].error

    # Verify that "unsupported.xyz" was ignored by Loader entirely and not attempted
    assert "unsupported.xyz" not in failures_by_doc

    # Direct parser check: verify parser.parse on an unsupported extension raises ValueError
    unsupported_doc = Document(
        source_path=input_dir / "unsupported.xyz",
        extension=".xyz",
        source_hash="xyz123",
        size_bytes=10,
    )
    with pytest.raises(ValueError, match="does not support extension"):
        parser.parse(unsupported_doc)


def test_cli_integration(tmp_path: Path) -> None:
    """
    Execute the real CLI via subprocess.
    Verify:
    - ingest command
    - exit code is 0
    - logging output contains expected events
    - collection is created and populated in VectorStore
    - successful completion summary is printed to stdout
    """
    input_dir = create_test_documents(tmp_path)
    chroma_dir = tmp_path / "chroma"
    collection_name = f"cli_coll_{uuid.uuid4().hex}"

    # Setup environment variables to point to temp directories and collection
    env = os.environ.copy()
    env["CHROMA_PERSIST_DIRECTORY"] = str(chroma_dir)
    env["CHROMA_COLLECTION"] = collection_name
    env["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    env["LOG_FORMAT"] = "json"  # Ensure JSON output format for reliable parsing

    # Run the CLI tool: uv run rag-ingestion ingest <input_dir>
    cmd = ["uv", "run", "rag-ingestion", "ingest", str(input_dir)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    print("STDOUT")
    print(result.stdout)

    print("STDERR")
    print(result.stderr)
    # 1. Verify exit code
    assert result.returncode == 0

    # 2. Verify successful completion message in stdout
    assert "Ingestion completed." in result.stdout
    assert "Documents processed : 4" in result.stdout
    assert "Failures            : 0" in result.stdout

    # 3. Verify logging events in stdout
    log_events = []
    # Both stdout and stderr can contain structured logs depending on configuration
    all_output = result.stdout + "\n" + result.stderr
    for line in all_output.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "event" in data:
                    log_events.append(data["event"])
            except json.JSONDecodeError:
                pass

    expected_events = [
        "pipeline_started",
        "document_loaded",
        "document_parsed",
        "document_normalized",
        "document_chunked",
        "metadata_enriched",
        "embeddings_created",
        "vectors_written",
        "pipeline_completed",
    ]
    for ev in expected_events:
        assert ev in log_events

    # 4. Verify collection creation & population using real ChromaVectorStore client from Python
    settings = Settings(
        CHROMA_PERSIST_DIRECTORY=chroma_dir,
        CHROMA_COLLECTION=collection_name,
    )
    store = ChromaVectorStore(settings)
    assert store.count() > 0
