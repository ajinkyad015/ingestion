"""Unit tests for IngestionOrchestrator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rag_ingestion.application.orchestrators.ingestion import (
    IngestionFailure,
    IngestionOrchestrator,
    IngestionResult,
)
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.entities.document import Document, NormalizedDocument, ParsedDocument

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

_DOC_PATH = Path("/docs/a.txt")
_DOC_HASH = "abc123"


def _make_document(
    path: Path = _DOC_PATH,
    source_hash: str = _DOC_HASH,
    size_bytes: int = 1,
) -> Document:
    return Document(
        source_path=path,
        extension=path.suffix,
        source_hash=source_hash,
        size_bytes=size_bytes,
    )


def _make_parsed(doc: Document) -> ParsedDocument:
    return ParsedDocument(source=doc, text="hello world", title="", num_pages=0)


def _make_normalized(parsed: ParsedDocument) -> NormalizedDocument:
    return NormalizedDocument(source=parsed, text=parsed.text)


def _make_chunk(doc: Document, index: int = 0) -> Chunk:
    meta = ChunkMetadata(
        document_hash=doc.source_hash,
        source_path=doc.source_path,
        chunk_index=index,
        total_chunks=1,
        heading="",
        page_number=0,
        char_start=0,
        char_end=5,
    )
    return Chunk(chunk_id=f"{doc.source_hash}_{index:06d}", text="hello", metadata=meta)


def _make_embedded(chunk: Chunk) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk, embedding=(0.1, 0.2))


class _StubLoader:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs: list[Document] = docs if docs is not None else [_make_document()]
        self.calls: list[str] = []

    def load(self, source: str) -> list[Document]:
        self.calls.append(source)
        return self.docs


class _StubParser:
    def __init__(self) -> None:
        self.calls: list[Document] = []

    def parse(self, document: Document) -> ParsedDocument:
        self.calls.append(document)
        return _make_parsed(document)


class _StubNormalizer:
    def __init__(self) -> None:
        self.calls: list[ParsedDocument] = []

    def normalize(self, document: ParsedDocument) -> NormalizedDocument:
        self.calls.append(document)
        return _make_normalized(document)


class _StubChunker:
    def __init__(self, chunks_per_doc: int = 1) -> None:
        self.calls: list[NormalizedDocument] = []
        self._n = chunks_per_doc

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        self.calls.append(document)
        doc = document.source.source
        return [_make_chunk(doc, i) for i in range(self._n)]


class _StubMetadataEnricher:
    def __init__(self) -> None:
        self.calls: list[list[Chunk]] = []

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        self.calls.append(list(chunks))
        return chunks


class _StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[Chunk]] = []

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.calls.append(list(chunks))
        return [_make_embedded(c) for c in chunks]


class _StubVectorStore:
    def __init__(self) -> None:
        self.calls: list[list[EmbeddedChunk]] = []

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        self.calls.append(list(embedded_chunks))

    def delete_by_document(self, document_hash: str) -> int:
        return 0

    def count(self) -> int:
        return 0


def _make_orchestrator(
    *,
    loader: Any = None,
    parser: Any = None,
    normalizer: Any = None,
    chunker: Any = None,
    metadata_enricher: Any = None,
    embedder: Any = None,
    vector_store: Any = None,
) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        loader=loader if loader is not None else _StubLoader(),
        parser=parser if parser is not None else _StubParser(),
        normalizer=normalizer if normalizer is not None else _StubNormalizer(),
        chunker=chunker if chunker is not None else _StubChunker(),
        metadata_enricher=metadata_enricher 
        if metadata_enricher is not None else _StubMetadataEnricher(),
        embedder=embedder if embedder is not None else _StubEmbedder(),
        vector_store=vector_store if vector_store is not None else _StubVectorStore(),
    )


# ---------------------------------------------------------------------------
# Result model tests
# ---------------------------------------------------------------------------


class TestResultModels:
    def test_ingestion_failure_is_frozen(self) -> None:
        f = IngestionFailure(document_path="/a.txt", stage="parse", error="boom")
        with pytest.raises(FrozenInstanceError):
            f.document_path = "/b.txt"  # type: ignore[misc]

    def test_ingestion_result_is_frozen(self) -> None:
        r = IngestionResult(
            documents_processed=1,
            chunks_created=2,
            embeddings_created=2,
            vectors_written=2,
            failures=(),
            elapsed_time_seconds=0.1,
        )
        with pytest.raises(FrozenInstanceError):
            r.documents_processed = 99  # type: ignore[misc]

    def test_ingestion_result_failures_is_tuple(self) -> None:
        failure = IngestionFailure(document_path="/a.txt", stage="parse", error="err")
        r = IngestionResult(
            documents_processed=0,
            chunks_created=0,
            embeddings_created=0,
            vectors_written=0,
            failures=(failure,),
            elapsed_time_seconds=0.0,
        )
        assert isinstance(r.failures, tuple)
        assert r.failures[0] is failure


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestIngestionOrchestratorHappyPath:
    def test_returns_ingestion_result(self) -> None:
        result = _make_orchestrator().ingest("/docs")
        assert isinstance(result, IngestionResult)

    def test_single_document_counters(self) -> None:
        loader = _StubLoader([_make_document()])
        chunker = _StubChunker(chunks_per_doc=3)
        result = _make_orchestrator(loader=loader, chunker=chunker).ingest("/docs")

        assert result.documents_processed == 1
        assert result.chunks_created == 3
        assert result.embeddings_created == 3
        assert result.vectors_written == 3
        assert result.failures == ()

    def test_elapsed_time_is_non_negative_float(self) -> None:
        result = _make_orchestrator().ingest("/docs")
        assert isinstance(result.elapsed_time_seconds, float)
        assert result.elapsed_time_seconds >= 0.0

    def test_empty_loader_returns_zero_counts(self) -> None:
        loader = _StubLoader(docs=[])
        parser = _StubParser()
        normalizer = _StubNormalizer()
        chunker = _StubChunker()
        metadata_enricher = _StubMetadataEnricher()
        embedder = _StubEmbedder()
        vector_store = _StubVectorStore()

        result = _make_orchestrator(
            loader=loader,
            parser=parser,
            normalizer=normalizer,
            chunker=chunker,
            metadata_enricher=metadata_enricher,
            embedder=embedder,
            vector_store=vector_store,
        ).ingest("/docs")

        assert result.documents_processed == 0
        assert result.chunks_created == 0
        assert result.embeddings_created == 0
        assert result.vectors_written == 0
        assert result.failures == ()

        assert parser.calls == []
        assert normalizer.calls == []
        assert chunker.calls == []
        assert metadata_enricher.calls == []
        assert embedder.calls == []
        assert vector_store.calls == []

    def test_multiple_documents_accumulate_counters(self) -> None:
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])
        chunker = _StubChunker(chunks_per_doc=2)

        result = _make_orchestrator(loader=loader, chunker=chunker).ingest("/docs")

        assert result.documents_processed == 2
        assert result.chunks_created == 4
        assert result.embeddings_created == 4
        assert result.vectors_written == 4
        assert result.failures == ()

    def test_multiple_documents_empty_chunks_not_counted(self) -> None:
        """Documents that produce zero chunks are still counted as processed."""
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        loader = _StubLoader([doc1])

        empty_chunker = _StubChunker(chunks_per_doc=0)
        metadata_enricher = _StubMetadataEnricher()
        embedder = _StubEmbedder()
        vector_store = _StubVectorStore()

        result = _make_orchestrator(
            loader=loader,
            chunker=empty_chunker,
            metadata_enricher=metadata_enricher,
            embedder=embedder,
            vector_store=vector_store,
        ).ingest("/docs")

        assert result.documents_processed == 1
        assert result.chunks_created == 0
        assert result.embeddings_created == 0
        assert result.vectors_written == 0
        assert result.failures == ()
        # downstream stages not called when no chunks
        assert metadata_enricher.calls == []
        assert embedder.calls == []
        assert vector_store.calls == []


# ---------------------------------------------------------------------------
# Pipeline ordering tests
# ---------------------------------------------------------------------------


class TestPipelineOrdering:
    def test_all_stages_called_in_order(self) -> None:
        call_order: list[str] = []

        class OrderLoader:
            def load(self, source: str) -> list[Document]:
                call_order.append("load")
                return [_make_document()]

        class OrderParser:
            def parse(self, document: Document) -> ParsedDocument:
                call_order.append("parse")
                return _make_parsed(document)

        class OrderNormalizer:
            def normalize(self, document: ParsedDocument) -> NormalizedDocument:
                call_order.append("normalize")
                return _make_normalized(document)

        class OrderChunker:
            def chunk(self, document: NormalizedDocument) -> list[Chunk]:
                call_order.append("chunk")
                return [_make_chunk(document.source.source)]

        class OrderEnricher:
            def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
                call_order.append("enrich")
                return chunks

        class OrderEmbedder:
            def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
                call_order.append("embed")
                return [_make_embedded(c) for c in chunks]

        class OrderStore:
            def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
                call_order.append("store")

        IngestionOrchestrator(
            loader=OrderLoader(),
            parser=OrderParser(),
            normalizer=OrderNormalizer(),
            chunker=OrderChunker(),
            metadata_enricher=OrderEnricher(),
            embedder=OrderEmbedder(),
            vector_store=OrderStore(),
        ).ingest("/docs")

        assert call_order == ["load", "parse", "normalize", "chunk", "enrich", "embed", "store"]

    def test_loader_called_with_source(self) -> None:
        loader = _StubLoader(docs=[])
        _make_orchestrator(loader=loader).ingest("/custom/source")
        assert loader.calls == ["/custom/source"]

    def test_embedder_called_once_per_document(self) -> None:
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])
        embedder = _StubEmbedder()

        _make_orchestrator(loader=loader, embedder=embedder).ingest("/docs")

        assert len(embedder.calls) == 2

    def test_vector_store_called_once_per_document(self) -> None:
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])
        vector_store = _StubVectorStore()

        _make_orchestrator(loader=loader, vector_store=vector_store).ingest("/docs")

        assert len(vector_store.calls) == 2

    def test_embedder_receives_enriched_chunks(self) -> None:
        """Enricher output is what reaches the Embedder, not raw chunks."""
        doc = _make_document()
        loader = _StubLoader([doc])

        enriched_marker: list[bool] = []

        class MarkerEnricher:
            def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
                # Return a new list so we can detect reference equality
                return [_make_chunk(doc, 99)]

        class ProbeEmbedder:
            received: list[list[Chunk]] = []

            def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
                ProbeEmbedder.received.append(chunks)
                enriched_marker.append(chunks[0].metadata.chunk_index == 99)
                return [_make_embedded(c) for c in chunks]

        _make_orchestrator(
            loader=loader,
            metadata_enricher=MarkerEnricher(),
            embedder=ProbeEmbedder(),
        ).ingest("/docs")

        assert enriched_marker == [True]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestDocumentFailures:
    """Per-document exceptions are caught, recorded, and pipeline continues."""

    def _failing(self, exc: Exception, stage: str) -> IngestionOrchestrator:
        doc = _make_document()
        loader = _StubLoader([doc])

        class RaisingParser:
            def parse(self, document: Document) -> ParsedDocument:
                if stage == "parse":
                    raise exc
                return _make_parsed(document)

        class RaisingNormalizer:
            def normalize(self, document: ParsedDocument) -> NormalizedDocument:
                if stage == "normalize":
                    raise exc
                return _make_normalized(document)

        class RaisingChunker:
            def chunk(self, document: NormalizedDocument) -> list[Chunk]:
                if stage == "chunk":
                    raise exc
                return [_make_chunk(document.source.source)]

        class RaisingEmbedder:
            def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
                if stage == "embed":
                    raise exc
                return [_make_embedded(c) for c in chunks]

        return IngestionOrchestrator(
            loader=loader,
            parser=RaisingParser(),
            normalizer=RaisingNormalizer(),
            chunker=RaisingChunker(),
            metadata_enricher=_StubMetadataEnricher(),
            embedder=RaisingEmbedder(),
            vector_store=_StubVectorStore(),
        )

    def test_parse_failure_recorded_as_ingestion_failure(self) -> None:
        orch = self._failing(RuntimeError("parse boom"), "parse")
        result = orch.ingest("/docs")

        assert result.documents_processed == 0
        assert len(result.failures) == 1
        assert result.failures[0].stage == "parse"
        assert "parse boom" in result.failures[0].error

    def test_normalize_failure_recorded(self) -> None:
        orch = self._failing(ValueError("norm err"), "normalize")
        result = orch.ingest("/docs")

        assert len(result.failures) == 1
        assert result.failures[0].stage == "normalize"

    def test_chunk_failure_recorded(self) -> None:
        orch = self._failing(RuntimeError("chunk err"), "chunk")
        result = orch.ingest("/docs")

        assert len(result.failures) == 1
        assert result.failures[0].stage == "chunk"

    def test_embed_failure_recorded(self) -> None:
        orch = self._failing(RuntimeError("embed err"), "embed")
        result = orch.ingest("/docs")

        assert len(result.failures) == 1
        assert result.failures[0].stage == "embed"

    def test_failure_continues_remaining_documents(self) -> None:
        """A failure in doc1 must not prevent doc2 from processing."""
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])

        call_count = [0]

        class PartialParser:
            def parse(self, document: Document) -> ParsedDocument:
                call_count[0] += 1
                if document.source_hash == "hash1":
                    raise RuntimeError("first fails")
                return _make_parsed(document)

        result = IngestionOrchestrator(
            loader=loader,
            parser=PartialParser(),
            normalizer=_StubNormalizer(),
            chunker=_StubChunker(),
            metadata_enricher=_StubMetadataEnricher(),
            embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
        ).ingest("/docs")

        assert result.documents_processed == 1
        assert len(result.failures) == 1
        assert result.failures[0].document_path == str(doc1.source_path)
        assert call_count[0] == 2  # both docs attempted

    def test_failure_document_path_recorded(self) -> None:
        doc = _make_document(Path("/some/special/doc.pdf"), "hashX")
        loader = _StubLoader([doc])

        class AlwaysFail:
            def parse(self, document: Document) -> ParsedDocument:
                raise RuntimeError("bang")

        result = IngestionOrchestrator(
            loader=loader,
            parser=AlwaysFail(),
            normalizer=_StubNormalizer(),
            chunker=_StubChunker(),
            metadata_enricher=_StubMetadataEnricher(),
            embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
        ).ingest("/docs")

        assert result.failures[0].document_path == str(doc.source_path)

    def test_multiple_failures_all_recorded(self) -> None:
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])

        class AlwaysFail:
            def parse(self, document: Document) -> ParsedDocument:
                raise RuntimeError("always")

        result = IngestionOrchestrator(
            loader=loader,
            parser=AlwaysFail(),
            normalizer=_StubNormalizer(),
            chunker=_StubChunker(),
            metadata_enricher=_StubMetadataEnricher(),
            embedder=_StubEmbedder(),
            vector_store=_StubVectorStore(),
        ).ingest("/docs")

        assert result.documents_processed == 0
        assert len(result.failures) == 2


class TestPipelineFailures:
    """Loader-level (pipeline) failures propagate immediately."""

    def test_loader_error_propagates(self) -> None:
        class BrokenLoader:
            def load(self, source: str) -> list[Document]:
                raise RuntimeError("loader broken")

        orch = _make_orchestrator(loader=BrokenLoader())
        with pytest.raises(RuntimeError, match="loader broken"):
            orch.ingest("/docs")

    def test_keyboard_interrupt_propagates(self) -> None:
        class KBIParser:
            def parse(self, document: Document) -> ParsedDocument:
                raise KeyboardInterrupt

        orch = _make_orchestrator(parser=KBIParser())
        with pytest.raises(KeyboardInterrupt):
            orch.ingest("/docs")

    def test_system_exit_propagates(self) -> None:
        class SysExitParser:
            def parse(self, document: Document) -> ParsedDocument:
                raise SystemExit(1)

        orch = _make_orchestrator(parser=SysExitParser())
        with pytest.raises(SystemExit):
            orch.ingest("/docs")


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestLogging:
    """Verify structured log events are emitted via get_logger()."""

    def _run_with_mock_log(
        self, docs: list[Document] | None = None, chunker: Any = None
    ) -> tuple[IngestionResult, MagicMock]:
        mock_log = MagicMock()
        with patch(
            "rag_ingestion.application.orchestrators.ingestion.get_logger",
            return_value=mock_log,
        ):
            orch = IngestionOrchestrator(
                loader=_StubLoader(docs if docs is not None else [_make_document()]),
                parser=_StubParser(),
                normalizer=_StubNormalizer(),
                chunker=chunker or _StubChunker(),
                metadata_enricher=_StubMetadataEnricher(),
                embedder=_StubEmbedder(),
                vector_store=_StubVectorStore(),
            )
            result = orch.ingest("/docs")
        return result, mock_log

    def _info_events(self, mock_log: MagicMock) -> list[str]:
        return [c.args[0] for c in mock_log.info.call_args_list]

    def test_pipeline_started_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "pipeline_started" in self._info_events(mock_log)

    def test_document_loaded_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "document_loaded" in self._info_events(mock_log)

    def test_document_parsed_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "document_parsed" in self._info_events(mock_log)

    def test_document_normalized_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "document_normalized" in self._info_events(mock_log)

    def test_document_chunked_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "document_chunked" in self._info_events(mock_log)

    def test_metadata_enriched_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "metadata_enriched" in self._info_events(mock_log)

    def test_embeddings_created_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "embeddings_created" in self._info_events(mock_log)

    def test_vectors_written_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "vectors_written" in self._info_events(mock_log)

    def test_pipeline_completed_logged(self) -> None:
        _, mock_log = self._run_with_mock_log()
        assert "pipeline_completed" in self._info_events(mock_log)

    def test_document_failed_logged_on_error(self) -> None:
        mock_log = MagicMock()

        class FailParser:
            def parse(self, document: Document) -> ParsedDocument:
                raise RuntimeError("oops")

        with patch(
            "rag_ingestion.application.orchestrators.ingestion.get_logger",
            return_value=mock_log,
        ):
            orch = IngestionOrchestrator(
                loader=_StubLoader(),
                parser=FailParser(),
                normalizer=_StubNormalizer(),
                chunker=_StubChunker(),
                metadata_enricher=_StubMetadataEnricher(),
                embedder=_StubEmbedder(),
                vector_store=_StubVectorStore(),
            )
            orch.ingest("/docs")

        error_events = [c.args[0] for c in mock_log.error.call_args_list]
        assert "document_failed" in error_events

    def test_no_downstream_log_events_when_loader_empty(self) -> None:
        _, mock_log = self._run_with_mock_log(docs=[])
        events = self._info_events(mock_log)
        assert "document_loaded" not in events
        assert "document_parsed" not in events

    def test_chunked_log_includes_chunk_count(self) -> None:
        chunker = _StubChunker(chunks_per_doc=4)
        _, mock_log = self._run_with_mock_log(chunker=chunker)
        chunked_calls = [
            c for c in mock_log.info.call_args_list if c.args[0] == "document_chunked"
        ]
        assert chunked_calls
        assert chunked_calls[0].kwargs.get("chunk_count") == 4


# ---------------------------------------------------------------------------
# VectorStore / Embedder call-count invariants
# ---------------------------------------------------------------------------


class TestCallCountInvariants:
    def test_vector_store_not_called_when_zero_chunks(self) -> None:
        vector_store = _StubVectorStore()
        _make_orchestrator(
            chunker=_StubChunker(chunks_per_doc=0),
            vector_store=vector_store,
        ).ingest("/docs")
        assert vector_store.calls == []

    def test_embedder_not_called_when_zero_chunks(self) -> None:
        embedder = _StubEmbedder()
        _make_orchestrator(
            chunker=_StubChunker(chunks_per_doc=0),
            embedder=embedder,
        ).ingest("/docs")
        assert embedder.calls == []

    def test_vector_store_called_once_per_success_on_mixed(self) -> None:
        """2 docs, first fails parse → VectorStore called exactly once."""
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])
        vector_store = _StubVectorStore()

        class PartialParser:
            def parse(self, document: Document) -> ParsedDocument:
                if document.source_hash == "hash1":
                    raise RuntimeError("fail")
                return _make_parsed(document)

        IngestionOrchestrator(
            loader=loader,
            parser=PartialParser(),
            normalizer=_StubNormalizer(),
            chunker=_StubChunker(),
            metadata_enricher=_StubMetadataEnricher(),
            embedder=_StubEmbedder(),
            vector_store=vector_store,
        ).ingest("/docs")

        assert len(vector_store.calls) == 1

    def test_embedder_called_once_per_success_on_mixed(self) -> None:
        doc1 = _make_document(Path("/docs/a.txt"), "hash1")
        doc2 = _make_document(Path("/docs/b.txt"), "hash2")
        loader = _StubLoader([doc1, doc2])
        embedder = _StubEmbedder()

        class PartialParser:
            def parse(self, document: Document) -> ParsedDocument:
                if document.source_hash == "hash1":
                    raise RuntimeError("fail")
                return _make_parsed(document)

        IngestionOrchestrator(
            loader=loader,
            parser=PartialParser(),
            normalizer=_StubNormalizer(),
            chunker=_StubChunker(),
            metadata_enricher=_StubMetadataEnricher(),
            embedder=embedder,
            vector_store=_StubVectorStore(),
        ).ingest("/docs")

        assert len(embedder.calls) == 1