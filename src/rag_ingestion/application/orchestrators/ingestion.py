from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

from rag_ingestion.domain.protocols.chunker import Chunker
from rag_ingestion.domain.protocols.embedder import Embedder
from rag_ingestion.domain.protocols.loader import Loader
from rag_ingestion.domain.protocols.metadata_enricher import MetadataEnricher
from rag_ingestion.domain.protocols.normalizer import Normalizer
from rag_ingestion.domain.protocols.parser import Parser
from rag_ingestion.domain.protocols.vector_store import VectorStore
from rag_ingestion.logging import get_logger

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionFailure:
    """Records a single document-level failure in the ingestion pipeline."""

    document_path: str
    stage: str
    error: str


@dataclass(frozen=True)
class IngestionResult:
    """Immutable summary of a completed ingestion run."""

    documents_processed: int
    """Number of documents that completed all pipeline stages successfully."""

    chunks_created: int
    """Total chunks produced across all successfully processed documents."""

    embeddings_created: int
    """Total embedding vectors generated across all successfully processed documents."""

    vectors_written: int
    """Total vectors successfully written to the VectorStore."""

    failures: tuple[IngestionFailure, ...]
    """One entry per document that failed at any pipeline stage."""

    elapsed_time_seconds: float
    """Total wall-clock execution time measured with ``time.perf_counter()``."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class IngestionOrchestrator:
    """
    Coordinate the end-to-end ingestion pipeline.

    The orchestrator composes the domain protocols in the same order as the
    production pipeline::

        Loader → Parser → Normalizer → Chunker
            → MetadataEnricher → Embedder → VectorStore

    All dependencies are supplied through constructor injection.  The
    orchestrator depends **only** on domain protocols; no infrastructure
    implementation is imported here.
    """

    def __init__(
        self,
        loader: Loader,
        parser: Parser,
        normalizer: Normalizer,
        chunker: Chunker,
        metadata_enricher: MetadataEnricher,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._loader = loader
        self._parser = parser
        self._normalizer = normalizer
        self._chunker = chunker
        self._metadata_enricher = metadata_enricher
        self._embedder = embedder
        self._vector_store = vector_store
        self._log = get_logger(component="ingestion_orchestrator")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, source: str) -> IngestionResult:
        """
        Ingest all supported documents found at *source*.

        Parameters
        ----------
        source:
            An opaque locator passed directly to the configured ``Loader``.

        Returns
        -------
        IngestionResult
            Immutable summary of the completed run.

        Raises
        ------
        Exception
            Re-raised immediately for any *pipeline* failure (e.g. the Loader
            itself cannot initialise).  ``KeyboardInterrupt`` and
            ``SystemExit`` always propagate normally.
        """
        pipeline_start = time.perf_counter()

        self._log.info("pipeline_started", source=source)

        # Pipeline failure: loader errors propagate immediately.
        documents = self._loader.load(source)

        documents_processed = 0
        chunks_created = 0
        embeddings_created = 0
        vectors_written = 0
        failures: list[IngestionFailure] = []

        for document in documents:
            doc_path = str(document.source_path)
            stage = "unknown"
            try:
                # --- Load (already done by Loader, log the reference) ------
                self._log.info(
                    "document_loaded",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                    size_bytes=document.size_bytes,
                )

                # --- Parse --------------------------------------------------
                stage = "parse"
                parsed = self._parser.parse(document)
                self._log.info(
                    "document_parsed",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                )

                # --- Normalize ----------------------------------------------
                stage = "normalize"
                normalized = self._normalizer.normalize(parsed)
                self._log.info(
                    "document_normalized",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                )

                # --- Chunk --------------------------------------------------
                stage = "chunk"
                raw_chunks = self._chunker.chunk(normalized)
                chunk_count = len(raw_chunks)
                self._log.info(
                    "document_chunked",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                    chunk_count=chunk_count,
                )

                if not raw_chunks:
                    # Nothing to embed — count as processed (no failure).
                    documents_processed += 1
                    continue

                # --- Enrich metadata ----------------------------------------
                stage = "metadata_enrich"
                enriched_chunks = self._metadata_enricher.enrich(raw_chunks)
                self._log.info(
                    "metadata_enriched",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                    chunk_count=len(enriched_chunks),
                )

                # --- Embed (one call per document) --------------------------
                stage = "embed"
                embedded_chunks = self._embedder.embed(enriched_chunks)
                embedding_count = len(embedded_chunks)
                self._log.info(
                    "embeddings_created",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                    embedding_count=embedding_count,
                )

                # --- Persist (one call per document) -----------------------
                stage = "vector_store"
                self._vector_store.upsert(embedded_chunks)
                self._log.info(
                    "vectors_written",
                    document_path=doc_path,
                    document_hash=document.source_hash,
                    vectors_written=embedding_count,
                )

                # Accumulate per-document counters only on full success.
                documents_processed += 1
                chunks_created += chunk_count
                embeddings_created += embedding_count
                vectors_written += embedding_count

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # noqa: BLE001
                failure = IngestionFailure(
                    document_path=doc_path,
                    stage=stage,
                    error=traceback.format_exception_only(type(exc), exc)[-1].strip(),
                )
                failures.append(failure)
                self._log.error(
                    "document_failed",
                    document_path=doc_path,
                    stage=stage,
                    error=str(exc),
                )

        elapsed = time.perf_counter() - pipeline_start
        result = IngestionResult(
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            vectors_written=vectors_written,
            failures=tuple(failures),
            elapsed_time_seconds=elapsed,
        )

        self._log.info(
            "pipeline_completed",
            source=source,
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            vectors_written=vectors_written,
            failure_count=len(failures),
            elapsed_ms=round(elapsed * 1000, 2),
        )

        return result
