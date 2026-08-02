from __future__ import annotations

from rag_ingestion.domain.protocols.chunker import Chunker
from rag_ingestion.domain.protocols.embedder import Embedder
from rag_ingestion.domain.protocols.loader import Loader
from rag_ingestion.domain.protocols.metadata_enricher import MetadataEnricher
from rag_ingestion.domain.protocols.normalizer import Normalizer
from rag_ingestion.domain.protocols.parser import Parser
from rag_ingestion.domain.protocols.vector_store import VectorStore


class IngestionOrchestrator:
    """
    Coordinate the end-to-end ingestion pipeline.

    The orchestrator composes the domain protocols in the same order as the
    production pipeline: load → parse → normalize → chunk → enrich → embed → store.
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

    def ingest(self, source: str) -> int:
        """
        Ingest all supported documents found at *source*.

        Returns the total number of embedded chunks persisted to the vector
        store.
        """
        documents = self._loader.load(source)
        total_embedded = 0

        for document in documents:
            parsed = self._parser.parse(document)
            normalized = self._normalizer.normalize(parsed)
            chunks = self._chunker.chunk(normalized)
            if not chunks:
                continue

            enriched_chunks = self._metadata_enricher.enrich(chunks)
            embedded_chunks = self._embedder.embed(enriched_chunks)
            self._vector_store.upsert(embedded_chunks)
            total_embedded += len(embedded_chunks)

        return total_embedded
