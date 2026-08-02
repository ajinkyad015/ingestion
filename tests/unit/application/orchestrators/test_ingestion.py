from __future__ import annotations

from pathlib import Path

from rag_ingestion.application.orchestrators.ingestion import IngestionOrchestrator
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.entities.document import Document, NormalizedDocument, ParsedDocument


class _StubLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self, source: str) -> list[Document]:
        self.calls.append(source)
        return [
            Document(
                source_path=Path(source),
                extension=".txt",
                source_hash="abc123",
                size_bytes=1,
            )
        ]


class _StubParser:
    def __init__(self) -> None:
        self.calls: list[Document] = []

    def parse(self, document: Document) -> ParsedDocument:
        self.calls.append(document)
        return ParsedDocument(source=document, text="parsed text", title="", num_pages=0)


class _StubNormalizer:
    def __init__(self) -> None:
        self.calls: list[ParsedDocument] = []

    def normalize(self, document: ParsedDocument) -> NormalizedDocument:
        self.calls.append(document)
        return NormalizedDocument(source=document, text=document.text)


class _StubChunker:
    def __init__(self) -> None:
        self.calls: list[NormalizedDocument] = []

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        self.calls.append(document)
        metadata = ChunkMetadata(
            document_hash=document.source.source.source_hash,
            source_path=document.source.source.source_path,
            chunk_index=0,
            total_chunks=1,
            heading="",
            page_number=0,
            char_start=0,
            char_end=len(document.text),
        )
        return [Chunk(chunk_id="abc123_000000", text=document.text, metadata=metadata)]


class _StubMetadataEnricher:
    def __init__(self) -> None:
        self.calls: list[list[Chunk]] = []

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        self.calls.append(chunks)
        return chunks


class _StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[Chunk]] = []

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.calls.append(chunks)
        return [EmbeddedChunk(chunk=chunk, embedding=(0.5,)) for chunk in chunks]


class _StubVectorStore:
    def __init__(self) -> None:
        self.calls: list[list[EmbeddedChunk]] = []

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        self.calls.append(embedded_chunks)

    def delete_by_document(self, document_hash: str) -> int:
        return 0

    def count(self) -> int:
        return 0


class TestIngestionOrchestrator:
    def test_ingest_runs_pipeline_and_returns_embedded_count(self) -> None:
        loader = _StubLoader()
        parser = _StubParser()
        normalizer = _StubNormalizer()
        chunker = _StubChunker()
        metadata_enricher = _StubMetadataEnricher()
        embedder = _StubEmbedder()
        vector_store = _StubVectorStore()
        orchestrator = IngestionOrchestrator(
            loader=loader,
            parser=parser,
            normalizer=normalizer,
            chunker=chunker,
            metadata_enricher=metadata_enricher,
            embedder=embedder,
            vector_store=vector_store,
        )

        count = orchestrator.ingest("/tmp/docs")

        assert count == 1
        assert loader.calls == ["/tmp/docs"]
        assert len(parser.calls) == 1
        assert len(normalizer.calls) == 1
        assert len(chunker.calls) == 1
        assert len(metadata_enricher.calls) == 1
        assert len(embedder.calls) == 1
        assert len(vector_store.calls) == 1
        assert vector_store.calls[0][0].chunk.text == "parsed text"
