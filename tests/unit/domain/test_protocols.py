from __future__ import annotations

from pathlib import Path

from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.entities.document import Document, NormalizedDocument, ParsedDocument
from rag_ingestion.domain.protocols.chunker import Chunker
from rag_ingestion.domain.protocols.embedder import Embedder
from rag_ingestion.domain.protocols.loader import Loader
from rag_ingestion.domain.protocols.metadata_enricher import MetadataEnricher
from rag_ingestion.domain.protocols.normalizer import Normalizer
from rag_ingestion.domain.protocols.parser import Parser
from rag_ingestion.domain.protocols.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Minimal stub implementations that satisfy each protocol
# ---------------------------------------------------------------------------

class _StubLoader:
    def load(self, source: str) -> list[Document]:
        return []


class _StubParser:
    def parse(self, document: Document) -> ParsedDocument:
        return ParsedDocument(source=document, text="", title="", num_pages=0)


class _StubNormalizer:
    def normalize(self, document: ParsedDocument) -> NormalizedDocument:
        return NormalizedDocument(source=document, text=document.text)


class _StubChunker:
    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        return []


class _StubMetadataEnricher:
    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class _StubEmbedder:
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk=c, embedding=(0.0,)) for c in chunks]


class _StubVectorStore:
    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        return

    def delete_by_document(self, document_hash: str) -> int:
        return 0

    def count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Structural subtype checks (runtime_checkable)
# ---------------------------------------------------------------------------


class TestLoaderProtocol:
    def test_stub_satisfies_loader_protocol(self) -> None:
        assert isinstance(_StubLoader(), Loader)

    def test_missing_load_method_does_not_satisfy(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), Loader)


class TestParserProtocol:
    def test_stub_satisfies_parser_protocol(self) -> None:
        assert isinstance(_StubParser(), Parser)

    def test_missing_parse_method_does_not_satisfy(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), Parser)


class TestNormalizerProtocol:
    def test_stub_satisfies_normalizer_protocol(self) -> None:
        assert isinstance(_StubNormalizer(), Normalizer)

    def test_missing_normalize_method_does_not_satisfy(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), Normalizer)


class TestChunkerProtocol:
    def test_stub_satisfies_chunker_protocol(self) -> None:
        assert isinstance(_StubChunker(), Chunker)

    def test_missing_chunk_method_does_not_satisfy(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), Chunker)


class TestMetadataEnricherProtocol:
    def test_stub_satisfies_metadata_enricher_protocol(self) -> None:
        assert isinstance(_StubMetadataEnricher(), MetadataEnricher)

    def test_missing_enrich_method_does_not_satisfy(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), MetadataEnricher)


class TestEmbedderProtocol:
    def test_stub_satisfies_embedder_protocol(self) -> None:
        assert isinstance(_StubEmbedder(), Embedder)

    def test_missing_embed_method_does_not_satisfy(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), Embedder)


class TestVectorStoreProtocol:
    def test_stub_satisfies_vector_store_protocol(self) -> None:
        assert isinstance(_StubVectorStore(), VectorStore)

    def test_missing_upsert_does_not_satisfy(self) -> None:
        class _Bad:
            def delete_by_document(self, document_hash: str) -> int:
                return 0

            def count(self) -> int:
                return 0

        assert not isinstance(_Bad(), VectorStore)

    def test_missing_delete_does_not_satisfy(self) -> None:
        class _Bad:
            def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
                return

            def count(self) -> int:
                return 0

        assert not isinstance(_Bad(), VectorStore)

    def test_missing_count_does_not_satisfy(self) -> None:
        class _Bad:
            def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
                return

            def delete_by_document(self, document_hash: str) -> int:
                return 0

        assert not isinstance(_Bad(), VectorStore)


# ---------------------------------------------------------------------------
# Behavioural contracts verified through stub implementations
# ---------------------------------------------------------------------------


class TestLoaderContract:
    def test_load_returns_list(self) -> None:
        loader = _StubLoader()
        result = loader.load("/some/path")
        assert isinstance(result, list)


class TestParserContract:
    def test_parse_returns_parsed_document(self) -> None:
        doc = Document(
            source_path=Path("/data/doc.pdf"),
            extension=".pdf",
            source_hash="a" * 64,
            size_bytes=100,
        )
        parser = _StubParser()
        result = parser.parse(doc)
        assert isinstance(result, ParsedDocument)
        assert result.source is doc


class TestNormalizerContract:
    def test_normalize_returns_normalized_document(self) -> None:
        doc = Document(
            source_path=Path("/data/doc.txt"),
            extension=".txt",
            source_hash="b" * 64,
            size_bytes=50,
        )
        parsed = ParsedDocument(source=doc, text="hello world", title="", num_pages=0)
        normalizer = _StubNormalizer()
        result = normalizer.normalize(parsed)
        assert isinstance(result, NormalizedDocument)
        assert result.source is parsed


class TestChunkerContract:
    def test_chunk_returns_list(self) -> None:
        doc = Document(
            source_path=Path("/data/doc.md"),
            extension=".md",
            source_hash="c" * 64,
            size_bytes=200,
        )
        parsed = ParsedDocument(source=doc, text="text", title="", num_pages=0)
        normalized = NormalizedDocument(source=parsed, text="text")
        chunker = _StubChunker()
        result = chunker.chunk(normalized)
        assert isinstance(result, list)


class TestEmbedderContract:
    def test_embed_output_length_matches_input(self) -> None:
        doc = Document(
            source_path=Path("/data/doc.html"),
            extension=".html",
            source_hash="d" * 64,
            size_bytes=300,
        )
        parsed = ParsedDocument(source=doc, text="text", title="", num_pages=0)
        normalized = NormalizedDocument(source=parsed, text="text")
        meta = ChunkMetadata(
            document_hash=doc.source_hash,
            source_path=doc.source_path,
            chunk_index=0,
            total_chunks=1,
            heading="",
            page_number=0,
            char_start=0,
            char_end=4,
        )
        chunks = [
            Chunk(chunk_id=f"{doc.source_hash}_000000", text="text", metadata=meta)
        ]
        embedder = _StubEmbedder()
        result = embedder.embed(chunks)
        assert len(result) == len(chunks)
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)


class TestVectorStoreContract:
    def test_count_returns_int(self) -> None:
        store = _StubVectorStore()
        assert isinstance(store.count(), int)

    def test_delete_returns_int(self) -> None:
        store = _StubVectorStore()
        assert isinstance(store.delete_by_document("a" * 64), int)

    def test_upsert_returns_none(self) -> None:
        store = _StubVectorStore()
        result = store.upsert([])
        assert result is None
