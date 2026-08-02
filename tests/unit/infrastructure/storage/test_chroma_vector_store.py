from __future__ import annotations

from pathlib import Path

import pytest

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata, EmbeddedChunk
from rag_ingestion.domain.protocols.vector_store import VectorStore
from rag_ingestion.infrastructure.storage.chroma import ChromaVectorStore


class _FakeCollection:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, object]] = []
        self.delete_calls: list[list[str]] = []
        self.get_calls: list[dict[str, object]] = []
        self._count = 2

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def get(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(kwargs)
        return {"ids": ["abc123_000000"]}

    def delete(self, ids: list[str]) -> None:
        self.delete_calls.append(ids)

    def count(self) -> int:
        return self._count


def _make_chunk(text: str) -> Chunk:
    metadata = ChunkMetadata(
        document_hash="abc123",
        source_path=Path("/docs/report.pdf"),
        chunk_index=0,
        total_chunks=1,
        heading="",
        page_number=0,
        char_start=0,
        char_end=len(text),
    )
    return Chunk(chunk_id="abc123_000000", text=text, metadata=metadata)


class TestChromaVectorStore:
    def test_protocol_conformance(self) -> None:
        settings = Settings(CHROMA_PERSIST_DIRECTORY="/tmp/chroma", CHROMA_COLLECTION="documents")
        store = ChromaVectorStore(settings)
        assert isinstance(store, VectorStore)

    def test_upsert_requires_at_least_one_chunk(self) -> None:
        settings = Settings(CHROMA_PERSIST_DIRECTORY="/tmp/chroma", CHROMA_COLLECTION="documents")
        store = ChromaVectorStore(settings)

        with pytest.raises(ValueError, match="requires at least one embedded chunk"):
            store.upsert([])

    def test_upsert_persists_ids_documents_embeddings_and_metadata(self) -> None:
        settings = Settings(CHROMA_PERSIST_DIRECTORY="/tmp/chroma", CHROMA_COLLECTION="documents")
        store = ChromaVectorStore(settings)
        collection = _FakeCollection()
        store._collection = collection

        embedded_chunk = EmbeddedChunk(chunk=_make_chunk("hello"), embedding=(0.1, 0.2))
        store.upsert([embedded_chunk])

        assert len(collection.upsert_calls) == 1
        call = collection.upsert_calls[0]
        assert call["ids"] == ["abc123_000000"]
        assert call["documents"] == ["hello"]
        assert call["embeddings"] == [[0.1, 0.2]]
        assert call["metadatas"][0]["document_hash"] == "abc123"
        assert call["metadatas"][0]["chunk_id"] == "abc123_000000"

    def test_delete_by_document_returns_deleted_count(self) -> None:
        settings = Settings(CHROMA_PERSIST_DIRECTORY="/tmp/chroma", CHROMA_COLLECTION="documents")
        store = ChromaVectorStore(settings)
        collection = _FakeCollection()
        store._collection = collection

        deleted = store.delete_by_document("abc123")

        assert deleted == 1
        assert collection.get_calls[0]["where"] == {"document_hash": "abc123"}
        assert collection.delete_calls == [["abc123_000000"]]

    def test_count_returns_collection_count(self) -> None:
        settings = Settings(CHROMA_PERSIST_DIRECTORY="/tmp/chroma", CHROMA_COLLECTION="documents")
        store = ChromaVectorStore(settings)
        collection = _FakeCollection()
        store._collection = collection

        assert store.count() == 2
