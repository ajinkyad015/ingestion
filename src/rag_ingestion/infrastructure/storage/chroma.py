from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import EmbeddedChunk
from rag_ingestion.domain.protocols.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """
    Persist embedded chunks in a ChromaDB collection.

    The underlying client and collection are created lazily on first use so that
    the application can be imported even when ChromaDB is not yet installed.
    """

    def __init__(self, settings: Settings) -> None:
        self._persist_directory = Path(settings.chroma_persist_directory)
        self._collection_name = settings.chroma_collection
        self._client: Any | None = None
        self._collection: Any | None = None

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            raise ValueError("ChromaVectorStore.upsert() requires at least one embedded chunk.")

        collection = self._get_collection()
        ids = [chunk.chunk_id for chunk in (item.chunk for item in embedded_chunks)]
        documents = [item.chunk.text for item in embedded_chunks]
        embeddings = [list(item.embedding) for item in embedded_chunks]
        metadatas = [self._build_metadata(item) for item in embedded_chunks]

        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def delete_by_document(self, document_hash: str) -> int:
        collection = self._get_collection()
        result = collection.get(where={"document_hash": document_hash}, include=[])
        ids = result.get("ids", [])
        if not ids:
            return 0

        collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        collection = self._get_collection()
        return int(collection.count())

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._client = self._build_client()
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _build_client(self) -> Any:
        try:
            import chromadb  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - exercised from runtime import failures
            raise RuntimeError("ChromaVectorStore requires chromadb to be installed.") from exc

        self._persist_directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._persist_directory))

    @staticmethod
    def _build_metadata(item: EmbeddedChunk) -> dict[str, object]:
        metadata = item.chunk.metadata
        return {
            "document_hash": metadata.document_hash,
            "source_path": str(metadata.source_path),
            "chunk_index": metadata.chunk_index,
            "total_chunks": metadata.total_chunks,
            "heading": metadata.heading,
            "page_number": metadata.page_number,
            "char_start": metadata.char_start,
            "char_end": metadata.char_end,
            "source_file": metadata.source_file,
            "section_hierarchy": json.dumps(metadata.section_hierarchy),
            "title": metadata.title,
            "document_created_at": metadata.document_created_at.isoformat()
            if metadata.document_created_at is not None
            else "",
            "document_modified_at": metadata.document_modified_at.isoformat()
            if metadata.document_modified_at is not None
            else "",
            "document_accessed_at": metadata.document_accessed_at.isoformat()
            if metadata.document_accessed_at is not None
            else "",
            "chunk_hash": metadata.chunk_hash,
            "chunk_id": item.chunk.chunk_id,
        }
