from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_ingestion.domain.entities.chunk import EmbeddedChunk


@runtime_checkable
class VectorStore(Protocol):
    """
    Contract for persisting and retrieving embedded document chunks.

    A VectorStore stores ``EmbeddedChunk`` objects and supports
    retrieval and deletion by document identifier.

    The concrete implementation uses ChromaDB; this protocol keeps the
    application layer decoupled from that dependency.
    """

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Insert or replace *embedded_chunks* in the store.

        When a chunk with the same ``chunk_id`` already exists, the stored
        embedding, text, and metadata are replaced with the new values.

        Parameters
        ----------
        embedded_chunks:
            Non-empty list of embedded chunks to persist.

        Raises
        ------
        ValueError
            If *embedded_chunks* is empty.
        RuntimeError
            If the underlying store operation fails.
        """
        ...

    def delete_by_document(self, document_hash: str) -> int:
        """
        Delete all chunks belonging to the document identified by
        *document_hash*.

        Parameters
        ----------
        document_hash:
            SHA-256 hex digest of the source document.

        Returns
        -------
        int
            Number of chunks deleted.  Zero when the document is not
            found in the store.
        """
        ...

    def count(self) -> int:
        """
        Return the total number of chunks currently held in the store.

        Returns
        -------
        int
            Total chunk count.
        """
        ...
