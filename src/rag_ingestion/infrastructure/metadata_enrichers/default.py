from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata
from rag_ingestion.domain.protocols.metadata_enricher import MetadataEnricher


class DefaultMetadataEnricher(MetadataEnricher):
    """
    Deterministically enrich chunk metadata.

    The enricher adds document-level and chunk-level provenance details while
    preserving chunk ordering and text content.
    """

    def __init__(self, settings: Settings) -> None:
        self._hash_algorithm = settings.hash_algorithm

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        enriched_chunks: list[Chunk] = []
        for chunk in chunks:
            enriched_metadata = self._enrich_metadata(chunk)
            enriched_chunks.append(
                replace(chunk, metadata=enriched_metadata)
            )
        return enriched_chunks

    def _enrich_metadata(self, chunk: Chunk) -> ChunkMetadata:
        metadata = chunk.metadata
        source_path = metadata.source_path

        source_file = metadata.source_file or source_path.name
        title = metadata.title or source_path.stem
        section_hierarchy = metadata.section_hierarchy or self._build_section_hierarchy(
            metadata.heading
        )
        document_created_at, document_modified_at, document_accessed_at = self._resolve_timestamps(
            source_path,
            metadata.document_created_at,
            metadata.document_modified_at,
            metadata.document_accessed_at,
        )

        document_hash = self._build_document_hash(
            source_path=source_path,
            source_file=source_file,
            title=title,
            document_created_at=document_created_at,
            document_modified_at=document_modified_at,
            document_accessed_at=document_accessed_at,
            seed=metadata.document_hash,
        )
        chunk_hash = self._build_chunk_hash(
            document_hash=document_hash,
            chunk=chunk,
            source_file=source_file,
            title=title,
            section_hierarchy=section_hierarchy,
            document_created_at=document_created_at,
            document_modified_at=document_modified_at,
            document_accessed_at=document_accessed_at,
        )

        return replace(
            metadata,
            document_hash=document_hash,
            source_file=source_file,
            section_hierarchy=section_hierarchy,
            title=title,
            document_created_at=document_created_at,
            document_modified_at=document_modified_at,
            document_accessed_at=document_accessed_at,
            chunk_hash=chunk_hash,
        )

    def _build_document_hash(
        self,
        *,
        source_path: Path,
        source_file: str,
        title: str,
        document_created_at: datetime | None,
        document_modified_at: datetime | None,
        document_accessed_at: datetime | None,
        seed: str,
    ) -> str:
        payload = self._join_parts(
            [
                "document",
                seed,
                str(source_path),
                source_file,
                title,
                self._format_timestamp(document_created_at),
                self._format_timestamp(document_modified_at),
                self._format_timestamp(document_accessed_at),
            ]
        )
        return self._hash(payload)

    def _build_chunk_hash(
        self,
        *,
        document_hash: str,
        chunk: Chunk,
        source_file: str,
        title: str,
        section_hierarchy: tuple[str, ...],
        document_created_at: datetime | None,
        document_modified_at: datetime | None,
        document_accessed_at: datetime | None,
    ) -> str:
        metadata = chunk.metadata
        payload = self._join_parts(
            [
                "chunk",
                document_hash,
                str(metadata.chunk_index),
                str(metadata.total_chunks),
                str(metadata.source_path),
                source_file,
                title,
                metadata.heading,
                str(metadata.page_number),
                str(metadata.char_start),
                str(metadata.char_end),
                ":".join(section_hierarchy),
                self._format_timestamp(document_created_at),
                self._format_timestamp(document_modified_at),
                self._format_timestamp(document_accessed_at),
                chunk.text,
            ]
        )
        return self._hash(payload)

    def _hash(self, payload: str) -> str:
        return hashlib.new(self._hash_algorithm, payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _join_parts(parts: list[str]) -> str:
        return "\u241f".join(parts)

    @staticmethod
    def _format_timestamp(value: datetime | None) -> str:
        if value is None:
            return ""
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _build_section_hierarchy(heading: str) -> tuple[str, ...]:
        if not heading:
            return ()
        return (heading,)

    @staticmethod
    def _resolve_timestamps(
        source_path: Path,
        created_at: datetime | None,
        modified_at: datetime | None,
        accessed_at: datetime | None,
    ) -> tuple[datetime | None, datetime | None, datetime | None]:
        if created_at is not None and modified_at is not None and accessed_at is not None:
            return created_at, modified_at, accessed_at

        try:
            stat_result = source_path.stat()
        except OSError:
            return created_at, modified_at, accessed_at

        resolved_created_at = created_at or datetime.fromtimestamp(
            getattr(stat_result, "st_birthtime", stat_result.st_ctime),
            tz=timezone.utc,
        )
        resolved_modified_at = modified_at or datetime.fromtimestamp(
            stat_result.st_mtime,
            tz=timezone.utc,
        )
        resolved_accessed_at = accessed_at or datetime.fromtimestamp(
            stat_result.st_atime,
            tz=timezone.utc,
        )
        return resolved_created_at, resolved_modified_at, resolved_accessed_at