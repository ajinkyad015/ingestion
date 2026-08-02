from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata
from rag_ingestion.domain.protocols.metadata_enricher import MetadataEnricher
from rag_ingestion.infrastructure.metadata_enrichers.default import DefaultMetadataEnricher


def _make_settings() -> Settings:
    return Settings(HASH_ALGORITHM="sha256")


def _make_chunk(
    *,
    source_path: str = "/docs/report.pdf",
    document_hash: str = "a" * 64,
    chunk_index: int = 0,
    total_chunks: int = 2,
    heading: str = "Overview",
    page_number: int = 1,
    char_start: int = 0,
    char_end: int = 16,
    title: str = "",
    source_file: str = "",
    section_hierarchy: tuple[str, ...] = (),
    document_created_at: datetime | None = None,
    document_modified_at: datetime | None = None,
    document_accessed_at: datetime | None = None,
    chunk_hash: str = "",
    text: str = "Chunk text",
) -> Chunk:
    metadata = ChunkMetadata(
        document_hash=document_hash,
        source_path=Path(source_path),
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        heading=heading,
        page_number=page_number,
        char_start=char_start,
        char_end=char_end,
        source_file=source_file,
        section_hierarchy=section_hierarchy,
        title=title,
        document_created_at=document_created_at,
        document_modified_at=document_modified_at,
        document_accessed_at=document_accessed_at,
        chunk_hash=chunk_hash,
    )
    return Chunk(
        chunk_id=f"{document_hash}_{chunk_index:06d}",
        text=text,
        metadata=metadata,
    )


class TestDefaultMetadataEnricherProtocol:
    def test_satisfies_protocol(self) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        assert isinstance(enricher, MetadataEnricher)


class TestDefaultMetadataEnricherEnrich:
    def test_preserves_order_and_does_not_mutate_input(self) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        original_chunks = [
            _make_chunk(chunk_index=0, text="First chunk"),
            _make_chunk(chunk_index=1, text="Second chunk"),
        ]

        enriched_chunks = enricher.enrich(original_chunks)

        assert [chunk.text for chunk in enriched_chunks] == ["First chunk", "Second chunk"]
        assert [chunk.chunk_id for chunk in enriched_chunks] == [
            original_chunks[0].chunk_id,
            original_chunks[1].chunk_id,
        ]
        assert enriched_chunks is not original_chunks
        assert enriched_chunks[0] is not original_chunks[0]
        assert enriched_chunks[1] is not original_chunks[1]
        assert [chunk.text for chunk in original_chunks] == ["First chunk", "Second chunk"]

    def test_attaches_metadata_and_preserves_existing_values(self) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        modified_at = datetime(2026, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
        accessed_at = datetime(2026, 1, 4, 5, 6, 7, tzinfo=timezone.utc)
        chunk = _make_chunk(
            source_path="/docs/annual-report.pdf",
            heading="Section A",
            title="Annual Report",
            source_file="annual-report.pdf",
            section_hierarchy=("Part I", "Section A"),
            document_created_at=created_at,
            document_modified_at=modified_at,
            document_accessed_at=accessed_at,
        )

        enriched = enricher.enrich([chunk])[0]
        metadata = enriched.metadata

        assert metadata.source_path == chunk.metadata.source_path
        assert metadata.source_file == "annual-report.pdf"
        assert metadata.page_number == 1
        assert metadata.section_hierarchy == ("Part I", "Section A")
        assert metadata.title == "Annual Report"
        assert metadata.document_created_at == created_at
        assert metadata.document_modified_at == modified_at
        assert metadata.document_accessed_at == accessed_at
        assert enriched.text == chunk.text

    def test_deterministic_hashes_for_identical_input(self) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        chunk = _make_chunk()

        enriched_a = enricher.enrich([chunk])[0]
        enriched_b = enricher.enrich([chunk])[0]

        assert enriched_a.metadata == enriched_b.metadata
        assert enriched_a.metadata.document_hash
        assert enriched_a.metadata.chunk_hash

    def test_different_chunk_text_changes_only_chunk_hash(self) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        base_chunk = _make_chunk(text="Alpha")
        changed_text_chunk = _make_chunk(text="Beta")

        enriched_base = enricher.enrich([base_chunk])[0]
        enriched_changed = enricher.enrich([changed_text_chunk])[0]

        assert enriched_base.metadata.document_hash == enriched_changed.metadata.document_hash
        assert enriched_base.metadata.chunk_hash != enriched_changed.metadata.chunk_hash

    def test_different_documents_change_document_hash(self) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        first = _make_chunk(source_path="/docs/report-a.pdf")
        second = _make_chunk(source_path="/docs/report-b.pdf")

        enriched_first = enricher.enrich([first])[0]
        enriched_second = enricher.enrich([second])[0]

        assert enriched_first.metadata.document_hash != enriched_second.metadata.document_hash

    def test_timestamps_are_loaded_from_filesystem_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        enricher = DefaultMetadataEnricher(_make_settings())
        chunk = _make_chunk(source_path="/docs/timestamps.pdf")
        stat_result = SimpleNamespace(
            st_ctime=1_700_000_000,
            st_mtime=1_700_000_100,
            st_atime=1_700_000_200,
        )
        monkeypatch.setattr(Path, "stat", lambda self: stat_result)

        enriched = enricher.enrich([chunk])[0]

        assert enriched.metadata.document_created_at == datetime.fromtimestamp(
            1_700_000_000,
            tz=timezone.utc,
        )
        assert enriched.metadata.document_modified_at == datetime.fromtimestamp(
            1_700_000_100,
            tz=timezone.utc,
        )
        assert enriched.metadata.document_accessed_at == datetime.fromtimestamp(
            1_700_000_200,
            tz=timezone.utc,
        )