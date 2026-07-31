"""
Domain protocol definitions.

This package contains the abstract interfaces (Protocols) that define the
contracts for each stage of the ingestion pipeline.

Every infrastructure implementation must satisfy one of these protocols.

Planned protocols:

- Loader
- Parser
- Normalizer
- Chunker
- MetadataEnricher
- Embedder
- VectorStore

The application layer depends exclusively on these abstractions, allowing
infrastructure implementations to be substituted without modifying
orchestrators or business logic.
"""

__all__: list[str] = []