"""
Application services.

Application services encapsulate reusable business workflows that support
orchestrators while remaining independent of infrastructure concerns.

Unlike infrastructure adapters, application services coordinate domain
operations and enforce application-specific policies.

Planned services include:

- IngestionService
- MetadataService
- DocumentHashService
- ChunkHashService

These services depend only on domain entities and protocols.
They must never import infrastructure implementations directly.
"""

__all__: list[str] = []