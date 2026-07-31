"""
Application orchestrators.

Orchestrators coordinate the end-to-end execution of application use cases.
They compose domain protocols through constructor injection and remain
agnostic to infrastructure implementations.

Planned orchestrators:

- IngestionOrchestrator

The orchestrator will eventually coordinate the following pipeline:

    Loader
        ↓
    Parser
        ↓
    Normalizer
        ↓
    Chunker
        ↓
    Metadata Enricher
        ↓
    Embedder
        ↓
    Vector Store

No orchestration logic should live in infrastructure adapters.
"""

__all__: list[str] = []