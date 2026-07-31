"""
Application layer.

The application layer coordinates the ingestion workflow using the
interfaces defined by the domain layer.

It contains orchestration and use-case logic while remaining independent
of infrastructure implementations.

Responsibilities:

- Coordinate pipeline execution
- Invoke domain protocols
- Handle application-specific workflow
- Remain free of infrastructure concerns

The application layer depends only on the Domain layer.
"""

__all__: list[str] = []