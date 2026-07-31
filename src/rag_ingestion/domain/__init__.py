"""
Domain layer.

The domain layer contains the core business abstractions of the RAG ingestion
pipeline. It is independent of infrastructure concerns and defines the
entities and protocols that the application depends upon.

Dependency direction:

    Domain
        ↑
Application
        ↑
Infrastructure
        ↑
Interface

No code in this package may depend on the Application, Infrastructure,
or Interface layers.
"""

__all__: list[str] = []