"""
Interface layer.

The interface layer exposes the application's capabilities to external
consumers such as command-line interfaces, HTTP APIs, background workers,
or scheduled jobs.

Version 1 exposes a CLI only.

Responsibilities:

- Accept user input
- Validate CLI arguments
- Invoke the application layer
- Format output
- Translate application exceptions into user-friendly messages

Dependency direction:

    Domain
        ↑
    Application
        ↑
    Infrastructure
        ↑
    Interface

The interface layer is the outermost layer of the application and is the
only layer that interacts directly with end users.
"""

__all__: list[str] = []