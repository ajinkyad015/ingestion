from __future__ import annotations

import logging
from typing import cast
from structlog.stdlib import BoundLogger
import sys
from typing import Any

import structlog

from rag_ingestion.config import Settings


def configure_logging(settings: Settings) -> None:
    """
    Configure application-wide structured logging.

    This function must be called exactly once during application startup.
    """

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(),
            ]
        )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
        force=True,
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(**context: Any) -> structlog.stdlib.BoundLogger:
    """
    Return a structured logger with optional bound context.

    Parameters
    ----------
    **context
        Key-value pairs attached to every emitted log entry.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Configured structured logger.
    """
    logger = cast(BoundLogger, structlog.get_logger())
    
    if context:
        return logger.bind(**context)
    

    return logger