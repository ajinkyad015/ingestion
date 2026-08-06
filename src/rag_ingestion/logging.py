from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

from rag_ingestion.config import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_FILENAME = "rag_ingestion.log"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 5                # keep rag_ingestion.log + up to 5 rotated copies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(settings: Settings) -> None:
    """
    Configure application-wide structured logging to a rotating file.

    All log output is written exclusively to ``<LOG_DIRECTORY>/rag_ingestion.log``
    (rotated at 10 MB, keeping 5 backups).  No output is emitted to stdout or
    stderr, so the CLI terminal stays clean for Rich-rendered output.

    This function must be called exactly once during application startup.
    """
    log_dir: Path = Path(settings.log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / _LOG_FILENAME

    # -----------------------------------------------------------------------
    # stdlib handler — rotating file only, no StreamHandler
    # -----------------------------------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, settings.log_level))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove any handlers that may have been added before (e.g. by a previous
    # configure_logging call or by a third-party library that called basicConfig).
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    # Prevent propagation noise from third-party loggers (docling, chromadb,
    # sentence_transformers, etc.) reaching any surviving root handlers.
    for noisy in ("docling", "chromadb", "sentence_transformers", "torch", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # -----------------------------------------------------------------------
    # structlog processors
    # -----------------------------------------------------------------------
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # -----------------------------------------------------------------------
    # structlog — write through the stdlib file handler, not to stdout
    # -----------------------------------------------------------------------
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(**context: Any) -> BoundLogger:
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