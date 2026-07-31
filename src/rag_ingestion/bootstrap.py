from __future__ import annotations

from dataclasses import dataclass

from rag_ingestion.config import Settings, get_settings
from rag_ingestion.logging import configure_logging, get_logger


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """
    Root dependency container for the application.

    Infrastructure adapters and application services are composed here.
    Additional dependencies will be introduced in later commits while
    preserving constructor injection throughout the application.
    """

    settings: Settings


def bootstrap() -> ApplicationContext:
    """
    Build the application dependency graph.

    This function serves as the single composition root for the
    application. All infrastructure implementations are instantiated
    here and injected into higher-level services.

    Returns
    -------
    ApplicationContext
        Configured application context.
    """
    settings = get_settings()

    configure_logging(settings)

    logger = get_logger(component="bootstrap")

    logger.info(
        "application_bootstrapped",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.app_debug,
    )

    return ApplicationContext(
        settings=settings,
    )