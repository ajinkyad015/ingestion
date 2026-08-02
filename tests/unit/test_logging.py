
from rag_ingestion.config import Settings
from rag_ingestion.logging import configure_logging, get_logger


def test_logger_can_be_created() -> None:
    settings = Settings()

    configure_logging(settings)

    logger = get_logger(component="unit-test")

    assert logger is not None


def test_logger_supports_bound_context() -> None:
    settings = Settings()

    configure_logging(settings)

    logger = get_logger(component="logging-test", pipeline="ingestion")

    assert logger is not None


def test_json_logging_configuration() -> None:
    settings = Settings(
        LOG_FORMAT="json",
    )

    configure_logging(settings)

    logger = get_logger(component="json-test")

    rendered = logger.info(
        "test_event",
        key="value",
    )

    # structlog.info() returns None after emitting the log
    assert rendered is None


def test_console_logging_configuration() -> None:
    settings = Settings(
        LOG_FORMAT="console",
    )

    configure_logging(settings)

    logger = get_logger(component="console-test")

    rendered = logger.info(
        "console_event",
        status="ok",
    )

    assert rendered is None