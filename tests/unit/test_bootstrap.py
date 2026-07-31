from rag_ingestion.bootstrap import ApplicationContext, bootstrap
from rag_ingestion.config import get_settings


def test_bootstrap_returns_application_context() -> None:
    context = bootstrap()

    assert isinstance(context, ApplicationContext)


def test_bootstrap_uses_cached_settings() -> None:
    context = bootstrap()

    assert context.settings is get_settings()


def test_bootstrap_contains_settings() -> None:
    context = bootstrap()

    assert context.settings.app_name == "rag-ingestion"
    assert context.settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert context.settings.chunk_size == 512