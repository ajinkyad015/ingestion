from rag_ingestion.config import Settings


def test_default_embedding_model() -> None:
    settings = Settings()

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


def test_supported_extensions_are_normalized() -> None:
    settings = Settings(
        SUPPORTED_EXTENSIONS=".PDF,.Md,.HTML"
    )

    assert settings.supported_extensions_list == (
        ".pdf",
        ".md",
        ".html",
    )


def test_chunk_configuration() -> None:
    settings = Settings(
        CHUNK_SIZE=1024,
        CHUNK_OVERLAP=128,
    )

    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 128


def test_default_collection_name() -> None:
    settings = Settings()

    assert settings.chroma_collection == "documents"


def test_default_persist_directory() -> None:
    settings = Settings()

    assert settings.chroma_persist_directory.name == "chroma"


def test_hash_algorithm() -> None:
    settings = Settings()

    assert settings.hash_algorithm == "sha256"