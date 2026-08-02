from __future__ import annotations

from dataclasses import dataclass

from rag_ingestion.application.orchestrators.ingestion import IngestionOrchestrator
from rag_ingestion.config import Settings, get_settings
from rag_ingestion.infrastructure.chunkers.recursive import RecursiveChunker
from rag_ingestion.infrastructure.embedders.sentence_transformer import SentenceTransformerEmbedder
from rag_ingestion.infrastructure.loaders.filesystem import FileSystemLoader
from rag_ingestion.infrastructure.metadata_enrichers.default import DefaultMetadataEnricher
from rag_ingestion.infrastructure.normalizers.default import DefaultDocumentNormalizer
from rag_ingestion.infrastructure.parsers.docling_parser import DoclingParser
from rag_ingestion.infrastructure.storage.chroma import ChromaVectorStore
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
    loader: FileSystemLoader
    parser: DoclingParser
    normalizer: DefaultDocumentNormalizer
    chunker: RecursiveChunker
    metadata_enricher: DefaultMetadataEnricher
    embedder: SentenceTransformerEmbedder
    vector_store: ChromaVectorStore
    orchestrator: IngestionOrchestrator


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

    loader = FileSystemLoader(settings)
    parser = DoclingParser(settings)
    normalizer = DefaultDocumentNormalizer()
    chunker = RecursiveChunker(settings)
    metadata_enricher = DefaultMetadataEnricher(settings)
    embedder = SentenceTransformerEmbedder(settings)
    vector_store = ChromaVectorStore(settings)
    orchestrator = IngestionOrchestrator(
        loader=loader,
        parser=parser,
        normalizer=normalizer,
        chunker=chunker,
        metadata_enricher=metadata_enricher,
        embedder=embedder,
        vector_store=vector_store,
    )

    logger.info(
        "application_bootstrapped",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.app_debug,
    )

    return ApplicationContext(
        settings=settings,
        loader=loader,
        parser=parser,
        normalizer=normalizer,
        chunker=chunker,
        metadata_enricher=metadata_enricher,
        embedder=embedder,
        vector_store=vector_store,
        orchestrator=orchestrator,
    )