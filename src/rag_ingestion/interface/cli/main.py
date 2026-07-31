from __future__ import annotations

import argparse
import sys

from rag_ingestion import __version__
from rag_ingestion.bootstrap import bootstrap
from rag_ingestion.logging import get_logger


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="rag-ingestion",
        description="Production-grade RAG ingestion pipeline.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Display the active configuration.",
    )

    return parser


def main() -> int:
    """
    CLI entry point.
    """
    parser = build_parser()
    args = parser.parse_args()

    context = bootstrap()

    logger = get_logger(component="cli")

    if args.config:
        logger.info(
            "configuration_loaded",
            app_name=context.settings.app_name,
            environment=context.settings.app_env,
            embedding_model=context.settings.embedding_model,
            chunk_size=context.settings.chunk_size,
            chunk_overlap=context.settings.chunk_overlap,
            chroma_collection=context.settings.chroma_collection,
        )

        print(context.settings.model_dump_json(indent=2))
        return 0

    parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())