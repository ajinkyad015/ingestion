from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from rag_ingestion import __version__
from rag_ingestion.application.orchestrators.ingestion import IngestionResult
from rag_ingestion.bootstrap import bootstrap
from rag_ingestion.logging import get_logger

# Single shared console so progress bars and output are rendered on the same stream.
console = Console(stderr=False)

# ---------------------------------------------------------------------------
# Stage labels — order matches the orchestrator pipeline
# ---------------------------------------------------------------------------

_STAGES = [
    ("Loading documents", "load"),
    ("Parsing",           "parse"),
    ("Chunking",          "chunk"),
    ("Embedding",         "embed"),
    ("Persisting",        "vector_store"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_text(has_failures: bool) -> Text:
    if has_failures:
        return Text("COMPLETED WITH ERRORS", style="bold yellow")
    return Text("SUCCESS", style="bold green")


def _render_summary(
    result: IngestionResult,
    *,
    collection: str,
    persist_dir: str,
) -> None:
    """Render the post-run ingestion summary panel to the console."""
    has_failures = bool(result.failures)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()

    rows: list[tuple[str, str | Text]] = [
        ("Status",               _status_text(has_failures)),
        ("Documents processed",  str(result.documents_processed)),
        ("Chunks generated",     str(result.chunks_created)),
        ("Embeddings created",   str(result.embeddings_created)),
        ("Failed documents",     _format_failures(len(result.failures))),
        ("Execution time",       f"{result.elapsed_time_seconds:.2f}s"),
        ("Collection",           collection),
        ("Persistence location", persist_dir),
    ]

    for label, value in rows:
        grid.add_row(label, value)

    border_style = "yellow" if has_failures else "green"
    console.print(
        Panel(
            grid,
            title="[bold]Ingestion Summary[/bold]",
            border_style=border_style,
            expand=False,
            padding=(1, 2),
        )
    )


def _format_failures(count: int) -> Text:
    if count == 0:
        return Text("0", style="green")
    return Text(str(count), style="bold red")


def _render_failure_report(result: IngestionResult) -> None:
    """Display a structured table of all failed documents."""
    if not result.failures:
        return

    table = Table(
        title="[bold red]Failed Documents[/bold red]",
        show_header=True,
        header_style="bold",
        border_style="red",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("Document", overflow="fold")
    table.add_column("Stage", style="yellow", no_wrap=True)
    table.add_column("Error", overflow="fold")

    for idx, failure in enumerate(result.failures, start=1):
        table.add_row(
            str(idx),
            str(Path(failure.document_path).name),
            failure.stage,
            failure.error,
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
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

    subparsers = parser.add_subparsers(dest="command")
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest_parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Path to the input directory to ingest.",
    )

    return parser


def _run_ingest(target: str, collection: str, persist_dir: str) -> IngestionResult:
    """
    Bootstrap the pipeline, run the progress display, then call the orchestrator.

    The Rich progress bar cycles through the five stage labels sequentially to
    give the user continuous visual feedback.  The actual orchestrator is a
    single blocking call; we cannot hook individual stage transitions without
    modifying the pipeline, so the spinner advances by stage name as each
    phase is *expected* to start.
    """
    context = bootstrap()
    logger = get_logger(component="cli")

    logger.info(
        "ingestion_started",
        source=target,
        embedding_model=context.settings.embedding_model,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,          # clears the bar when done; summary replaces it
    ) as progress:
        overall = progress.add_task("[cyan]Ingesting…", total=len(_STAGES))

        # Advance through stage labels in lockstep with the pipeline phases.
        # Because the orchestrator is a single blocking call, we show each
        # stage label as a *current* step then hand off to the next.
        stage_tasks = [
            progress.add_task(f"[dim]{label}", total=1, visible=False)
            for label, _ in _STAGES
        ]

        def _advance_to(stage_idx: int) -> None:
            if stage_idx > 0:
                progress.update(stage_tasks[stage_idx - 1], completed=1, visible=False)
            if stage_idx < len(stage_tasks):
                progress.update(stage_tasks[stage_idx], visible=True)
                progress.update(overall, description=f"[cyan]{_STAGES[stage_idx][0]}")
                progress.advance(overall)

        # Kick off the visual sequence before blocking on ingest().
        for i in range(len(_STAGES)):
            _advance_to(i)

        result: IngestionResult = context.orchestrator.ingest(target)

    return result


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        context = bootstrap()
        logger = get_logger(component="cli")
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

    if args.command == "ingest":
        # Resolve target *before* bootstrapping so we can show it early.
        context_settings = bootstrap().settings
        target = args.source or str(context_settings.default_input_directory)
        collection = context_settings.chroma_collection
        persist_dir = str(context_settings.chroma_persist_directory)

        console.print(
            f"\n[bold]rag-ingestion[/bold] [dim]v{__version__}[/dim]"
            f"  →  [cyan]{target}[/cyan]\n"
        )

        try:
            result = _run_ingest(target, collection, persist_dir)
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Pipeline error:[/bold red] {exc}")
            return 1

        _render_summary(result, collection=collection, persist_dir=persist_dir)
        _render_failure_report(result)

        # Requirement 4: non-zero exit if any document failed.
        return 1 if result.failures else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())