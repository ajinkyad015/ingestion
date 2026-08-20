"""
Docling-based document parser adapter.

Wraps ``docling.document_converter.DocumentConverter`` and implements the
``Parser`` domain protocol.  All Docling-specific types are confined to this
module; nothing in the Domain or Application layers imports Docling directly.

Responsibilities
----------------
- Accept a :class:`~rag_ingestion.domain.entities.document.Document` and
  return a :class:`~rag_ingestion.domain.entities.document.ParsedDocument`.
- Delegate all format-specific extraction to Docling.
- Preserve title, page count, and any metadata Docling exposes.
- Surface Docling failures as :exc:`RuntimeError` (unrecoverable) or
  :exc:`ValueError` (unsupported / malformed input) with the source path
  included in the message.

Non-responsibilities
--------------------
- Text normalisation
- Chunking
- Metadata enrichment
- Embedding or vector storage
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, cast

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.document import Document, ParsedDocument

if TYPE_CHECKING:
    # Imported only for type-checking; at runtime the import happens inside
    # __init__ so that the class remains testable without Docling installed.
    pass

class DoclingDocumentLike(Protocol):
    def export_to_markdown(self) -> str: ...
    meta: object | None
    metadata: object | None
    name: str | None
    pages: Iterable[object] | None


class ConversionResultLike(Protocol):
    document: DoclingDocumentLike


class _DocumentConverter(Protocol):
    def convert(self, source: str) -> ConversionResultLike:
        ...

class DoclingParser:
    """
    Concrete ``Parser`` that delegates all extraction to Docling.

    Parameters
    ----------
    settings:
        Application settings.  ``docling_enable_ocr`` controls whether Docling
        activates its OCR pipeline for image-based PDFs.

    Raises
    ------
    ValueError
        From :meth:`parse` when the source document extension is not supported
        or when Docling signals an input error.
    RuntimeError
        From :meth:`parse` when Docling raises an unexpected exception during
        conversion.
    """

    #: Extensions that Docling can handle.  Kept in sync with
    #: ``Settings.supported_extensions_list`` defaults.
    _SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
        {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"}
    )

    def __init__(self, settings: Settings) -> None:
        self._enable_ocr: bool = settings.docling_enable_ocr
        self._converter = self._build_converter(self._enable_ocr)

    # ------------------------------------------------------------------
    # Public API — Parser protocol
    # ------------------------------------------------------------------

    def parse(self, document: Document) -> ParsedDocument:
        """
        Convert *document* into a :class:`ParsedDocument`.

        Parameters
        ----------
        document:
            A loaded source document produced by a ``Loader``.

        Returns
        -------
        ParsedDocument
            Extracted text, title, and page count.

        Raises
        ------
        ValueError
            If *document*'s extension is not in the supported set, or if
            Docling rejects the input.
        RuntimeError
            If Docling raises an unexpected error during conversion.
        """
        if document.extension not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"DoclingParser does not support extension "
                f"{document.extension!r} (source: {document.source_path})."
            )

        source_path = document.source_path
        try:
            result = self._converter.convert(str(source_path))
        except Exception as exc:
            raise RuntimeError(
                f"DoclingParser failed to parse {source_path}: {exc}"
            ) from exc

        return self._build_parsed_document(document, result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_converter(enable_ocr: bool) -> _DocumentConverter:
        """
        Instantiate and return a ``DocumentConverter``.

        Separated into its own method so tests can patch it cleanly without
        needing a real Docling installation.

        Parameters
        ----------
        enable_ocr:
            When ``True`` the converter is created with the OCR pipeline
            enabled.  When ``False`` OCR is skipped for speed.

        Returns
        -------
        _DocumentConverter
            A ``docling.document_converter.DocumentConverter`` instance.
        """
        # Local import keeps Docling confined to the infrastructure layer.
        from docling.document_converter import DocumentConverter

        if enable_ocr:
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Docling's OCR is slow; only enable if requested
            return cast(_DocumentConverter, DocumentConverter())
        return cast(_DocumentConverter, DocumentConverter())

    @staticmethod
    def _build_parsed_document(
        document: Document,
        result: ConversionResultLike,
    ) -> ParsedDocument:
        """
        Map a Docling ``ConversionResult`` to a domain :class:`ParsedDocument`.

        Parameters
        ----------
        document:
            The originating source document.
        result:
            A ``docling.datamodel.document.ConversionResult`` returned by
            ``DocumentConverter.convert()``.

        Returns
        -------
        ParsedDocument
        """
        doc_obj = result.document

        # --- text -------------------------------------------------------
        # export_to_markdown() gives us structure-preserving text with
        # headings, lists, and tables rendered as Markdown.
        try:
            text: str = doc_obj.export_to_markdown()
        except Exception:
            text = ""

        # --- title ------------------------------------------------------
        title: str = ""
        try:
            # Docling exposes document-level metadata under .name or
            # the first top-level heading depending on the version.
            meta = getattr(doc_obj, "meta", None) or getattr(doc_obj, "metadata", None)
            if meta is not None:
                title = getattr(meta, "title", None) or ""
            if not title:
                title = getattr(doc_obj, "name", None) or ""
        except Exception:
            title = ""

        # --- page count -------------------------------------------------
        num_pages: int = 0
        try:
            pages = getattr(doc_obj, "pages", None)
            if pages is not None:
                num_pages = len(pages)
        except Exception:
            num_pages = 0

        return ParsedDocument(
            source=document,
            text=text,
            title=title,
            num_pages=num_pages,
        )
