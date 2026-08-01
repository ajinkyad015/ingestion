from __future__ import annotations

import re
import unicodedata

from rag_ingestion.domain.entities.document import NormalizedDocument, ParsedDocument
from rag_ingestion.domain.protocols.normalizer import Normalizer


class DefaultDocumentNormalizer(Normalizer):
    """
    Standard normalizer that cleans text and extracts sections.

    Normalizes whitespace, line endings, Unicode, and repeated blank lines.
    Preserves document structure by splitting content into sections based
    on Markdown headings.
    """

    def normalize(self, document: ParsedDocument) -> NormalizedDocument:
        text = document.text

        # 1. Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Line endings to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Normalize non-breaking spaces
        text = text.replace("\xa0", " ")

        # 4. Remove trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 5. Squeeze repeated blank lines (3+ newlines to 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 6. Trim entire document
        text = text.strip()

        sections = self._extract_sections(text)

        return NormalizedDocument(
            source=document,
            text=text,
            sections=sections,
        )

    def _extract_sections(self, text: str) -> tuple[tuple[str, str], ...]:
        if not text:
            return ()

        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_body: list[str] = []

        for line in text.split("\n"):
            # Match Markdown headings like "# Heading" up to "###### Heading"
            match = re.match(r"^(#{1,6})\s+(.*)", line)
            if match:
                # Flush previous section
                if current_body or current_heading:
                    sections.append((current_heading, "\n".join(current_body).strip()))
                current_heading = match.group(2).strip()
                current_body = []
            else:
                current_body.append(line)

        # Flush final section
        if current_body or current_heading:
            sections.append((current_heading, "\n".join(current_body).strip()))

        return tuple(sections)
