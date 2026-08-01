from __future__ import annotations

import re

from rag_ingestion.config import Settings
from rag_ingestion.domain.entities.chunk import Chunk, ChunkMetadata
from rag_ingestion.domain.entities.document import NormalizedDocument
from rag_ingestion.domain.protocols.chunker import Chunker


class RecursiveChunker(Chunker):
    """
    Splits normalized documents into chunks using a deterministic recursive strategy.

    Configured via application Settings for chunk size and overlap.
    Preserves document structure by splitting section by section, ensuring
    headings are kept with their immediately following content unless a
    section alone exceeds the chunk size.
    """

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        # Standard recursive split separators
        self._separators = ["\n\n", "\n", " ", ""]

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """
        Split the normalized document into an ordered list of chunks.
        """
        if not document.text:
            return []

        section_blocks = self._extract_section_blocks(document.text)
        chunks: list[Chunk] = []
        chunk_index = 0
        doc_hash = document.source.source.source_hash
        source_path = document.source.source.source_path

        # Generate all raw splits with offsets
        raw_chunks = []
        for heading, section_text, start_offset, end_offset in section_blocks:
            if not section_text.strip():
                continue
                
            splits = self._split_text(
                text=section_text,
                start_offset=start_offset,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=self._separators,
            )
            
            for chunk_str, chunk_start, chunk_end in splits:
                raw_chunks.append((chunk_str, chunk_start, chunk_end, heading))
                
        total_chunks = len(raw_chunks)
        
        for chunk_str, chunk_start, chunk_end, heading in raw_chunks:
            # Deterministic chunk ID: {document_hash}_{chunk_index:06d}
            chunk_id = f"{doc_hash}_{chunk_index:06d}"
            
            metadata = ChunkMetadata(
                document_hash=doc_hash,
                source_path=source_path,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                heading=heading,
                page_number=0,
                char_start=chunk_start,
                char_end=chunk_end,
            )
            
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    metadata=metadata,
                )
            )
            chunk_index += 1

        return chunks

    def _extract_section_blocks(self, text: str) -> list[tuple[str, str, int, int]]:
        """
        Finds section boundaries in the full normalized text.
        Returns a list of (heading, section_text, char_start, char_end).
        """
        blocks = []
        if not text:
            return blocks
            
        current_heading = ""
        section_start = 0
        lines = text.split("\n")
        offset = 0
        
        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,6})\s+(.*)", line)
            if match:
                if i > 0:
                    block_text = text[section_start:offset]
                    blocks.append((current_heading, block_text, section_start, offset))
                current_heading = match.group(2).strip()
                section_start = offset
                
            offset += len(line) + 1 # +1 for newline
            
        if offset > section_start:
            end_idx = min(offset, len(text))
            block_text = text[section_start:end_idx]
            blocks.append((current_heading, block_text, section_start, end_idx))
            
        return blocks

    def _split_text(
        self,
        text: str,
        start_offset: int,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
    ) -> list[tuple[str, int, int]]:
        if not text:
            return []
            
        if len(text) <= chunk_size:
            if text.strip():
                return [(text, start_offset, start_offset + len(text))]
            return []
            
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                new_separators = separators[i+1:]
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break
                
        if separator == "":
            splits = [c for c in text]
        else:
            splits = text.split(separator)
            
        splits_with_offsets = []
        current_off = start_offset
        for s in splits:
            splits_with_offsets.append((s, current_off))
            current_off += len(s) + len(separator)
            
        chunks = []
        current_splits = []
        current_len = 0
        
        for s_text, s_off in splits_with_offsets:
            if current_len > 0 and current_len + len(separator) + len(s_text) > chunk_size:
                chunk_str = separator.join([x[0] for x in current_splits])
                chunk_start = current_splits[0][1]
                
                if len(chunk_str) > chunk_size and new_separators:
                    chunks.extend(self._split_text(chunk_str, chunk_start, chunk_size, chunk_overlap, new_separators))
                elif chunk_str.strip():
                    chunks.append((chunk_str, chunk_start, chunk_start + len(chunk_str)))
                    
                overlap_splits = []
                overlap_len = 0
                for xs_text, xs_off in reversed(current_splits):
                    addition = len(xs_text) if overlap_len == 0 else len(separator) + len(xs_text)
                    if overlap_len + addition > chunk_overlap:
                        break
                    overlap_splits.insert(0, (xs_text, xs_off))
                    overlap_len += addition
                    
                current_splits = overlap_splits
                current_splits.append((s_text, s_off))
                current_len = overlap_len + len(s_text) if overlap_len == 0 else overlap_len + len(separator) + len(s_text)
            else:
                current_splits.append((s_text, s_off))
                current_len += len(s_text) if current_len == 0 else len(separator) + len(s_text)
                
        if current_splits:
            chunk_str = separator.join([x[0] for x in current_splits])
            chunk_start = current_splits[0][1]
            if len(chunk_str) > chunk_size and new_separators:
                chunks.extend(self._split_text(chunk_str, chunk_start, chunk_size, chunk_overlap, new_separators))
            elif chunk_str.strip():
                chunks.append((chunk_str, chunk_start, chunk_start + len(chunk_str)))
                
        return chunks
