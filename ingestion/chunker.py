"""Heading-aware text chunker with source offset and hierarchy tracking.

Splits technical documents (Markdown/text) into bounded passages while
retaining exact character offsets in the raw document for citation verification.
"""

from __future__ import annotations

import hashlib
import re
from uuid import UUID

from ingestion.normalizer import PersianNormalizer
from retrieval.protocols import Chunk, SourceLocation

# Markdown heading pattern (# Heading)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?$", re.MULTILINE)

# Default chunking parameters (tunable per roadmap specification)
DEFAULT_TARGET_CHUNK_CHARS = 1800  # approximately ~450-500 tokens
DEFAULT_OVERLAP_CHARS = 250        # approximately ~60-75 tokens
DEFAULT_MIN_CHUNK_CHARS = 120      # avoid tiny fragments unless end of section


class HeadingAwareChunker:
    """Chunks documents while preserving heading hierarchy and exact raw character offsets."""

    def __init__(
        self,
        target_chunk_chars: int = DEFAULT_TARGET_CHUNK_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
        normalizer: PersianNormalizer | None = None,
    ) -> None:
        if overlap_chars >= target_chunk_chars:
            raise ValueError("overlap_chars must be strictly less than target_chunk_chars")
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.normalizer = normalizer or PersianNormalizer()

    def chunk(
        self,
        text: str,
        document_id: UUID,
        workspace_id: UUID,
        source_version: str,
        language: str = "en",
    ) -> list[Chunk]:
        """Split document text into location-tracked chunks."""
        if not text or not text.strip():
            return []

        sections = self._split_by_headings(text)
        raw_chunks: list[tuple[int, int, str, tuple[str, ...]]] = []

        for start_offset, end_offset, section_text, heading_path in sections:
            if len(section_text) <= self.target_chunk_chars:
                raw_chunks.append((start_offset, end_offset, section_text, heading_path))
            else:
                # Sub-divide large section by paragraphs/sentences with overlap
                sub_chunks = self._subdivide_text(
                    section_text,
                    base_offset=start_offset,
                    heading_path=heading_path,
                )
                raw_chunks.extend(sub_chunks)

        # Merge trailing tiny chunks with predecessor if feasible
        merged_chunks = self._merge_tiny_residuals(raw_chunks, text)

        # Build formal Chunk models
        result: list[Chunk] = []
        for idx, (start_c, end_c, content, h_path) in enumerate(merged_chunks):
            # Compute stable deterministic chunk identifier
            chunk_hash = hashlib.sha256(
                f"{document_id}:{source_version}:{start_c}:{end_c}".encode()
            ).hexdigest()[:16]
            chunk_id = f"chk_{chunk_hash}_{idx:04d}"

            search_text = self.normalizer.normalize_for_search(content)

            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    source_version=source_version,
                    content=content,
                    search_text=search_text,
                    location=SourceLocation(
                        start_char=start_c,
                        end_char=end_c,
                        heading_path=h_path,
                    ),
                    language=language,
                    metadata={"index": idx, "char_count": len(content)},
                )
            )

        return result

    def _split_by_headings(
        self, text: str
    ) -> list[tuple[int, int, str, tuple[str, ...]]]:
        """Find headings and return text slices along with their hierarchical path."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(0, len(text), text, ())]

        sections: list[tuple[int, int, str, tuple[str, ...]]] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title)

        # Text before first heading (if any)
        first_start = matches[0].start()
        if first_start > 0:
            preamble = text[:first_start]
            if preamble.strip():
                sections.append((0, first_start, preamble, ()))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Adjust hierarchy stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = tuple(t for _, t in heading_stack)

            section_body = text[start_pos:end_pos]
            sections.append((start_pos, end_pos, section_body, current_path))

        return sections

    def _subdivide_text(
        self,
        text: str,
        base_offset: int,
        heading_path: tuple[str, ...],
    ) -> list[tuple[int, int, str, tuple[str, ...]]]:
        """Subdivide a section using natural paragraph or sentence boundaries."""
        results: list[tuple[int, int, str, tuple[str, ...]]] = []
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + self.target_chunk_chars, total_len)

            if end < total_len:
                # Look for paragraph break
                para_break = text.rfind("\n\n", start + self.target_chunk_chars // 2, end)
                if para_break != -1 and para_break > start:
                    end = para_break + 2
                else:
                    # Look for sentence break (. ! ? followed by space or newline)
                    sentence_break = max(
                        text.rfind(". ", start + self.target_chunk_chars // 2, end),
                        text.rfind(".\n", start + self.target_chunk_chars // 2, end),
                        text.rfind("? ", start + self.target_chunk_chars // 2, end),
                        text.rfind("\n", start + self.target_chunk_chars // 2, end),
                    )
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1

            chunk_content = text[start:end]
            if chunk_content.strip():
                results.append(
                    (base_offset + start, base_offset + end, chunk_content, heading_path)
                )

            if end >= total_len:
                break

            # Advance with overlap
            next_start = max(start + 1, end - self.overlap_chars)
            # Try to align next_start with whitespace
            space_pos = text.find(" ", next_start, min(next_start + 50, end))
            start = space_pos + 1 if space_pos != -1 else next_start

        return results

    def _merge_tiny_residuals(
        self,
        chunks: list[tuple[int, int, str, tuple[str, ...]]],
        full_text: str,
    ) -> list[tuple[int, int, str, tuple[str, ...]]]:
        """Merge undersized tail chunks into their predecessor to prevent fragmentation."""
        if len(chunks) <= 1:
            return chunks

        merged: list[tuple[int, int, str, tuple[str, ...]]] = []
        for start_c, end_c, content, h_path in chunks:
            is_undersized = len(content.strip()) < self.min_chunk_chars
            same_heading = bool(merged and merged[-1][3] == h_path)
            fits_budget = bool(
                merged
                and (merged[-1][1] - merged[-1][0] + (end_c - start_c))
                <= (self.target_chunk_chars * 1.3)
            )
            if merged and is_undersized and same_heading and fits_budget:
                # Merge into previous chunk
                prev_start, _, _, prev_path = merged.pop()
                combined_content = full_text[prev_start:end_c]
                merged.append((prev_start, end_c, combined_content, prev_path))
            else:
                merged.append((start_c, end_c, content, h_path))

        return merged
