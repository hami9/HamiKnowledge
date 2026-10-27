"""Retrieval and pipeline stage protocol interfaces for HamiKnowledge.

Every pipeline stage is an explicit, typed, swappable protocol without
framework lock-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True)
class SourceLocation:
    """Exact location of a text passage in its original source document."""

    start_char: int
    end_char: int
    page_number: int | None = None
    heading_path: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Chunk:
    """A bounded passage extracted from a source document version."""

    chunk_id: str
    document_id: UUID
    workspace_id: UUID
    source_version: str
    content: str
    search_text: str
    location: SourceLocation
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalFilter:
    """Mandatory tenant and document authorization filter applied at query level."""

    workspace_id: UUID
    allowed_document_ids: frozenset[UUID] | None = None
    excluded_document_ids: frozenset[UUID] | None = None


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate retrieved from an individual retrieval branch or stage."""

    chunk_id: str
    document_id: UUID
    workspace_id: UUID
    source_version: str
    content: str
    score: float
    rank: int
    branch: str  # "lexical", "vector", "fused", "reranked"
    location: SourceLocation
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TextNormalizer(Protocol):
    """Normalizes text for indexing and search while preserving raw text for citations."""

    def normalize_for_search(self, text: str) -> str:
        ...

    def clean_persian_letters(self, text: str) -> str:
        ...


@runtime_checkable
class TextChunker(Protocol):
    """Splits normalized document text into bounded, location-aware chunks."""

    def chunk(
        self,
        text: str,
        document_id: UUID,
        workspace_id: UUID,
        source_version: str,
        language: str = "en",
    ) -> list[Chunk]:
        ...


@runtime_checkable
class LexicalRetriever(Protocol):
    """Performs full-text search with mandatory tenant isolation in the query."""

    def retrieve_lexical(
        self,
        query: str,
        retrieval_filter: RetrievalFilter,
        limit: int = 30,
    ) -> list[ScoredCandidate]:
        ...


@runtime_checkable
class VectorRetriever(Protocol):
    """Performs dense vector similarity search with mandatory tenant isolation."""

    def retrieve_vector(
        self,
        query_vector: list[float],
        retrieval_filter: RetrievalFilter,
        limit: int = 30,
    ) -> list[ScoredCandidate]:
        ...


@runtime_checkable
class RankFuser(Protocol):
    """Fuses ranked candidate lists from multiple retrieval branches (e.g. RRF)."""

    def fuse(
        self,
        branch_candidates: list[list[ScoredCandidate]],
        k: int = 60,
    ) -> list[ScoredCandidate]:
        ...


@runtime_checkable
class CrossEncoderReranker(Protocol):
    """Reranks candidate chunks using a cross-encoder model."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_k: int = 20,
    ) -> list[ScoredCandidate]:
        ...
