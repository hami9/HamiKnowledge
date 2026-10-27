"""Lexical retrieval baseline (B0) using PostgreSQL Full-Text Search.

Explicitly labeled as PostgreSQL native ts_rank_cd (NOT BM25).
Enforces tenant isolation (workspace_id and document ACL) inside the query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any
from uuid import UUID

from ingestion.normalizer import PersianNormalizer
from retrieval.protocols import Chunk, RetrievalFilter, ScoredCandidate

_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def build_postgres_fts_sql(
    query_text: str,
    retrieval_filter: RetrievalFilter,
    limit: int = 30,
) -> tuple[str, dict[str, Any]]:
    """Generate PostgreSQL full-text search query with strict tenant isolation.

    Uses `ts_rank_cd` on `search_vector`. Never omits the workspace_id predicate.
    """
    conditions = ["c.workspace_id = :workspace_id"]
    params: dict[str, Any] = {
        "workspace_id": str(retrieval_filter.workspace_id),
        "limit": limit,
        "query_text": query_text,
    }

    if retrieval_filter.allowed_document_ids is not None:
        conditions.append("c.document_id = ANY(:allowed_doc_ids)")
        params["allowed_doc_ids"] = [str(d) for d in retrieval_filter.allowed_document_ids]

    if retrieval_filter.excluded_document_ids:
        conditions.append("NOT (c.document_id = ANY(:excluded_doc_ids))")
        params["excluded_doc_ids"] = [str(d) for d in retrieval_filter.excluded_document_ids]

    where_clause = " AND ".join(conditions)

    sql = f"""
    SELECT
        c.chunk_id,
        c.document_id,
        c.workspace_id,
        c.source_version,
        c.content,
        c.start_char,
        c.end_char,
        c.heading_path,
        ts_rank_cd(c.search_vector, websearch_to_tsquery('simple', :query_text)) AS fts_score
    FROM chunks c
    WHERE {where_clause}
      AND c.search_vector @@ websearch_to_tsquery('simple', :query_text)
    ORDER BY fts_score DESC, c.chunk_id ASC
    LIMIT :limit;
    """.strip()

    return sql, params


class InMemoryLexicalRetriever:
    """In-memory lexical retriever for unit tests and offline evaluation.

    Implements term-frequency ranking with exact workspace authorization filtering.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks_by_workspace: dict[UUID, list[Chunk]] = {}
        self.normalizer = PersianNormalizer()
        if chunks:
            for c in chunks:
                self.index_chunk(c)

    def index_chunk(self, chunk: Chunk) -> None:
        self.chunks_by_workspace.setdefault(chunk.workspace_id, []).append(chunk)

    def retrieve_lexical(
        self,
        query: str,
        retrieval_filter: RetrievalFilter,
        limit: int = 30,
    ) -> list[ScoredCandidate]:
        """Retrieve and rank candidates with strict tenant isolation."""
        ws_chunks = self.chunks_by_workspace.get(retrieval_filter.workspace_id, [])
        if not ws_chunks or not query.strip():
            return []

        # Filter by document permissions
        authorized_chunks: list[Chunk] = []
        for c in ws_chunks:
            if (
                retrieval_filter.allowed_document_ids is not None
                and c.document_id not in retrieval_filter.allowed_document_ids
            ):
                continue
            if (
                retrieval_filter.excluded_document_ids
                and c.document_id in retrieval_filter.excluded_document_ids
            ):
                continue
            authorized_chunks.append(c)

        if not authorized_chunks:
            return []

        # Tokenize normalized query
        norm_query = self.normalizer.normalize_for_search(query).lower()
        query_tokens = [t for t in _TOKEN_RE.findall(norm_query) if len(t) > 1]
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scored: list[tuple[float, Chunk]] = []

        for chunk in authorized_chunks:
            search_doc = chunk.search_text.lower()
            doc_tokens = _TOKEN_RE.findall(search_doc)
            if not doc_tokens:
                continue
            doc_counts = Counter(doc_tokens)
            doc_len = len(doc_tokens)

            # Score matching terms (simple term-frequency ranking)
            score = 0.0
            matched_terms = 0
            for term, q_count in query_counts.items():
                if term in doc_counts:
                    tf = doc_counts[term] / doc_len
                    score += tf * q_count
                    matched_terms += 1

            if matched_terms > 0:
                # Coordinate matching bonus
                coordination = matched_terms / len(query_counts)
                final_score = score * math.sqrt(coordination)
                scored.append((final_score, chunk))

        # Sort descending by score, ascending by chunk_id for tie-breaking
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))

        results: list[ScoredCandidate] = []
        for rank, (score, chunk) in enumerate(scored[:limit], start=1):
            results.append(
                ScoredCandidate(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    workspace_id=chunk.workspace_id,
                    source_version=chunk.source_version,
                    content=chunk.content,
                    score=round(score, 6),
                    rank=rank,
                    branch="lexical",
                    location=chunk.location,
                    metadata={"search_method": "postgres_fts_reference"},
                )
            )

        return results
