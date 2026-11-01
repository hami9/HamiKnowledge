"""Vector retrieval baseline (B1) using pgvector and in-memory cosine similarity.

Enforces tenant isolation (workspace_id and document ACL) inside the query.
"""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

from retrieval.protocols import Chunk, RetrievalFilter, ScoredCandidate, VectorRetriever


def build_pgvector_sql(
    query_vector: list[float],
    retrieval_filter: RetrievalFilter,
    limit: int = 30,
) -> tuple[str, dict[str, Any]]:
    """Generate PostgreSQL pgvector cosine similarity search query with strict tenant isolation.

    Uses cosine distance `<=>` on the `embedding` column.
    Never omits the workspace_id predicate.
    """
    conditions = ["c.workspace_id = :workspace_id"]
    params: dict[str, Any] = {
        "workspace_id": str(retrieval_filter.workspace_id),
        "limit": limit,
        "query_vector": str(query_vector),
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
        1 - (c.embedding <=> :query_vector) AS vector_score
    FROM chunks c
    WHERE {where_clause}
    ORDER BY c.embedding <=> :query_vector ASC, c.chunk_id ASC
    LIMIT :limit;
    """.strip()

    return sql, params


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    sim = dot / (norm1 * norm2)
    # Clamp to [-1.0, 1.0] to guard against floating point inaccuracies
    return max(-1.0, min(1.0, sim))


class InMemoryVectorRetriever(VectorRetriever):
    """In-memory vector retriever for unit tests, offline evaluation, and local search.

    Implements cosine similarity ranking with strict tenant isolation.
    """

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self.chunks_by_workspace: dict[UUID, list[Chunk]] = {}
        self.embeddings: dict[str, list[float]] = {}

        if chunks:
            for c in chunks:
                self.chunks_by_workspace.setdefault(c.workspace_id, []).append(c)

        if embeddings:
            self.embeddings.update(embeddings)

    def index_chunk(self, chunk: Chunk, embedding: list[float]) -> None:
        """Register a chunk and its dense embedding."""
        self.chunks_by_workspace.setdefault(chunk.workspace_id, []).append(chunk)
        self.embeddings[chunk.chunk_id] = embedding

    def retrieve_vector(
        self,
        query_vector: list[float],
        retrieval_filter: RetrievalFilter,
        limit: int = 30,
    ) -> list[ScoredCandidate]:
        """Retrieve and rank candidates by vector similarity with strict tenant isolation."""
        ws_chunks = self.chunks_by_workspace.get(retrieval_filter.workspace_id, [])
        if not ws_chunks or not query_vector:
            return []

        # Filter by document authorization
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

        scored: list[tuple[float, Chunk]] = []
        for chunk in authorized_chunks:
            chunk_vec = self.embeddings.get(chunk.chunk_id)
            if chunk_vec is None:
                continue

            sim = cosine_similarity(query_vector, chunk_vec)
            scored.append((sim, chunk))

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
                    branch="vector",
                    location=chunk.location,
                    metadata={"search_method": "pgvector_reference", "similarity": "cosine"},
                )
            )

        return results
