"""Unit tests for dense vector retrieval (B1) and pgvector query generation."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from retrieval.protocols import Chunk, RetrievalFilter, SourceLocation
from retrieval.vector import (
    InMemoryVectorRetriever,
    build_pgvector_sql,
    cosine_similarity,
)


def _make_chunk(
    content: str,
    workspace_id: UUID | None = None,
    document_id: UUID | None = None,
    chunk_id: str | None = None,
) -> Chunk:
    ws_id = workspace_id or uuid4()
    doc_id = document_id or uuid4()
    cid = chunk_id or f"chk_{uuid4().hex[:8]}"
    return Chunk(
        chunk_id=cid,
        document_id=doc_id,
        workspace_id=ws_id,
        source_version="v1",
        content=content,
        search_text=content.lower(),
        location=SourceLocation(start_char=0, end_char=len(content)),
        language="en",
    )


def test_cosine_similarity_math() -> None:
    # Identical vectors -> 1.0
    v1 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v1), 1e-6) == 1.0

    # Orthogonal vectors -> 0.0
    v2 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 1e-6) == 0.0

    # Opposite vectors -> -1.0
    v3 = [-1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3), 1e-6) == -1.0

    # Zero vector handled safely
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_build_pgvector_sql_enforces_tenant_isolation() -> None:
    ws_id = uuid4()
    doc_id1 = uuid4()
    doc_id2 = uuid4()
    query_vec = [0.1, 0.2, 0.3]

    retrieval_filter = RetrievalFilter(
        workspace_id=ws_id,
        allowed_document_ids=frozenset([doc_id1, doc_id2]),
    )

    sql, params = build_pgvector_sql(
        query_vector=query_vec,
        retrieval_filter=retrieval_filter,
        limit=20,
    )

    # Invariant: workspace_id predicate MUST be present
    assert "c.workspace_id = :workspace_id" in sql
    assert params["workspace_id"] == str(ws_id)
    assert "c.embedding <=>" in sql
    assert "LIMIT :limit" in sql
    assert params["limit"] == 20
    assert "c.document_id = ANY(:allowed_doc_ids)" in sql


def test_in_memory_vector_retriever_ranking_and_isolation() -> None:
    ws1 = uuid4()
    ws2 = uuid4()

    c1 = _make_chunk("Kubernetes Pod Security Standards", workspace_id=ws1, chunk_id="c1")
    c2 = _make_chunk("Kubernetes Network Policy", workspace_id=ws1, chunk_id="c2")
    c3 = _make_chunk("Etcd Cluster Backup in Workspace 2", workspace_id=ws2, chunk_id="c3")

    # Define simple orthogonal vectors
    embeddings = {
        "c1": [1.0, 0.0, 0.0],
        "c2": [0.7, 0.7, 0.0],
        "c3": [0.0, 0.0, 1.0],
    }

    retriever = InMemoryVectorRetriever(chunks=[c1, c2, c3], embeddings=embeddings)

    # Search in workspace 1 with query vector aligning closely with c1
    q_vec = [0.99, 0.05, 0.0]
    results_ws1 = retriever.retrieve_vector(
        query_vector=q_vec,
        retrieval_filter=RetrievalFilter(workspace_id=ws1),
        limit=10,
    )

    assert len(results_ws1) == 2
    assert results_ws1[0].chunk_id == "c1"
    assert results_ws1[0].score > results_ws1[1].score
    assert results_ws1[0].branch == "vector"

    # ZERO LEAKAGE: c3 from workspace 2 must not be returned
    assert all(r.workspace_id == ws1 for r in results_ws1)
    assert not any(r.chunk_id == "c3" for r in results_ws1)


def test_vector_retriever_document_filters() -> None:
    ws = uuid4()
    doc_a = uuid4()
    doc_b = uuid4()

    c_a = _make_chunk("Doc A content", workspace_id=ws, document_id=doc_a, chunk_id="ca")
    c_b = _make_chunk("Doc B content", workspace_id=ws, document_id=doc_b, chunk_id="cb")

    embeddings = {
        "ca": [1.0, 0.0],
        "cb": [0.9, 0.1],
    }

    retriever = InMemoryVectorRetriever(chunks=[c_a, c_b], embeddings=embeddings)

    # Restrict to doc_b only
    filter_b = RetrievalFilter(workspace_id=ws, allowed_document_ids=frozenset([doc_b]))
    results = retriever.retrieve_vector(query_vector=[1.0, 0.0], retrieval_filter=filter_b)

    assert len(results) == 1
    assert results[0].chunk_id == "cb"
    assert results[0].document_id == doc_b
