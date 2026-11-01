"""Unit tests for cross-encoder reranking stage and tenant isolation."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from retrieval.protocols import ScoredCandidate, SourceLocation
from retrieval.reranker import DeterministicRerankerAdapter


def _make_candidate(
    chunk_id: str,
    content: str,
    score: float,
    rank: int,
    workspace_id: UUID | None = None,
    heading_path: tuple[str, ...] = (),
) -> ScoredCandidate:
    ws_id = workspace_id or uuid4()
    return ScoredCandidate(
        chunk_id=chunk_id,
        document_id=uuid4(),
        workspace_id=ws_id,
        source_version="v1",
        content=content,
        score=score,
        rank=rank,
        branch="fused",
        location=SourceLocation(start_char=0, end_char=len(content), heading_path=heading_path),
    )


def test_deterministic_reranker_reordering() -> None:
    ws_id = uuid4()
    # Candidate 1 has vague overlap
    c1 = _make_candidate(
        chunk_id="c1",
        content="General container configuration and Pod namespaces.",
        score=0.8,
        rank=1,
        workspace_id=ws_id,
    )
    # Candidate 2 has exact phrase and heading match
    c2 = _make_candidate(
        chunk_id="c2",
        content="Pod Security Admission restricts pod creation using Pod Security Standards.",
        score=0.6,
        rank=2,
        workspace_id=ws_id,
        heading_path=("Security", "Pod Security Admission"),
    )

    reranker = DeterministicRerankerAdapter()
    reranked = reranker.rerank(
        query="Pod Security Standards",
        candidates=[c1, c2],
        top_k=5,
    )

    assert len(reranked) == 2
    # c2 must be boosted to rank 1 due to high term overlap and heading alignment
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[0].branch == "reranked"
    assert reranked[1].chunk_id == "c1"
    assert reranked[1].rank == 2


def test_reranker_rejects_cross_tenant_candidates() -> None:
    ws1 = uuid4()
    ws2 = uuid4()

    c1 = _make_candidate("c1", "Team 1 document", 0.9, 1, workspace_id=ws1)
    c2 = _make_candidate("c2", "Team 2 document", 0.8, 2, workspace_id=ws2)

    reranker = DeterministicRerankerAdapter()
    with pytest.raises(ValueError, match="Cross-workspace candidate detected"):
        reranker.rerank(query="document", candidates=[c1, c2])
