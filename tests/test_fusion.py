"""Tests for Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import uuid

import pytest

from retrieval.fusion import ReciprocalRankFuser
from retrieval.protocols import ScoredCandidate, SourceLocation


def _make_candidate(
    cid: str,
    ws_id: uuid.UUID,
    rank: int,
    branch: str,
    score: float = 1.0,
) -> ScoredCandidate:
    return ScoredCandidate(
        chunk_id=cid,
        document_id=uuid.uuid4(),
        workspace_id=ws_id,
        source_version="v1",
        content=f"Content for {cid}",
        score=score,
        rank=rank,
        branch=branch,
        location=SourceLocation(start_char=0, end_char=10),
    )


def test_rrf_accumulates_scores_across_branches() -> None:
    ws_id = uuid.uuid4()
    fuser = ReciprocalRankFuser(k=60)

    # Branch 1 (Lexical): doc A at rank 1, doc B at rank 2
    lexical = [
        _make_candidate("docA", ws_id, rank=1, branch="lexical"),
        _make_candidate("docB", ws_id, rank=2, branch="lexical"),
    ]

    # Branch 2 (Vector): doc B at rank 1, doc C at rank 2
    vector = [
        _make_candidate("docB", ws_id, rank=1, branch="vector"),
        _make_candidate("docC", ws_id, rank=2, branch="vector"),
    ]

    fused = fuser.fuse([lexical, vector], k=60)

    # docB was rank 2 in lexical (1/62) and rank 1 in vector (1/61)
    # docA was rank 1 in lexical (1/61)
    # docC was rank 2 in vector (1/62)
    # Therefore docB must win the top fused rank!
    assert len(fused) == 3
    assert fused[0].chunk_id == "docB"
    assert fused[0].rank == 1
    assert fused[0].branch == "fused"
    assert "lexical" in fused[0].metadata["fused_from"]
    assert "vector" in fused[0].metadata["fused_from"]

    assert fused[1].chunk_id == "docA"
    assert fused[2].chunk_id == "docC"


def test_rrf_fails_closed_on_cross_workspace_candidates() -> None:
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    fuser = ReciprocalRankFuser()

    branch_1 = [_make_candidate("doc1", ws_a, rank=1, branch="lexical")]
    branch_2 = [_make_candidate("doc2", ws_b, rank=1, branch="vector")]

    with pytest.raises(ValueError, match="Cross-workspace candidate detected"):
        fuser.fuse([branch_1, branch_2])
