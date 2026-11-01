"""Unit tests for citation validation, abstention policy, and cost accounting."""

from __future__ import annotations

from uuid import UUID, uuid4

from retrieval.generation import (
    DeterministicGenerationAdapter,
    compute_cost_usd,
    estimate_tokens,
    validate_citations,
)
from retrieval.protocols import ScoredCandidate, SourceLocation


def _make_candidate(chunk_id: str, content: str, workspace_id: UUID) -> ScoredCandidate:
    return ScoredCandidate(
        chunk_id=chunk_id,
        document_id=uuid4(),
        workspace_id=workspace_id,
        source_version="v1",
        content=content,
        score=0.95,
        rank=1,
        branch="reranked",
        location=SourceLocation(
            start_char=10,
            end_char=10 + len(content),
            heading_path=("Overview",),
        ),
    )


def test_token_and_cost_estimation() -> None:
    text = "Kubernetes clusters use kube-apiserver for API requests."
    tokens = estimate_tokens(text)
    assert tokens > 0

    cost = compute_cost_usd(prompt_tokens=1000, completion_tokens=200)
    assert cost > 0.0
    # $0.15/1M * 1000 + $0.60/1M * 200 = 0.00015 + 0.00012 = 0.00027
    assert round(cost, 5) == 0.00027


def test_citation_validation_strips_fabrications_and_cross_tenant_chunks() -> None:
    ws_authorized = uuid4()
    ws_unauthorized = uuid4()

    valid_cand = _make_candidate("chk_valid", "Valid passage content.", ws_authorized)
    foreign_cand = _make_candidate("chk_foreign", "Foreign workspace content.", ws_unauthorized)

    raw_answer = (
        "Here is a claim supported by [chk_valid], and an invalid claim [chk_fake], "
        "and a leaked citation [chk_foreign]."
    )

    cleaned_answer, citations = validate_citations(
        raw_answer=raw_answer,
        context_candidates=[valid_cand, foreign_cand],
        workspace_id=ws_authorized,
    )

    # Valid citation kept
    assert len(citations) == 1
    assert citations[0].chunk_id == "chk_valid"
    assert citations[0].start_char == 10

    # Fabricated [chk_fake] and cross-tenant [chk_foreign] stripped
    assert "[chk_fake]" not in cleaned_answer
    assert "[chk_foreign]" not in cleaned_answer
    assert "[chk_valid]" in cleaned_answer


def test_generation_adapter_sufficient_evidence() -> None:
    ws_id = uuid4()
    cand = _make_candidate(
        "chk_sec",
        "Pod Security Standards define Privileged, Baseline, and Restricted levels.",
        ws_id,
    )

    gen = DeterministicGenerationAdapter()
    result = gen.generate(
        query="What are Pod Security Standards levels?",
        candidates=[cand],
        workspace_id=ws_id,
    )

    assert not result.abstained
    assert "[chk_sec]" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "chk_sec"
    assert result.usage.total_tokens > 0
    assert result.usage.estimated_cost_usd > 0.0
    assert result.latency_ms >= 0.0


def test_generation_adapter_insufficient_evidence_abstains() -> None:
    ws_id = uuid4()

    gen = DeterministicGenerationAdapter()
    # No candidates provided
    result_empty = gen.generate(
        query="What is quantum teleportation?",
        candidates=[],
        workspace_id=ws_id,
    )
    assert result_empty.abstained
    assert "insufficient evidence" in result_empty.answer.lower()
    assert len(result_empty.citations) == 0

    # Persian query abstains in Persian
    result_fa = gen.generate(
        query="تله پورت کوانتومی چگونه کار میکند؟",
        candidates=[],
        workspace_id=ws_id,
    )
    assert result_fa.abstained
    assert "اطلاعات کافی" in result_fa.answer
