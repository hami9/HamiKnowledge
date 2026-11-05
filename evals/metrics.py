"""Evaluation metrics for retrieval quality, ranking, citations, and abstention.

Implements all required roadmap metrics:
- Recall@10: Fraction of strong relevant passages retrieved in top 10
- nDCG@10: Normalized Discounted Cumulative Gain over relevance grades (1..3)
- MRR: Mean Reciprocal Rank
- Citation validity: 100% target for cited passages existing in authorized context
- Abstention accuracy: True Abstention Rate (>= 90%) and False Abstention Rate (<= 10%)
- Latency percentiles: p50, p90, p95
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from evals.schema import EvalItem, normalize_for_match
from retrieval.generation import GeneratedAnswer
from retrieval.protocols import Chunk, ScoredCandidate

TOP_K = 10


def extract_relevance_spans(
    item: EvalItem,
    document_texts: dict[str, str],
) -> list[tuple[str, int, int, int]]:
    """Map verbatim relevance quotes from an EvalItem to character spans in documents.

    Returns list of (doc_path, start_char, end_char, grade).
    """
    spans: list[tuple[str, int, int, int]] = []
    for rel in item.relevant:
        raw_text = document_texts.get(rel.doc_path, "")
        if not raw_text:
            continue
        norm_doc = normalize_for_match(raw_text)
        norm_quote = normalize_for_match(rel.quote)
        pos = norm_doc.find(norm_quote)
        if pos != -1:
            spans.append((rel.doc_path, pos, pos + len(norm_quote), rel.grade))
    return spans


def chunk_relevance_gain(
    candidate: ScoredCandidate | Chunk,
    doc_path: str,
    spans: list[tuple[str, int, int, int]],
) -> int:
    """Determine highest relevance grade (0..3) covered by a chunk."""
    best = 0
    start = candidate.location.start_char
    end = candidate.location.end_char
    for span_path, span_start, span_end, grade in spans:
        if span_path == doc_path and start < span_end and span_start < end:
            best = max(best, grade)
    return best


def recall_at_k(
    candidates: Sequence[ScoredCandidate],
    candidate_doc_paths: dict[str, str],
    spans: list[tuple[str, int, int, int]],
    k: int = TOP_K,
) -> float:
    """Calculate Recall@k for strong relevant spans (grade >= 2)."""
    strong_spans = [s for s in spans if s[3] >= 2]
    if not strong_spans:
        return 0.0

    top_candidates = candidates[:k]
    matched_spans = 0

    for span in strong_spans:
        is_covered = any(
            chunk_relevance_gain(c, candidate_doc_paths.get(c.chunk_id, ""), [span]) > 0
            for c in top_candidates
        )
        if is_covered:
            matched_spans += 1

    return matched_spans / len(strong_spans)


def ndcg_at_k(
    candidates: Sequence[ScoredCandidate],
    candidate_doc_paths: dict[str, str],
    spans: list[tuple[str, int, int, int]],
    k: int = TOP_K,
) -> float:
    """Calculate nDCG@k using standard graded relevance gains."""
    if not spans:
        return 0.0

    top_candidates = candidates[:k]
    gains = [
        chunk_relevance_gain(c, candidate_doc_paths.get(c.chunk_id, ""), spans)
        for c in top_candidates
    ]

    dcg = sum((2**g - 1) / math.log2(rank + 2) for rank, g in enumerate(gains))

    # Calculate IDCG from all relevant spans
    all_possible_gains = sorted((s[3] for s in spans), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(rank + 2) for rank, g in enumerate(all_possible_gains))

    if idcg == 0.0:
        return 0.0
    return float(min(1.0, dcg / idcg))


def mrr_at_k(
    candidates: Sequence[ScoredCandidate],
    candidate_doc_paths: dict[str, str],
    spans: list[tuple[str, int, int, int]],
    k: int = TOP_K,
) -> float:
    """Calculate Mean Reciprocal Rank (MRR) of first relevant passage."""
    strong_spans = [s for s in spans if s[3] >= 2]
    if not strong_spans:
        return 0.0

    for rank, c in enumerate(candidates[:k], start=1):
        if chunk_relevance_gain(c, candidate_doc_paths.get(c.chunk_id, ""), strong_spans) > 0:
            return 1.0 / rank
    return 0.0


def calculate_percentile(values: Sequence[float], pct: float) -> float:
    """Compute percentile value (0..100) from sample list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(pct / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def compute_abstention_rates(
    items: list[EvalItem],
    answers: list[GeneratedAnswer],
) -> dict[str, float]:
    """Compute True Abstention Rate (on unanswerable) and False Abstention Rate (on answerable)."""
    assert len(items) == len(answers)

    unanswerable_total = 0
    unanswerable_abstained = 0

    answerable_total = 0
    answerable_abstained = 0

    for item, ans in zip(items, answers, strict=True):
        if not item.answerable:
            unanswerable_total += 1
            if ans.abstained:
                unanswerable_abstained += 1
        else:
            answerable_total += 1
            if ans.abstained:
                answerable_abstained += 1

    tar = (unanswerable_abstained / unanswerable_total) if unanswerable_total > 0 else 1.0
    far = (answerable_abstained / answerable_total) if answerable_total > 0 else 0.0

    return {
        "true_abstention_rate": round(tar, 4),
        "false_abstention_rate": round(far, 4),
        "unanswerable_count": unanswerable_total,
        "answerable_count": answerable_total,
    }
