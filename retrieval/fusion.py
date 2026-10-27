"""Reciprocal Rank Fusion (RRF) for combining multi-branch retrieval candidates.

Fuses ranked candidate lists (e.g. lexical and vector) into a single ordered candidate set.
Deduplicates by chunk_id and enforces tenant isolation.
"""

from __future__ import annotations

from retrieval.protocols import ScoredCandidate

DEFAULT_RRF_K = 60


class ReciprocalRankFuser:
    """Combines ranked candidate lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        if k <= 0:
            raise ValueError("RRF constant k must be positive")
        self.k = k

    def fuse(
        self,
        branch_candidates: list[list[ScoredCandidate]],
        k: int | None = None,
    ) -> list[ScoredCandidate]:
        """Fuse candidates from multiple branches (e.g. lexical and vector).

        Candidates with identical chunk_id across branches accumulate reciprocal ranks.
        """
        if not branch_candidates:
            return []

        rrf_k = k if k is not None else self.k
        rrf_scores: dict[str, float] = {}
        candidate_map: dict[str, ScoredCandidate] = {}
        branch_presence: dict[str, list[str]] = {}

        # Validate tenant isolation: all candidates must belong to the same workspace
        common_workspace_id = None

        for branch_list in branch_candidates:
            for rank, candidate in enumerate(branch_list, start=1):
                if common_workspace_id is None:
                    common_workspace_id = candidate.workspace_id
                elif candidate.workspace_id != common_workspace_id:
                    raise ValueError(
                        f"Cross-workspace candidate detected during fusion: "
                        f"{candidate.workspace_id} vs {common_workspace_id}"
                    )

                cid = candidate.chunk_id
                candidate_map[cid] = candidate
                branch_presence.setdefault(cid, []).append(candidate.branch)

                # Standard RRF formula: 1 / (k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # Sort candidates by combined RRF score descending, chunk_id ascending
        sorted_cids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        fused_results: list[ScoredCandidate] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            base = candidate_map[cid]
            fused_results.append(
                ScoredCandidate(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    workspace_id=base.workspace_id,
                    source_version=base.source_version,
                    content=base.content,
                    score=round(rrf_scores[cid], 6),
                    rank=rank,
                    branch="fused",
                    location=base.location,
                    metadata={
                        **base.metadata,
                        "fused_from": branch_presence[cid],
                        "rrf_score": round(rrf_scores[cid], 6),
                    },
                )
            )

        return fused_results
