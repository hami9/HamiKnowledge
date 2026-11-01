"""Cross-encoder reranking stage for HamiKnowledge.

Implements the B3 reranking pipeline stage with:
- Pinned candidate model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (revision 1427fd6)
- DeterministicRerankerAdapter: Fast, zero-network adapter for offline tests and CI
- SentenceTransformerReranker: Production cross-encoder using SentenceTransformers
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from retrieval.protocols import CrossEncoderReranker, ScoredCandidate

PINNED_RERANKER_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
PINNED_RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


class DeterministicRerankerAdapter(CrossEncoderReranker):
    """Deterministic, zero-network cross-encoder adapter for testing and CI.

    Scores candidates using term overlap, lexical-semantic coverage, and heading alignment.
    Enforces strict tenant isolation across all candidates.
    """

    def __init__(
        self,
        model_id: str = PINNED_RERANKER_ID,
        revision: str = PINNED_RERANKER_REVISION,
    ) -> None:
        self.model_id = model_id
        self.revision = revision

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_k: int = 20,
    ) -> list[ScoredCandidate]:
        if not candidates or not query.strip():
            return []

        # Enforce tenant isolation across candidates
        workspace_id: UUID | None = None
        for c in candidates:
            if workspace_id is None:
                workspace_id = c.workspace_id
            elif c.workspace_id != workspace_id:
                raise ValueError(
                    f"Cross-workspace candidate detected during reranking: "
                    f"{c.workspace_id} vs {workspace_id}"
                )

        q_tokens = set(t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 1)
        if not q_tokens:
            return candidates[:top_k]

        scored: list[tuple[float, ScoredCandidate]] = []
        for candidate in candidates:
            c_text = candidate.content.lower()
            c_tokens = set(t.lower() for t in _TOKEN_RE.findall(c_text))

            # Overlap score
            intersection = q_tokens & c_tokens
            overlap_ratio = len(intersection) / len(q_tokens)

            # Heading bonus
            heading_text = " ".join(candidate.location.heading_path).lower()
            heading_bonus = 0.2 if any(t in heading_text for t in q_tokens) else 0.0

            # Exact phrase bonus
            exact_bonus = 0.3 if query.lower().strip() in c_text else 0.0

            # Relevance gate: Discard candidates with zero semantic/lexical overlap
            if overlap_ratio == 0 and heading_bonus == 0 and exact_bonus == 0:
                continue

            # Combined reranking score scaled with original candidate score
            rerank_score = (
                overlap_ratio * 1.5 + heading_bonus + exact_bonus + (candidate.score * 0.1)
            )
            scored.append((rerank_score, candidate))

        # Sort descending by rerank score, ascending by chunk_id
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))

        results: list[ScoredCandidate] = []
        for rank, (score, orig) in enumerate(scored[:top_k], start=1):
            results.append(
                ScoredCandidate(
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    workspace_id=orig.workspace_id,
                    source_version=orig.source_version,
                    content=orig.content,
                    score=round(score, 6),
                    rank=rank,
                    branch="reranked",
                    location=orig.location,
                    metadata={
                        **orig.metadata,
                        "reranker_model": self.model_id,
                        "reranker_revision": self.revision,
                    },
                )
            )

        return results


class SentenceTransformerReranker(CrossEncoderReranker):
    """Production cross-encoder reranker using SentenceTransformers CrossEncoder."""

    def __init__(
        self,
        model_id: str = PINNED_RERANKER_ID,
        revision: str = PINNED_RERANKER_REVISION,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_id,
                revision=self.revision,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_k: int = 20,
    ) -> list[ScoredCandidate]:
        if not candidates or not query.strip():
            return []

        # Validate tenant isolation
        workspace_id: UUID | None = None
        for c in candidates:
            if workspace_id is None:
                workspace_id = c.workspace_id
            elif c.workspace_id != workspace_id:
                raise ValueError(
                    f"Cross-workspace candidate detected during reranking: "
                    f"{c.workspace_id} vs {workspace_id}"
                )

        model = self._get_model()
        pairs = [[query, c.content] for c in candidates]
        scores = model.predict(pairs)

        scored = list(zip(scores, candidates, strict=True))
        scored.sort(key=lambda x: (-float(x[0]), x[1].chunk_id))

        results: list[ScoredCandidate] = []
        for rank, (score, orig) in enumerate(scored[:top_k], start=1):
            results.append(
                ScoredCandidate(
                    chunk_id=orig.chunk_id,
                    document_id=orig.document_id,
                    workspace_id=orig.workspace_id,
                    source_version=orig.source_version,
                    content=orig.content,
                    score=round(float(score), 6),
                    rank=rank,
                    branch="reranked",
                    location=orig.location,
                    metadata={
                        **orig.metadata,
                        "reranker_model": self.model_id,
                        "reranker_revision": self.revision,
                    },
                )
            )

        return results
