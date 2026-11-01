"""Unified retrieval pipeline executing ablation variants B0, B1, B2, and B3.

- B0: Pure Lexical (PostgreSQL FTS reference)
- B1: Pure Vector (Dense vector cosine similarity)
- B2: Hybrid RRF (Lexical + Vector combined via Reciprocal Rank Fusion)
- B3: Hybrid RRF + Cross-Encoder Reranking

Enforces tenant isolation (workspace_id and document ACL) across all variants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from retrieval.embeddings import DeterministicEmbeddingAdapter, EmbeddingModel
from retrieval.fusion import ReciprocalRankFuser
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.protocols import (
    CrossEncoderReranker,
    LexicalRetriever,
    RankFuser,
    RetrievalFilter,
    ScoredCandidate,
    VectorRetriever,
)
from retrieval.reranker import DeterministicRerankerAdapter
from retrieval.vector import InMemoryVectorRetriever

RetrievalVariant = Literal["B0", "B1", "B2", "B3"]


@dataclass(frozen=True)
class PipelineTiming:
    """Breakdown of retrieval pipeline phase durations in milliseconds."""

    vector_embed_ms: float = 0.0
    retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class PipelineResult:
    """Output of a retrieval pipeline execution."""

    query: str
    variant: RetrievalVariant
    candidates: list[ScoredCandidate]
    timing: PipelineTiming


class RetrievalPipeline:
    """Orchestrates candidate retrieval, fusion, and reranking across B0-B3 variants."""

    def __init__(
        self,
        lexical_retriever: LexicalRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        embedder: EmbeddingModel | None = None,
        fuser: RankFuser | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.lexical_retriever = lexical_retriever or InMemoryLexicalRetriever()
        self.vector_retriever = vector_retriever or InMemoryVectorRetriever()
        self.embedder = embedder or DeterministicEmbeddingAdapter()
        self.fuser = fuser or ReciprocalRankFuser()
        self.reranker = reranker or DeterministicRerankerAdapter()

    def execute(
        self,
        query: str,
        retrieval_filter: RetrievalFilter,
        variant: RetrievalVariant = "B2",
        limit: int = 30,
        rerank_top_k: int = 20,
    ) -> PipelineResult:
        """Run the chosen retrieval variant with end-to-end tenant isolation."""
        t_start = time.perf_counter()
        t_embed_ms = 0.0
        t_retrieval_ms = 0.0
        t_fusion_ms = 0.0
        t_rerank_ms = 0.0

        if not query.strip():
            return PipelineResult(
                query=query,
                variant=variant,
                candidates=[],
                timing=PipelineTiming(total_ms=0.0),
            )

        candidates: list[ScoredCandidate] = []

        if variant == "B0":
            # Lexical baseline only
            t0 = time.perf_counter()
            candidates = self.lexical_retriever.retrieve_lexical(
                query=query,
                retrieval_filter=retrieval_filter,
                limit=limit,
            )
            t_retrieval_ms = (time.perf_counter() - t0) * 1000.0

        elif variant == "B1":
            # Vector baseline only
            t_emb0 = time.perf_counter()
            query_vector = self.embedder.embed_query(query)
            t_embed_ms = (time.perf_counter() - t_emb0) * 1000.0

            t0 = time.perf_counter()
            candidates = self.vector_retriever.retrieve_vector(
                query_vector=query_vector,
                retrieval_filter=retrieval_filter,
                limit=limit,
            )
            t_retrieval_ms = (time.perf_counter() - t0) * 1000.0

        elif variant in ("B2", "B3"):
            # Hybrid retrieval: both lexical and vector branches
            t_emb0 = time.perf_counter()
            query_vector = self.embedder.embed_query(query)
            t_embed_ms = (time.perf_counter() - t_emb0) * 1000.0

            t0 = time.perf_counter()
            lex_candidates = self.lexical_retriever.retrieve_lexical(
                query=query,
                retrieval_filter=retrieval_filter,
                limit=limit,
            )
            vec_candidates = self.vector_retriever.retrieve_vector(
                query_vector=query_vector,
                retrieval_filter=retrieval_filter,
                limit=limit,
            )
            t_retrieval_ms = (time.perf_counter() - t0) * 1000.0

            # Fuse ranks via Reciprocal Rank Fusion
            t_fuse0 = time.perf_counter()
            fused_candidates = self.fuser.fuse([lex_candidates, vec_candidates])
            t_fusion_ms = (time.perf_counter() - t_fuse0) * 1000.0

            if variant == "B2":
                candidates = fused_candidates[:limit]
            else:  # B3: RRF + Cross-Encoder Reranking
                t_rerank0 = time.perf_counter()
                candidates = self.reranker.rerank(
                    query=query,
                    candidates=fused_candidates[:limit],
                    top_k=rerank_top_k,
                )
                t_rerank_ms = (time.perf_counter() - t_rerank0) * 1000.0

        # Post-retrieval validation: Tenant isolation verification
        for c in candidates:
            if c.workspace_id != retrieval_filter.workspace_id:
                raise ValueError(
                    f"Cross-workspace candidate leak detected: "
                    f"candidate {c.workspace_id} != filter {retrieval_filter.workspace_id}"
                )
            if (
                retrieval_filter.allowed_document_ids is not None
                and c.document_id not in retrieval_filter.allowed_document_ids
            ):
                raise ValueError(
                    f"Unauthorized document candidate leak detected: "
                    f"document {c.document_id} not in allowed list"
                )

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        return PipelineResult(
            query=query,
            variant=variant,
            candidates=candidates,
            timing=PipelineTiming(
                vector_embed_ms=round(t_embed_ms, 2),
                retrieval_ms=round(t_retrieval_ms, 2),
                fusion_ms=round(t_fusion_ms, 2),
                rerank_ms=round(t_rerank_ms, 2),
                total_ms=round(t_total_ms, 2),
            ),
        )
