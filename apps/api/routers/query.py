"""Query and question answering API router with cited answers, abstention, and cost tracking."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.deps import get_ledger, get_store
from apps.api.schemas import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
    SearchResultItem,
    UsageResponse,
)
from db.budget_ledger import BudgetExhaustedError, BudgetLedger
from db.memory_store import MemoryStore
from retrieval.embeddings import DeterministicEmbeddingAdapter
from retrieval.fusion import ReciprocalRankFuser
from retrieval.generation import DeterministicGenerationAdapter
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.protocols import RetrievalFilter
from retrieval.reranker import DeterministicRerankerAdapter
from retrieval.vector import InMemoryVectorRetriever

router = APIRouter(prefix="/v1/workspaces/{workspace_id}", tags=["query"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]
LedgerDep = Annotated[BudgetLedger, Depends(get_ledger)]


def _build_pipeline(store: MemoryStore, workspace_id: UUID) -> RetrievalPipeline:
    active_chunks = store.get_active_workspace_chunks(workspace_id)
    active_embeddings = store.get_active_workspace_embeddings(workspace_id)
    lex_retriever = InMemoryLexicalRetriever(active_chunks)
    vec_retriever = InMemoryVectorRetriever(active_chunks, active_embeddings)
    return RetrievalPipeline(
        lexical_retriever=lex_retriever,
        vector_retriever=vec_retriever,
        embedder=DeterministicEmbeddingAdapter(),
        fuser=ReciprocalRankFuser(),
        reranker=DeterministicRerankerAdapter(),
    )


@router.post("/query", response_model=QueryResponse)
def ask_question(
    workspace_id: UUID,
    payload: QueryRequest,
    store: StoreDep,
    ledger: LedgerDep,
) -> QueryResponse:
    """Answer a user question with cited passages, abstention, and cost accounting."""
    # 1. Enforce workspace authorization
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # 2. Reserve conservative budget ceiling before external execution
    estimated_ceiling_usd = 0.001
    try:
        reservation_id = ledger.reserve(workspace_id, estimated_ceiling_usd)
    except BudgetExhaustedError as exc:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(exc),
        ) from exc

    try:
        # 3. Build tenant filter
        allowed_docs = (
            frozenset(payload.allowed_document_ids)
            if payload.allowed_document_ids
            else None
        )
        retrieval_filter = RetrievalFilter(
            workspace_id=workspace_id,
            allowed_document_ids=allowed_docs,
        )

        # 4. Execute retrieval pipeline (B0, B1, B2, or B3)
        pipeline = _build_pipeline(store, workspace_id)
        retrieval_res = pipeline.execute(
            query=payload.query,
            retrieval_filter=retrieval_filter,
            variant=payload.variant,  # type: ignore[arg-type]
            limit=payload.limit,
        )

        # 5. Mandatory server-side authorization re-check during source assembly
        authorized_candidates = [
            c for c in retrieval_res.candidates
            if c.workspace_id == workspace_id
            and (allowed_docs is None or c.document_id in allowed_docs)
        ]

        # 6. Generate answer with structural citation validation and abstention
        generator = DeterministicGenerationAdapter()
        gen_result = generator.generate(
            query=payload.query,
            candidates=authorized_candidates,
            workspace_id=workspace_id,
            variant=payload.variant,
        )

        # 7. Reconcile actual consumption into ledger
        ledger.reconcile(
            workspace_id=workspace_id,
            reservation_id=reservation_id,
            actual_cost_usd=gen_result.usage.estimated_cost_usd,
            tokens_used=gen_result.usage.total_tokens,
        )

    except Exception:
        ledger.release(workspace_id, reservation_id)
        raise

    # 6. Format response
    citations = [
        CitationResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            start_char=c.start_char,
            end_char=c.end_char,
            heading_path=c.heading_path,
            snippet=c.snippet,
        )
        for c in gen_result.citations
    ]

    supporting_passages = [
        SearchResultItem(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            workspace_id=c.workspace_id,
            source_version=c.source_version,
            content=c.content,
            score=c.score,
            rank=c.rank,
            branch=c.branch,
            start_char=c.location.start_char,
            end_char=c.location.end_char,
            heading_path=c.location.heading_path,
        )
        for c in gen_result.supporting_passages
    ]

    return QueryResponse(
        workspace_id=workspace_id,
        query=payload.query,
        answer=gen_result.answer,
        abstained=gen_result.abstained,
        citations=citations,
        supporting_passages=supporting_passages,
        retrieval_variant=gen_result.retrieval_variant,
        usage=UsageResponse(
            prompt_tokens=gen_result.usage.prompt_tokens,
            completion_tokens=gen_result.usage.completion_tokens,
            total_tokens=gen_result.usage.total_tokens,
            estimated_cost_usd=gen_result.usage.estimated_cost_usd,
        ),
        latency_ms=round(retrieval_res.timing.total_ms + gen_result.latency_ms, 2),
    )
