"""Search API router implementing lexical, vector, hybrid (RRF), and reranked retrieval.

Enforces strict tenant isolation across all retrieval branches.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.deps import get_store
from apps.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from db.memory_store import MemoryStore
from retrieval.embeddings import DeterministicEmbeddingAdapter
from retrieval.fusion import ReciprocalRankFuser
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.pipeline import RetrievalPipeline, RetrievalVariant
from retrieval.protocols import RetrievalFilter, ScoredCandidate
from retrieval.reranker import DeterministicRerankerAdapter
from retrieval.vector import InMemoryVectorRetriever

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/search", tags=["search"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]


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


def _format_candidates(candidates: list[ScoredCandidate]) -> list[SearchResultItem]:
    return [
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
        for c in candidates
    ]


def _execute_search(
    workspace_id: UUID,
    payload: SearchRequest,
    store: MemoryStore,
    variant: RetrievalVariant,
) -> SearchResponse:
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    allowed_docs = frozenset(payload.allowed_document_ids) if payload.allowed_document_ids else None
    retrieval_filter = RetrievalFilter(
        workspace_id=workspace_id,
        allowed_document_ids=allowed_docs,
    )

    pipeline = _build_pipeline(store, workspace_id)
    result = pipeline.execute(
        query=payload.query,
        retrieval_filter=retrieval_filter,
        variant=variant,
        limit=payload.limit,
    )

    items = _format_candidates(result.candidates)
    return SearchResponse(
        workspace_id=workspace_id,
        query=payload.query,
        count=len(items),
        results=items,
    )


@router.post("/lexical", response_model=SearchResponse)
def search_lexical(
    workspace_id: UUID,
    payload: SearchRequest,
    store: StoreDep,
) -> SearchResponse:
    """Execute B0 lexical baseline retrieval."""
    return _execute_search(workspace_id, payload, store, variant="B0")


@router.post("/vector", response_model=SearchResponse)
def search_vector(
    workspace_id: UUID,
    payload: SearchRequest,
    store: StoreDep,
) -> SearchResponse:
    """Execute B1 dense vector similarity baseline retrieval."""
    return _execute_search(workspace_id, payload, store, variant="B1")


@router.post("/hybrid", response_model=SearchResponse)
def search_hybrid(
    workspace_id: UUID,
    payload: SearchRequest,
    store: StoreDep,
) -> SearchResponse:
    """Execute B2 hybrid retrieval combining lexical and vector branches via RRF."""
    return _execute_search(workspace_id, payload, store, variant="B2")


@router.post("/reranked", response_model=SearchResponse)
def search_reranked(
    workspace_id: UUID,
    payload: SearchRequest,
    store: StoreDep,
) -> SearchResponse:
    """Execute B3 hybrid retrieval with Cross-Encoder reranking."""
    return _execute_search(workspace_id, payload, store, variant="B3")
