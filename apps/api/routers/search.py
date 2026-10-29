"""Search API router implementing lexical retrieval with tenant isolation."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.deps import get_store
from apps.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from db.memory_store import MemoryStore
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.protocols import RetrievalFilter

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/search", tags=["search"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]


@router.post("/lexical", response_model=SearchResponse)
def search_lexical(
    workspace_id: UUID,
    payload: SearchRequest,
    store: StoreDep,
) -> SearchResponse:
    # 1. Enforce workspace existence
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # 2. Get active chunks strictly scoped to this workspace and in READY status
    active_chunks = store.get_active_workspace_chunks(workspace_id)

    # 3. Build tenant-isolated filter
    allowed_docs = frozenset(payload.allowed_document_ids) if payload.allowed_document_ids else None
    retrieval_filter = RetrievalFilter(
        workspace_id=workspace_id,
        allowed_document_ids=allowed_docs,
    )

    # 4. Execute lexical retrieval
    retriever = InMemoryLexicalRetriever(active_chunks)
    candidates = retriever.retrieve_lexical(
        query=payload.query,
        retrieval_filter=retrieval_filter,
        limit=payload.limit,
    )

    # 5. Format results
    items = [
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

    return SearchResponse(
        workspace_id=workspace_id,
        query=payload.query,
        count=len(items),
        results=items,
    )
