"""Bounded agent query API router."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from agent.bounded_agent import BoundedAgent
from apps.api.deps import get_ledger, get_store
from apps.api.schemas import (
    AgentQueryResponse,
    AgentToolCallResponse,
    QueryRequest,
    SearchResultItem,
)
from db.budget_ledger import BudgetLedger
from db.memory_store import MemoryStore

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/agent", tags=["agent"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]
LedgerDep = Annotated[BudgetLedger, Depends(get_ledger)]


@router.post("/query", response_model=AgentQueryResponse)
def run_agent_query(
    workspace_id: UUID,
    payload: QueryRequest,
    store: StoreDep,
    ledger: LedgerDep,
) -> AgentQueryResponse:
    """Execute bounded read-only agent query with max 3 tool calls and 20s deadline."""
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    agent = BoundedAgent(store=store, ledger=ledger)
    allowed_ids = (
        set(payload.allowed_document_ids)
        if payload.allowed_document_ids
        else None
    )

    result = agent.run(
        workspace_id=workspace_id,
        query=payload.query,
        allowed_document_ids=allowed_ids,
        variant=payload.variant,
    )

    tool_call_responses = [
        AgentToolCallResponse(
            tool_name=tc.tool_name,
            arguments=tc.arguments,
            duration_ms=tc.duration_ms,
            success=tc.success,
            output_summary=tc.output_summary,
        )
        for tc in result.tool_calls
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
        for c in result.retrieved_candidates
    ]

    return AgentQueryResponse(
        workspace_id=workspace_id,
        query=result.query,
        answer=result.answer,
        abstained=result.abstained,
        stop_reason=result.stop_reason.value,
        tool_calls=tool_call_responses,
        supporting_passages=supporting_passages,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        latency_ms=result.total_duration_ms,
    )
