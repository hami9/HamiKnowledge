"""Budget inspection and management router."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.deps import get_ledger, get_store
from apps.api.schemas import AddFundsRequest, BudgetResponse, SetBudgetRequest
from db.budget_ledger import BudgetLedger
from db.memory_store import MemoryStore

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/budget", tags=["budget"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]
LedgerDep = Annotated[BudgetLedger, Depends(get_ledger)]


@router.get("", response_model=BudgetResponse)
def get_workspace_budget(
    workspace_id: UUID,
    store: StoreDep,
    ledger: LedgerDep,
) -> BudgetResponse:
    """Retrieve current workspace financial balance and usage metrics."""
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    budget = ledger.get_or_create_budget(workspace_id)
    return BudgetResponse(
        workspace_id=workspace_id,
        total_budget_usd=budget.total_budget_usd,
        spent_usd=budget.spent_usd,
        reserved_usd=budget.reserved_usd,
        available_usd=budget.available_usd,
        request_count=budget.request_count,
        total_tokens=budget.total_tokens,
    )


@router.post("/add-funds", response_model=BudgetResponse)
def add_funds(
    workspace_id: UUID,
    payload: AddFundsRequest,
    store: StoreDep,
    ledger: LedgerDep,
) -> BudgetResponse:
    """Add funds to workspace budget."""
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    budget = ledger.add_funds(workspace_id, payload.amount_usd)
    return BudgetResponse(
        workspace_id=workspace_id,
        total_budget_usd=budget.total_budget_usd,
        spent_usd=budget.spent_usd,
        reserved_usd=budget.reserved_usd,
        available_usd=budget.available_usd,
        request_count=budget.request_count,
        total_tokens=budget.total_tokens,
    )


@router.put("", response_model=BudgetResponse)
def set_workspace_budget(
    workspace_id: UUID,
    payload: SetBudgetRequest,
    store: StoreDep,
    ledger: LedgerDep,
) -> BudgetResponse:
    """Set total budget ceiling for workspace."""
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    budget = ledger.set_budget(workspace_id, payload.total_budget_usd)
    return BudgetResponse(
        workspace_id=workspace_id,
        total_budget_usd=budget.total_budget_usd,
        spent_usd=budget.spent_usd,
        reserved_usd=budget.reserved_usd,
        available_usd=budget.available_usd,
        request_count=budget.request_count,
        total_tokens=budget.total_tokens,
    )
