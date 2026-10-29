"""Workspace management API router."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.deps import get_store
from apps.api.schemas import WorkspaceCreate, WorkspaceResponse
from db.memory_store import MemoryStore
from db.models import Workspace

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    store: StoreDep,
) -> WorkspaceResponse:
    # Check slug uniqueness
    for ws in store.workspaces.values():
        if ws.slug == payload.slug:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workspace with slug '{payload.slug}' already exists",
            )

    ws = Workspace(
        name=payload.name,
        slug=payload.slug,
        metadata=payload.metadata,
    )
    saved = store.create_workspace(ws)
    return WorkspaceResponse(
        workspace_id=saved.workspace_id,
        name=saved.name,
        slug=saved.slug,
        created_at=saved.created_at,
        metadata=saved.metadata,
    )


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
    workspace_id: UUID,
    store: StoreDep,
) -> WorkspaceResponse:
    ws = store.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    return WorkspaceResponse(
        workspace_id=ws.workspace_id,
        name=ws.name,
        slug=ws.slug,
        created_at=ws.created_at,
        metadata=ws.metadata,
    )


@router.get("", response_model=list[WorkspaceResponse])
def list_workspaces(
    store: StoreDep,
) -> list[WorkspaceResponse]:
    return [
        WorkspaceResponse(
            workspace_id=ws.workspace_id,
            name=ws.name,
            slug=ws.slug,
            created_at=ws.created_at,
            metadata=ws.metadata,
        )
        for ws in sorted(store.workspaces.values(), key=lambda w: w.created_at)
    ]
