"""API schemas for request validation and response serialization."""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from db.models import DocumentStatus


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceResponse(BaseModel):
    workspace_id: UUID
    name: str
    slug: str
    created_at: dt.datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: UUID
    workspace_id: UUID
    filename: str
    content_type: str
    content_hash: str
    source_version: str
    status: DocumentStatus
    error_message: str | None = None
    byte_size: int
    created_at: dt.datetime
    updated_at: dt.datetime
    is_duplicate: bool = False


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: UUID
    workspace_id: UUID
    source_version: str
    content: str
    start_char: int
    end_char: int
    heading_path: tuple[str, ...]
    language: str


class DocumentDetailResponse(BaseModel):
    document: DocumentResponse
    chunks: list[ChunkResponse] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)
    allowed_document_ids: list[UUID] | None = None


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: UUID
    workspace_id: UUID
    source_version: str
    content: str
    score: float
    rank: int
    branch: str
    start_char: int
    end_char: int
    heading_path: tuple[str, ...]


class SearchResponse(BaseModel):
    workspace_id: UUID
    query: str
    count: int
    results: list[SearchResultItem]


class DocumentExcerptResponse(BaseModel):
    document_id: UUID
    workspace_id: UUID
    start_char: int
    end_char: int
    excerpt: str


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: UUID
    start_char: int
    end_char: int
    heading_path: tuple[str, ...]
    snippet: str


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    variant: str = Field(default="B2", pattern=r"^B[0-3]$")
    limit: int = Field(default=10, ge=1, le=50)
    allowed_document_ids: list[UUID] | None = None


class QueryResponse(BaseModel):
    workspace_id: UUID
    query: str
    answer: str
    abstained: bool
    citations: list[CitationResponse]
    supporting_passages: list[SearchResultItem]
    retrieval_variant: str
    usage: UsageResponse
    latency_ms: float


class BudgetResponse(BaseModel):
    workspace_id: UUID
    total_budget_usd: float
    spent_usd: float
    reserved_usd: float
    available_usd: float
    request_count: int
    total_tokens: int


class AddFundsRequest(BaseModel):
    amount_usd: float = Field(gt=0.0, le=1000.0)


