"""Core data models and state definitions for HamiKnowledge.

Implements document status lifecycle, workspace scoping, and versioned chunking.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    """Document lifecycle states as required by HamiKnowledge roadmap."""

    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"
    DELETING = "DELETING"


class UserRole(StrEnum):
    OWNER = "owner"
    MEMBER = "member"
    VIEWER = "viewer"


class Workspace(BaseModel):
    workspace_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    workspace_id: UUID
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="text/markdown")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_version: str = Field(default="v1")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED)
    error_message: str | None = None
    uploader_id: UUID | None = None
    byte_size: int = Field(ge=0)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class DocumentVersion(BaseModel):
    version_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    workspace_id: UUID
    source_version: str
    content_hash: str
    raw_text: str
    byte_size: int
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
