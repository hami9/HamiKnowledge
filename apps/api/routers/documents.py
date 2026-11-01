"""Document ingestion, retrieval, and lifecycle API router."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from apps.api.deps import get_ingestion_service, get_store
from apps.api.schemas import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentExcerptResponse,
    DocumentResponse,
)
from db.memory_store import MemoryStore
from db.models import Document
from ingestion.service import FileValidationError, IngestionService, ScannedDocumentError

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/documents", tags=["documents"])
StoreDep = Annotated[MemoryStore, Depends(get_store)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def _to_doc_response(doc: Document, is_duplicate: bool = False) -> DocumentResponse:
    return DocumentResponse(
        document_id=doc.document_id,
        workspace_id=doc.workspace_id,
        filename=doc.filename,
        content_type=doc.content_type,
        content_hash=doc.content_hash,
        source_version=doc.source_version,
        status=doc.status,
        error_message=doc.error_message,
        byte_size=doc.byte_size,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        is_duplicate=is_duplicate,
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: UUID,
    file: Annotated[UploadFile, File()],
    store: StoreDep,
    service: IngestionDep,
    source_version: Annotated[str, Form()] = "v1",
    language: Annotated[str, Form()] = "en",
) -> DocumentResponse:
    # 1. Verify workspace existence
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # 2. Read content bytes
    content_bytes = await file.read()
    filename = file.filename or "uploaded.md"
    content_type = file.content_type or "text/markdown"

    try:
        doc, is_new = service.ingest_document(
            workspace_id=workspace_id,
            filename=filename,
            content_bytes=content_bytes,
            content_type=content_type,
            source_version=source_version,
            language=language,
        )
        return _to_doc_response(doc, is_duplicate=not is_new)

    except FileValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ScannedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    workspace_id: UUID,
    store: StoreDep,
) -> list[DocumentResponse]:
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    return [_to_doc_response(doc) for doc in store.list_documents(workspace_id)]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    workspace_id: UUID,
    document_id: UUID,
    store: StoreDep,
) -> DocumentDetailResponse:
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    doc = store.get_document(workspace_id, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found in workspace",
        )

    chunks = store.get_chunks_for_document(workspace_id, document_id)
    chunk_responses = [
        ChunkResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            workspace_id=c.workspace_id,
            source_version=c.source_version,
            content=c.content,
            start_char=c.location.start_char,
            end_char=c.location.end_char,
            heading_path=c.location.heading_path,
            language=c.language,
        )
        for c in chunks
    ]

    return DocumentDetailResponse(
        document=_to_doc_response(doc),
        chunks=chunk_responses,
    )


@router.get("/{document_id}/excerpt", response_model=DocumentExcerptResponse)
def get_document_excerpt(
    workspace_id: UUID,
    document_id: UUID,
    start_char: int,
    end_char: int,
    store: StoreDep,
) -> DocumentExcerptResponse:
    """Retrieve raw source excerpt at exact character offsets with strict authorization."""
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    excerpt = store.get_document_excerpt(
        workspace_id=workspace_id,
        document_id=document_id,
        start_char=start_char,
        end_char=end_char,
    )
    if excerpt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document excerpt not found or access revoked",
        )
    return DocumentExcerptResponse(
        document_id=document_id,
        workspace_id=workspace_id,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    workspace_id: UUID,
    document_id: UUID,
    store: StoreDep,
) -> None:
    if not store.get_workspace(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    success = store.delete_document(workspace_id, document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found in workspace",
        )
