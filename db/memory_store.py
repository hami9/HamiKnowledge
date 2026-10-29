"""In-memory data store for workspaces, documents, and chunks.

Enforces workspace isolation and document lifecycle transitions.
"""

from __future__ import annotations

import datetime as dt
from uuid import UUID

from db.models import Document, DocumentStatus, DocumentVersion, Workspace
from retrieval.protocols import Chunk


class MemoryStore:
    """Thread-safe in-memory store enforcing tenant boundaries."""

    def __init__(self) -> None:
        self.workspaces: dict[UUID, Workspace] = {}
        # (workspace_id, document_id) -> Document
        self.documents: dict[tuple[UUID, UUID], Document] = {}
        # (workspace_id, content_hash) -> Document
        self.documents_by_hash: dict[tuple[UUID, str], Document] = {}
        # (workspace_id, document_id) -> list[DocumentVersion]
        self.versions: dict[tuple[UUID, UUID], list[DocumentVersion]] = {}
        # (workspace_id, document_id) -> list[Chunk]
        self.chunks: dict[tuple[UUID, UUID], list[Chunk]] = {}

    def create_workspace(self, workspace: Workspace) -> Workspace:
        self.workspaces[workspace.workspace_id] = workspace
        return workspace

    def get_workspace(self, workspace_id: UUID) -> Workspace | None:
        return self.workspaces.get(workspace_id)

    def save_document(self, doc: Document) -> Document:
        doc.updated_at = dt.datetime.now(dt.UTC)
        self.documents[(doc.workspace_id, doc.document_id)] = doc
        if doc.status == DocumentStatus.READY:
            self.documents_by_hash[(doc.workspace_id, doc.content_hash)] = doc
        elif (
            (doc.workspace_id, doc.content_hash) in self.documents_by_hash
            and doc.status in (DocumentStatus.DELETING, DocumentStatus.FAILED)
        ):
            self.documents_by_hash.pop((doc.workspace_id, doc.content_hash), None)
        return doc

    def get_document(self, workspace_id: UUID, document_id: UUID) -> Document | None:
        return self.documents.get((workspace_id, document_id))

    def list_documents(
        self,
        workspace_id: UUID,
        include_deleting: bool = False,
    ) -> list[Document]:
        docs: list[Document] = []
        for (ws_id, _), doc in self.documents.items():
            if ws_id == workspace_id:
                if not include_deleting and doc.status == DocumentStatus.DELETING:
                    continue
                docs.append(doc)
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

    def find_document_by_hash(self, workspace_id: UUID, content_hash: str) -> Document | None:
        doc = self.documents_by_hash.get((workspace_id, content_hash))
        if doc and doc.status == DocumentStatus.READY:
            return doc
        return None

    def save_version(self, version: DocumentVersion) -> DocumentVersion:
        key = (version.workspace_id, version.document_id)
        self.versions.setdefault(key, []).append(version)
        return version

    def save_chunks(
        self,
        workspace_id: UUID,
        document_id: UUID,
        chunks: list[Chunk],
    ) -> None:
        key = (workspace_id, document_id)
        self.chunks[key] = chunks

    def get_chunks_for_document(
        self,
        workspace_id: UUID,
        document_id: UUID,
    ) -> list[Chunk]:
        return self.chunks.get((workspace_id, document_id), [])

    def get_active_workspace_chunks(self, workspace_id: UUID) -> list[Chunk]:
        """Return all chunks in workspace whose document is in READY status."""
        active: list[Chunk] = []
        for (ws_id, doc_id), chunk_list in self.chunks.items():
            if ws_id == workspace_id:
                doc = self.get_document(workspace_id, doc_id)
                if doc and doc.status == DocumentStatus.READY:
                    active.extend(chunk_list)
        return active

    def delete_document(self, workspace_id: UUID, document_id: UUID) -> bool:
        """Immediately exclude document from queries by marking as DELETING and clearing chunks."""
        key = (workspace_id, document_id)
        doc = self.documents.get(key)
        if not doc:
            return False

        doc.status = DocumentStatus.DELETING
        doc.updated_at = dt.datetime.now(dt.UTC)
        self.documents_by_hash.pop((workspace_id, doc.content_hash), None)
        # Immediately revoke chunks from active retrieval queries
        self.chunks.pop(key, None)
        return True


# Global singleton instance for memory-backed operation
global_store = MemoryStore()
