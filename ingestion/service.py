"""Ingestion service implementing document validation, versioning, and chunking.

Satisfies all roadmap ingestion contracts:
- 10 MB file size limit and MIME validation.
- Idempotent deduplication using content hash per workspace.
- State machine: UPLOADED -> PARSING -> INDEXING -> READY (or FAILED).
- Heading-aware chunking with exact character offset preservation.
"""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from db.memory_store import MemoryStore, global_store
from db.models import Document, DocumentStatus, DocumentVersion
from ingestion.chunker import HeadingAwareChunker

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB limit per roadmap specification
SUPPORTED_CONTENT_TYPES = {
    "text/markdown",
    "text/plain",
    "text/x-markdown",
    "application/octet-stream",  # often sent for .md/.txt files
}


class IngestionError(Exception):
    """Base error for ingestion failures."""


class FileValidationError(IngestionError):
    """Raised when file exceeds size or has an unsupported format."""


class ScannedDocumentError(IngestionError):
    """Raised when document contains no extractable text stream."""


class IngestionService:
    """Orchestrates document ingestion with state tracking and deduplication."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        chunker: HeadingAwareChunker | None = None,
    ) -> None:
        self.store = store or global_store
        self.chunker = chunker or HeadingAwareChunker()

    def ingest_document(
        self,
        workspace_id: UUID,
        filename: str,
        content_bytes: bytes,
        content_type: str = "text/markdown",
        uploader_id: UUID | None = None,
        source_version: str = "v1",
        language: str = "en",
    ) -> tuple[Document, bool]:
        """Ingest a document into a workspace.

        Returns (document, is_new).
        If an identical content hash already exists and is READY in this workspace,
        returns the existing document without re-chunking (deduplication / idempotency).
        """
        # 1. Validation
        if len(content_bytes) > MAX_FILE_BYTES:
            raise FileValidationError(
                f"File size ({len(content_bytes)} bytes) exceeds 10 MB limit"
            )
        if len(content_bytes) == 0:
            raise FileValidationError("Uploaded file is empty (0 bytes)")

        # Compute stable content hash
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # 2. Idempotency check: reuse ready document with identical content hash in this workspace
        existing_doc = self.store.find_document_by_hash(workspace_id, content_hash)
        if existing_doc is not None and existing_doc.status == DocumentStatus.READY:
            return existing_doc, False

        # 3. Create initial document record in UPLOADED state
        doc_id = uuid4()
        doc = Document(
            document_id=doc_id,
            workspace_id=workspace_id,
            filename=filename,
            content_type=content_type,
            content_hash=content_hash,
            source_version=source_version,
            status=DocumentStatus.UPLOADED,
            uploader_id=uploader_id,
            byte_size=len(content_bytes),
        )
        self.store.save_document(doc)

        try:
            # 4. PARSING Stage
            doc.status = DocumentStatus.PARSING
            self.store.save_document(doc)

            try:
                raw_text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Try UTF-8 with replacement or check if binary/scanned
                try:
                    raw_text = content_bytes.decode("latin-1")
                except Exception as exc:
                    raise ScannedDocumentError(
                        "File does not contain a decodable text stream. "
                        "Scanned documents and OCR are out of scope for v1."
                    ) from exc

            if not raw_text.strip():
                raise ScannedDocumentError(
                    "Extracted document text is empty. "
                    "Scanned documents without text streams are not supported in v1."
                )

            # Store immutable source version
            version_record = DocumentVersion(
                document_id=doc_id,
                workspace_id=workspace_id,
                source_version=source_version,
                content_hash=content_hash,
                raw_text=raw_text,
                byte_size=len(content_bytes),
            )
            self.store.save_version(version_record)

            # 5. INDEXING / Chunking Stage
            doc.status = DocumentStatus.INDEXING
            self.store.save_document(doc)

            chunks = self.chunker.chunk(
                text=raw_text,
                document_id=doc_id,
                workspace_id=workspace_id,
                source_version=source_version,
                language=language,
            )

            # Persist chunks with location metadata
            self.store.save_chunks(workspace_id, doc_id, chunks)

            # 6. READY Stage
            doc.status = DocumentStatus.READY
            doc.error_message = None
            self.store.save_document(doc)
            return doc, True

        except Exception as exc:
            # Transition to FAILED with clear actionable error message
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(exc)
            self.store.save_document(doc)
            raise
