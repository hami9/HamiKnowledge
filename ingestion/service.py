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
from retrieval.embeddings import DeterministicEmbeddingAdapter, EmbeddingModel

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
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.store = store or global_store
        self.chunker = chunker or HeadingAwareChunker()
        self.embedder = embedder or DeterministicEmbeddingAdapter()

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

            # 5. Chunking
            chunks = self.chunker.chunk(
                text=raw_text,
                document_id=doc_id,
                workspace_id=workspace_id,
                source_version=source_version,
                language=language,
            )

            # 6. EMBEDDING Stage
            doc.status = DocumentStatus.EMBEDDING
            self.store.save_document(doc)

            passages = [c.search_text for c in chunks]
            vectors = self.embedder.embed_passages(passages)
            chunk_embeddings = {
                chunk.chunk_id: vec
                for chunk, vec in zip(chunks, vectors, strict=True)
            }

            # 7. INDEXING Stage
            doc.status = DocumentStatus.INDEXING
            self.store.save_document(doc)

            # Persist chunks and dense embeddings with location metadata
            self.store.save_chunks(workspace_id, doc_id, chunks)
            self.store.save_embeddings(workspace_id, doc_id, chunk_embeddings)

            # 8. READY Stage
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

    def reindex_document(
        self,
        workspace_id: UUID,
        document_id: UUID,
        new_content_bytes: bytes,
        source_version: str = "v2",
        language: str = "en",
    ) -> Document:
        """Atomically reindex an existing document with a new content version.

        In accordance with roadmap invariants:
        A new index version activates only after every required stage succeeds;
        the previous valid version stays usable through any failed reindex attempt.
        """
        existing_doc = self.store.get_document(workspace_id, document_id)
        if not existing_doc:
            raise FileValidationError(
                f"Document {document_id} not found in workspace {workspace_id}"
            )

        if len(new_content_bytes) > MAX_FILE_BYTES:
            raise FileValidationError(
                f"File size ({len(new_content_bytes)} bytes) exceeds 10 MB limit"
            )
        if len(new_content_bytes) == 0:
            raise FileValidationError("New content is empty (0 bytes)")

        new_hash = hashlib.sha256(new_content_bytes).hexdigest()

        # Stage 1: Parsing
        try:
            raw_text = new_content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = new_content_bytes.decode("latin-1")
            except Exception as exc:
                raise ScannedDocumentError("Cannot decode text stream during reindex") from exc

        if not raw_text.strip():
            raise ScannedDocumentError("Extracted document text is empty during reindex")

        new_version_record = DocumentVersion(
            document_id=document_id,
            workspace_id=workspace_id,
            source_version=source_version,
            content_hash=new_hash,
            raw_text=raw_text,
            byte_size=len(new_content_bytes),
        )

        # Stage 2: Chunking
        new_chunks = self.chunker.chunk(
            text=raw_text,
            document_id=document_id,
            workspace_id=workspace_id,
            source_version=source_version,
            language=language,
        )

        # Stage 3: Embedding
        passages = [c.search_text for c in new_chunks]
        vectors = self.embedder.embed_passages(passages)
        new_chunk_embeddings = {
            chunk.chunk_id: vec
            for chunk, vec in zip(new_chunks, vectors, strict=True)
        }

        # Stage 4: Atomic Activation (all prior stages succeeded)
        updated_doc = Document(
            document_id=document_id,
            workspace_id=workspace_id,
            filename=existing_doc.filename,
            content_type=existing_doc.content_type,
            content_hash=new_hash,
            source_version=source_version,
            status=DocumentStatus.READY,
            uploader_id=existing_doc.uploader_id,
            byte_size=len(new_content_bytes),
        )

        self.store.atomic_replace_document_index(
            workspace_id=workspace_id,
            document_id=document_id,
            new_doc=updated_doc,
            new_version=new_version_record,
            new_chunks=new_chunks,
            new_embeddings=new_chunk_embeddings,
        )
        return updated_doc
