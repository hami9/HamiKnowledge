"""Tests for document lifecycle, atomic reindexing, and cache invalidation.

Verifies roadmap invariants:
- Deletion immediately excludes content from queries, previews, and invalidates caches.
- A new index version activates only after every stage succeeds.
- Previous valid version stays usable through any failed reindex attempt.
"""

from __future__ import annotations

import io
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from db.cache import global_query_cache
from db.memory_store import MemoryStore
from db.models import Workspace
from ingestion.service import IngestionService, ScannedDocumentError


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_atomic_reindex_success_and_cache_invalidation(client: TestClient) -> None:
    # 1. Create workspace
    ws_res = client.post("/v1/workspaces", json={"name": "Lifecycle WS", "slug": "lifecycle-ws"})
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["workspace_id"]

    # 2. Upload initial document v1
    v1_text = "# Deployment Guide\n\nKubernetes Deployments manage replica sets for stateless apps."
    upload_res = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("deploy.md", io.BytesIO(v1_text.encode("utf-8")), "text/markdown")},
        data={"source_version": "v1"},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]
    assert upload_res.json()["source_version"] == "v1"

    # 3. Query document v1
    q1 = client.post(
        f"/v1/workspaces/{ws_id}/query",
        json={"query": "What do Kubernetes Deployments manage?", "variant": "B2"},
    )
    assert q1.status_code == 200
    assert not q1.json()["abstained"]
    assert "replica sets" in q1.json()["answer"]

    # Verify query response is now cached
    perm_ver_before = global_query_cache.get_permission_version(ws_id)

    # 4. Reindex with v2 content
    v2_text = (
        "# Deployment Guide\n\n"
        "Kubernetes Deployments now manage automated rollouts and rollbacks."
    )
    reindex_res = client.post(
        f"/v1/workspaces/{ws_id}/documents/{doc_id}/reindex",
        files={"file": ("deploy.md", io.BytesIO(v2_text.encode("utf-8")), "text/markdown")},
        data={"source_version": "v2"},
    )
    assert reindex_res.status_code == 200
    assert reindex_res.json()["source_version"] == "v2"

    # Permission version was bumped and cache invalidated
    perm_ver_after = global_query_cache.get_permission_version(ws_id)
    assert perm_ver_after != perm_ver_before

    # 5. Query document v2
    q2 = client.post(
        f"/v1/workspaces/{ws_id}/query",
        json={"query": "What do Kubernetes Deployments manage?", "variant": "B2"},
    )
    assert q2.status_code == 200
    assert "automated rollouts" in q2.json()["answer"]


def test_failed_reindex_preserves_previous_valid_version() -> None:
    store = MemoryStore()
    service = IngestionService(store=store)

    ws_id = uuid4()
    store.create_workspace(Workspace(workspace_id=ws_id, name="Fail WS", slug="fail-ws"))

    # Initial valid ingest
    doc, _ = service.ingest_document(
        workspace_id=ws_id,
        filename="notes.md",
        content_bytes=b"# Notes\n\nValid documentation text.",
        source_version="v1",
    )
    assert doc.source_version == "v1"
    initial_chunks = store.get_chunks_for_document(ws_id, doc.document_id)
    assert len(initial_chunks) > 0

    # Attempt reindex with invalid empty/scanned content (triggers ScannedDocumentError)
    with pytest.raises(ScannedDocumentError):
        service.reindex_document(
            workspace_id=ws_id,
            document_id=doc.document_id,
            new_content_bytes=b"   \n\t  ",  # empty text stream
            source_version="v2",
        )

    # Invariant: Document remains READY and previous chunks remain intact
    doc_after = store.get_document(ws_id, doc.document_id)
    assert doc_after is not None
    assert doc_after.source_version == "v1"
    assert doc_after.status.value == "READY"
    chunks_after = store.get_chunks_for_document(ws_id, doc.document_id)
    assert len(chunks_after) == len(initial_chunks)
    assert chunks_after[0].content == initial_chunks[0].content


def test_deletion_immediately_excludes_content_and_invalidates(client: TestClient) -> None:
    # 1. Create workspace & document
    ws_res = client.post("/v1/workspaces", json={"name": "Del WS", "slug": "del-ws"})
    ws_id = ws_res.json()["workspace_id"]

    text = "# Temporary Secret\n\nInternal key is SECRET_ALPHA_123."
    upload_res = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("secret.md", io.BytesIO(text.encode("utf-8")), "text/markdown")},
    )
    doc_id = upload_res.json()["document_id"]

    # 2. Query succeeds before deletion
    q1 = client.post(
        f"/v1/workspaces/{ws_id}/query",
        json={"query": "What is the internal key?", "variant": "B2"},
    )
    assert q1.status_code == 200
    assert not q1.json()["abstained"]
    assert "SECRET_ALPHA_123" in q1.json()["answer"]

    # 3. Delete document
    del_res = client.delete(f"/v1/workspaces/{ws_id}/documents/{doc_id}")
    assert del_res.status_code == 204

    # 4. Immediate exclusion: Excerpt returns 404
    exc_res = client.get(
        f"/v1/workspaces/{ws_id}/documents/{doc_id}/excerpt?start_char=0&end_char=20"
    )
    assert exc_res.status_code == 404

    # 5. Immediate exclusion: Query abstains, zero citations
    q2 = client.post(
        f"/v1/workspaces/{ws_id}/query",
        json={"query": "What is the internal key?", "variant": "B2"},
    )
    assert q2.status_code == 200
    assert q2.json()["abstained"] is True
    assert len(q2.json()["citations"]) == 0
