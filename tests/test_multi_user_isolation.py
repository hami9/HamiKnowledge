"""Milestone A1 Exit Criteria Integration Tests.

Validates that:
1. Two users in separate workspaces ingest content with zero cross-tenant leakage.
2. Duplicate uploads are handled idempotently without duplicating index state.
3. Invalid or oversized files fail predictably with clear error messages.
4. Deletion immediately excludes content from search queries.
"""

from __future__ import annotations

import io
from uuid import UUID

from fastapi.testclient import TestClient

from apps.api.main import create_app
from db.memory_store import MemoryStore


def _setup_client() -> tuple[TestClient, MemoryStore]:
    # Isolated store for clean test run
    test_store = MemoryStore()
    app = create_app()

    from apps.api.deps import get_ingestion_service, get_store
    from ingestion.chunker import HeadingAwareChunker
    from ingestion.service import IngestionService

    app.dependency_overrides[get_store] = lambda: test_store
    app.dependency_overrides[get_ingestion_service] = lambda: IngestionService(
        store=test_store, chunker=HeadingAwareChunker()
    )

    client = TestClient(app)
    return client, test_store


def test_two_users_ingest_separate_content_with_strict_isolation() -> None:
    client, _ = _setup_client()

    # 1. Create Workspace 1 (User A) and Workspace 2 (User B)
    resp_ws1 = client.post("/v1/workspaces", json={"name": "Team Alpha", "slug": "team-alpha"})
    assert resp_ws1.status_code == 201
    ws1_id = resp_ws1.json()["workspace_id"]

    resp_ws2 = client.post("/v1/workspaces", json={"name": "Team Beta", "slug": "team-beta"})
    assert resp_ws2.status_code == 201
    ws2_id = resp_ws2.json()["workspace_id"]

    # 2. User A uploads Doc 1 into Workspace 1
    doc1_content = (
        "# Pod Security Standards\n\n"
        "Pod Security Standards define different isolation levels for Pods.\n"
        "These include Privileged, Baseline, and Restricted profiles."
    )
    doc1_file = io.BytesIO(doc1_content.encode("utf-8"))
    resp_up1 = client.post(
        f"/v1/workspaces/{ws1_id}/documents/upload",
        files={"file": ("pod-security.md", doc1_file, "text/markdown")},
    )
    assert resp_up1.status_code == 201
    doc1_data = resp_up1.json()
    assert doc1_data["status"] == "READY"
    assert not doc1_data["is_duplicate"]

    # 3. User B uploads Doc 2 into Workspace 2
    doc2_content = (
        "# Kubeadm Troubleshooting\n\n"
        "When setting up a high availability etcd cluster with kubeadm,\n"
        "ensure all etcd peer certificates and endpoints are properly configured."
    )
    doc2_file = io.BytesIO(doc2_content.encode("utf-8"))
    resp_up2 = client.post(
        f"/v1/workspaces/{ws2_id}/documents/upload",
        files={"file": ("kubeadm-ha.md", doc2_file, "text/markdown")},
    )
    assert resp_up2.status_code == 201
    doc2_data = resp_up2.json()
    assert doc2_data["status"] == "READY"
    assert not doc2_data["is_duplicate"]

    # 4. CROSS-TENANT ISOLATION CHECK:
    # User A searches for "etcd peer certificates" (terms from Workspace 2)
    search_a = client.post(
        f"/v1/workspaces/{ws1_id}/search/lexical",
        json={"query": "etcd peer certificates"},
    )
    assert search_a.status_code == 200
    # ZERO EXPOSURE: must return 0 results
    assert search_a.json()["count"] == 0

    # User B searches for "Pod Security Standards" (terms from Workspace 1)
    search_b = client.post(
        f"/v1/workspaces/{ws2_id}/search/lexical",
        json={"query": "Pod Security Standards"},
    )
    assert search_b.status_code == 200
    # ZERO EXPOSURE: must return 0 results
    assert search_b.json()["count"] == 0

    # 5. INTRA-TENANT RETRIEVAL CHECK:
    # User A searches for their own document
    search_a_own = client.post(
        f"/v1/workspaces/{ws1_id}/search/lexical",
        json={"query": "Pod Security Standards"},
    )
    assert search_a_own.status_code == 200
    res_a = search_a_own.json()
    assert res_a["count"] >= 1
    assert res_a["results"][0]["document_id"] == doc1_data["document_id"]
    # Check that location offsets correctly extract text from original content
    loc_start = res_a["results"][0]["start_char"]
    loc_end = res_a["results"][0]["end_char"]
    assert doc1_content[loc_start:loc_end] == res_a["results"][0]["content"]


def test_duplicate_upload_is_idempotent() -> None:
    client, store = _setup_client()

    resp_ws = client.post("/v1/workspaces", json={"name": "Dev Team", "slug": "dev-team"})
    ws_id = resp_ws.json()["workspace_id"]

    file_bytes = b"# Architecture Overview\n\nMicroservices platform architecture."

    # First upload
    up1 = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("arch.md", io.BytesIO(file_bytes), "text/markdown")},
    )
    assert up1.status_code == 201
    data1 = up1.json()
    assert data1["status"] == "READY"
    assert not data1["is_duplicate"]
    initial_doc_id = data1["document_id"]

    # Second upload with identical bytes
    up2 = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("arch.md", io.BytesIO(file_bytes), "text/markdown")},
    )
    assert up2.status_code == 201
    data2 = up2.json()
    assert data2["is_duplicate"]
    assert data2["document_id"] == initial_doc_id

    # Verify chunks were not duplicated
    chunks = store.get_chunks_for_document(
        workspace_id=UUID(ws_id),
        document_id=UUID(initial_doc_id),
    )
    assert len(chunks) == 1


def test_validation_rejects_empty_and_oversized_files() -> None:
    client, _ = _setup_client()

    resp_ws = client.post("/v1/workspaces", json={"name": "Test WS", "slug": "test-ws"})
    ws_id = resp_ws.json()["workspace_id"]

    # Empty file
    up_empty = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("empty.md", io.BytesIO(b""), "text/markdown")},
    )
    assert up_empty.status_code == 400
    assert "empty" in up_empty.json()["detail"].lower()

    # Oversized file (> 10 MB)
    oversized = b"a" * (10 * 1024 * 1024 + 1)
    up_huge = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("huge.md", io.BytesIO(oversized), "text/markdown")},
    )
    assert up_huge.status_code == 400
    assert "10 mb" in up_huge.json()["detail"].lower()


def test_deletion_immediately_revokes_search_results() -> None:
    client, _ = _setup_client()

    resp_ws = client.post("/v1/workspaces", json={"name": "Lifecycle WS", "slug": "lifecycle-ws"})
    ws_id = resp_ws.json()["workspace_id"]

    content = "# Secret Protocol\n\nConfidential internal procedure."
    up = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("secret.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
    )
    assert up.status_code == 201
    doc_id = up.json()["document_id"]

    # Confirm searchable before deletion
    search_before = client.post(
        f"/v1/workspaces/{ws_id}/search/lexical",
        json={"query": "Confidential internal"},
    )
    assert search_before.json()["count"] == 1

    # Delete document
    del_resp = client.delete(f"/v1/workspaces/{ws_id}/documents/{doc_id}")
    assert del_resp.status_code == 204

    # Must immediately be excluded from search
    search_after = client.post(
        f"/v1/workspaces/{ws_id}/search/lexical",
        json={"query": "Confidential internal"},
    )
    assert search_after.json()["count"] == 0
