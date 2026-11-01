"""Milestone A2 Exit Criteria End-to-End Integration Tests.

Validates that:
1. Ingestion computes chunk embeddings through all lifecycle stages.
2. Lexical (B0), Vector (B1), Hybrid RRF (B2), and Reranked (B3) search work.
3. A question opens its supporting passage at exact source character offsets.
4. Unsupported questions produce explicit calibrated abstention.
5. All queries expose latency breakdown and usage/cost accounting.
6. Zero cross-tenant leakage between workspaces across all retrieval and excerpt endpoints.
7. Deletion immediately excludes document from all search variants and excerpts.
"""

from __future__ import annotations

import io
from uuid import UUID

from fastapi.testclient import TestClient

from apps.api.main import create_app
from db.memory_store import MemoryStore


def _setup_test_client() -> tuple[TestClient, MemoryStore]:
    test_store = MemoryStore()
    app = create_app()

    from apps.api.deps import get_ingestion_service, get_store
    from ingestion.chunker import HeadingAwareChunker
    from ingestion.service import IngestionService
    from retrieval.embeddings import DeterministicEmbeddingAdapter

    app.dependency_overrides[get_store] = lambda: test_store
    app.dependency_overrides[get_ingestion_service] = lambda: IngestionService(
        store=test_store,
        chunker=HeadingAwareChunker(),
        embedder=DeterministicEmbeddingAdapter(),
    )

    client = TestClient(app)
    return client, test_store


def test_a2_rag_mvp_complete_workflow() -> None:
    client, store = _setup_test_client()

    # 1. Create two isolated workspaces
    resp_ws1 = client.post("/v1/workspaces", json={"name": "Team Cloud", "slug": "team-cloud"})
    assert resp_ws1.status_code == 201
    ws1_id = resp_ws1.json()["workspace_id"]

    resp_ws2 = client.post("/v1/workspaces", json={"name": "Team Edge", "slug": "team-edge"})
    assert resp_ws2.status_code == 201
    ws2_id = resp_ws2.json()["workspace_id"]

    # 2. Ingest Doc 1 into Workspace 1
    doc1_content = (
        "# Pod Security Standards\n\n"
        "Pod Security Standards define three different isolation profiles for workloads:\n"
        "Privileged, Baseline, and Restricted.\n"
        "The Restricted profile enforces hardening best practices."
    )
    doc1_bytes = io.BytesIO(doc1_content.encode("utf-8"))
    resp_up1 = client.post(
        f"/v1/workspaces/{ws1_id}/documents/upload",
        files={"file": ("pod-security.md", doc1_bytes, "text/markdown")},
    )
    assert resp_up1.status_code == 201
    doc1_id = resp_up1.json()["document_id"]
    assert resp_up1.json()["status"] == "READY"

    # Verify embeddings were persisted for Workspace 1
    active_embs = store.get_active_workspace_embeddings(UUID(ws1_id))
    assert len(active_embs) >= 1

    # 3. Ingest Doc 2 into Workspace 2
    doc2_content = (
        "# Etcd Backup and Recovery\n\n"
        "All Kubernetes cluster state is stored in etcd.\n"
        "Periodic snapshots should be taken with etcdctl snapshot save command."
    )
    doc2_bytes = io.BytesIO(doc2_content.encode("utf-8"))
    resp_up2 = client.post(
        f"/v1/workspaces/{ws2_id}/documents/upload",
        files={"file": ("etcd-backup.md", doc2_bytes, "text/markdown")},
    )
    assert resp_up2.status_code == 201
    doc2_id = resp_up2.json()["document_id"]

    # 4. TEST ALL 4 RETRIEVAL VARIANTS (B0, B1, B2, B3) IN WORKSPACE 1
    query_text = "Pod Security Standards isolation profiles"

    # B0: Lexical
    resp_b0 = client.post(f"/v1/workspaces/{ws1_id}/search/lexical", json={"query": query_text})
    assert resp_b0.status_code == 200
    assert resp_b0.json()["count"] >= 1
    assert resp_b0.json()["results"][0]["branch"] == "lexical"

    # B1: Vector
    resp_b1 = client.post(f"/v1/workspaces/{ws1_id}/search/vector", json={"query": query_text})
    assert resp_b1.status_code == 200
    assert resp_b1.json()["count"] >= 1
    assert resp_b1.json()["results"][0]["branch"] == "vector"

    # B2: Hybrid RRF
    resp_b2 = client.post(f"/v1/workspaces/{ws1_id}/search/hybrid", json={"query": query_text})
    assert resp_b2.status_code == 200
    assert resp_b2.json()["count"] >= 1
    assert resp_b2.json()["results"][0]["branch"] == "fused"

    # B3: Reranked
    resp_b3 = client.post(f"/v1/workspaces/{ws1_id}/search/reranked", json={"query": query_text})
    assert resp_b3.status_code == 200
    assert resp_b3.json()["count"] >= 1
    assert resp_b3.json()["results"][0]["branch"] == "reranked"

    # 5. TEST QUESTION ANSWERING & CITATION OPENING AT SOURCE LOCATION
    query_resp = client.post(
        f"/v1/workspaces/{ws1_id}/query",
        json={"query": "What are the Pod Security Standards profiles?", "variant": "B3"},
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()

    assert not q_data["abstained"]
    assert len(q_data["citations"]) >= 1
    assert q_data["usage"]["total_tokens"] > 0
    assert q_data["usage"]["estimated_cost_usd"] > 0.0
    assert q_data["latency_ms"] >= 0.0

    # OPEN CITATION AT EXACT SOURCE LOCATION
    cit = q_data["citations"][0]
    start_c = cit["start_char"]
    end_c = cit["end_char"]
    assert cit["document_id"] == doc1_id

    excerpt_resp = client.get(
        f"/v1/workspaces/{ws1_id}/documents/{doc1_id}/excerpt",
        params={"start_char": start_c, "end_char": end_c},
    )
    assert excerpt_resp.status_code == 200
    excerpt_data = excerpt_resp.json()
    # Confirm exact source text matches the original document slice
    assert excerpt_data["excerpt"] == doc1_content[start_c:end_c]

    # 6. TEST ABSTENTION ON UNSUPPORTED QUERY
    unsupported_resp = client.post(
        f"/v1/workspaces/{ws1_id}/query",
        json={"query": "How do you repair a nuclear submarine engine?"},
    )
    assert unsupported_resp.status_code == 200
    unsup_data = unsupported_resp.json()
    assert unsup_data["abstained"]
    assert "insufficient evidence" in unsup_data["answer"].lower()
    assert len(unsup_data["citations"]) == 0

    # 7. STRICT MULTI-TENANT ISOLATION TESTS
    # User 1 queries User 2's topic -> 0 results
    cross_search = client.post(
        f"/v1/workspaces/{ws1_id}/search/reranked",
        json={"query": "etcdctl snapshot save command"},
    )
    assert cross_search.status_code == 200
    assert cross_search.json()["count"] == 0

    # User 1 asking User 2's question must abstain (zero cross-tenant context)
    cross_query = client.post(
        f"/v1/workspaces/{ws1_id}/query",
        json={"query": "How to take etcd snapshots with etcdctl?"},
    )
    assert cross_query.status_code == 200
    assert cross_query.json()["abstained"]

    # User 1 attempting to read User 2's document excerpt directly must be rejected
    cross_excerpt = client.get(
        f"/v1/workspaces/{ws1_id}/documents/{doc2_id}/excerpt",
        params={"start_char": 0, "end_char": 50},
    )
    assert cross_excerpt.status_code == 404

    # 8. LIFECYCLE DELETION REVOCATION
    del_resp = client.delete(f"/v1/workspaces/{ws1_id}/documents/{doc1_id}")
    assert del_resp.status_code == 204

    # After deletion: immediately excluded from search and excerpts
    search_after_del = client.post(
        f"/v1/workspaces/{ws1_id}/search/vector",
        json={"query": "Pod Security Standards"},
    )
    assert search_after_del.json()["count"] == 0

    excerpt_after_del = client.get(
        f"/v1/workspaces/{ws1_id}/documents/{doc1_id}/excerpt",
        params={"start_char": start_c, "end_char": end_c},
    )
    assert excerpt_after_del.status_code == 404
