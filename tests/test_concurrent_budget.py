"""Concurrent budget enforcement stress tests.

Satisfies roadmap invariant:
20 concurrent requests against a nearly exhausted workspace budget:
- No unreserved provider call.
- Immediate, clear budget-exhausted response (HTTP 402 / BudgetExhaustedError).
- Thread-safe ledger invariant: spent + reserved <= max_budget at all times.
"""

from __future__ import annotations

import concurrent.futures
import io
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from db.budget_ledger import BudgetExhaustedError, BudgetLedger


def test_concurrent_budget_reservations_ledger_level() -> None:
    ledger = BudgetLedger()
    ws_id = uuid4()
    # Initial budget allows exactly 5 reservations of $0.001
    ledger.set_budget(ws_id, 0.005)

    num_threads = 20
    reservation_amount = 0.001

    success_reservations: list[str] = []
    exhausted_exceptions: list[BudgetExhaustedError] = []

    def attempt_reservation(thread_idx: int) -> str | None:
        try:
            res_id = ledger.reserve(ws_id, reservation_amount)
            return res_id
        except BudgetExhaustedError as exc:
            exhausted_exceptions.append(exc)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(attempt_reservation, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            res_id = f.result()
            if res_id:
                success_reservations.append(res_id)

    # Invariants verification
    assert len(success_reservations) == 5
    assert len(exhausted_exceptions) == 15

    state = ledger.get_or_create_budget(ws_id)
    assert state is not None
    assert state.reserved_usd == pytest.approx(0.005, abs=1e-6)
    assert state.spent_usd == 0.0
    assert state.available_usd == pytest.approx(0.0, abs=1e-6)

    # Now concurrently reconcile all 5 reservations with actual cost $0.0006 each
    def attempt_reconcile(res_id: str) -> None:
        ledger.reconcile(ws_id, res_id, actual_cost_usd=0.0006, tokens_used=10)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(attempt_reconcile, success_reservations))

    state_after = ledger.get_or_create_budget(ws_id)
    assert state_after is not None
    assert state_after.reserved_usd == 0.0
    assert state_after.spent_usd == pytest.approx(0.003, abs=1e-6)
    assert state_after.available_usd == pytest.approx(0.002, abs=1e-6)


def test_concurrent_query_endpoint_under_budget_pressure() -> None:
    app = create_app()
    client = TestClient(app)

    # 1. Create workspace
    ws_res = client.post("/v1/workspaces", json={"name": "Stress WS", "slug": "stress-ws"})
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["workspace_id"]

    # 2. Upload a test document
    doc_content = "# Architecture\n\nMicroservices scale independently."
    upload_res = client.post(
        f"/v1/workspaces/{ws_id}/documents/upload",
        files={"file": ("arch.md", io.BytesIO(doc_content.encode("utf-8")), "text/markdown")},
    )
    assert upload_res.status_code == 201

    # 3. Configure tight budget of $0.001 (each /query reserves $0.001 ceiling)
    set_budget_res = client.put(
        f"/v1/workspaces/{ws_id}/budget",
        json={"total_budget_usd": 0.001},
    )
    assert set_budget_res.status_code == 200

    # 4. Fire 20 concurrent requests
    num_requests = 20
    status_codes: list[int] = []

    def make_query(idx: int) -> int:
        thread_client = TestClient(app)
        res = thread_client.post(
            f"/v1/workspaces/{ws_id}/query",
            json={"query": f"How do microservices scale? query_{idx}", "variant": "B2"},
        )
        return int(res.status_code)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_query, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            status_codes.append(f.result())

    # 5. Invariant checks:
    # Exactly 1 request succeeds in reserving the entire remaining budget ceiling.
    # All other 19 requests receive HTTP 402 Payment Required without making unreserved calls.
    num_200 = status_codes.count(200)
    num_402 = status_codes.count(402)

    assert num_200 == 1, f"Expected exactly 1 HTTP 200 response, got {num_200}"
    assert num_402 == 19, f"Expected exactly 19 HTTP 402 responses, got {num_402}"
    assert len(status_codes) == 20
