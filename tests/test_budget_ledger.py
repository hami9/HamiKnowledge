"""Unit and concurrency tests for budget reservation and enforcement."""

from __future__ import annotations

import concurrent.futures
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from db.budget_ledger import BudgetExhaustedError, BudgetLedger
from db.memory_store import MemoryStore
from db.models import Workspace


def test_budget_reservation_and_reconciliation() -> None:
    ledger = BudgetLedger()
    ws_id = uuid4()

    budget = ledger.get_or_create_budget(ws_id, initial_budget_usd=1.0)
    assert budget.available_usd == 1.0

    # Reserve $0.40
    res_id = ledger.reserve(ws_id, 0.40)
    assert budget.reserved_usd == 0.40
    assert budget.available_usd == 0.60

    # Reconcile with actual cost $0.35 and 150 tokens
    ledger.reconcile(ws_id, res_id, actual_cost_usd=0.35, tokens_used=150)
    assert budget.reserved_usd == 0.0
    assert budget.spent_usd == 0.35
    assert budget.available_usd == 0.65
    assert budget.request_count == 1
    assert budget.total_tokens == 150


def test_budget_exhaustion_raises_error() -> None:
    ledger = BudgetLedger()
    ws_id = uuid4()

    ledger.get_or_create_budget(ws_id, initial_budget_usd=0.10)

    # First reservation takes $0.08
    ledger.reserve(ws_id, 0.08)

    # Second reservation requests $0.05 (only $0.02 available) -> must raise
    with pytest.raises(BudgetExhaustedError, match="insufficient budget"):
        ledger.reserve(ws_id, 0.05)


def test_concurrent_budget_reservations_prevent_overdraft() -> None:
    """Roadmap requirement: 20 concurrent requests against nearly exhausted budget."""
    ledger = BudgetLedger()
    ws_id = uuid4()

    # Total budget allows exactly 5 requests of $0.10 each
    ledger.get_or_create_budget(ws_id, initial_budget_usd=0.50)

    results: list[bool] = []

    def _attempt_reserve() -> bool:
        try:
            ledger.reserve(ws_id, 0.10)
            return True
        except BudgetExhaustedError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_attempt_reserve) for _ in range(20)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Exactly 5 should succeed, 15 should be rejected
    successes = sum(1 for r in results if r is True)
    failures = sum(1 for r in results if r is False)

    assert successes == 5
    assert failures == 15
    budget = ledger.get_or_create_budget(ws_id)
    assert budget.available_usd == 0.0


def test_api_budget_exhaustion_returns_402() -> None:
    test_store = MemoryStore()
    test_ledger = BudgetLedger()
    app = create_app()

    from apps.api.deps import get_ledger, get_store
    app.dependency_overrides[get_store] = lambda: test_store
    app.dependency_overrides[get_ledger] = lambda: test_ledger

    client = TestClient(app)

    ws_id = uuid4()
    test_store.create_workspace(Workspace(workspace_id=ws_id, name="Broke WS", slug="broke-ws"))
    # Set total budget to 0 USD
    test_ledger.get_or_create_budget(ws_id, initial_budget_usd=0.0)

    resp = client.post(
        f"/v1/workspaces/{ws_id}/query",
        json={"query": "test query"},
    )
    assert resp.status_code == 402
    assert "insufficient budget" in resp.json()["detail"].lower()
