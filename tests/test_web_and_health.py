"""Tests for health checks and web dashboard serving."""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_healthz_and_readyz() -> None:
    client = TestClient(create_app())
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "healthy"}

    res_ready = client.get("/readyz")
    assert res_ready.status_code == 200
    assert res_ready.json() == {"status": "ready"}


def test_web_dashboard_root() -> None:
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "HamiKnowledge" in res.text
    assert "Vazirmatn" in res.text
    assert "results-viewport" in res.text
