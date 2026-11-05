"""Unit tests for the CI quality regression gate."""

from __future__ import annotations

from evals.regression import evaluate_regression_gate


def test_regression_gate_passes_valid_report() -> None:
    valid_report = {
        "variants": {
            "B0": {
                "citation_validity": 1.0,
                "latency_p95_ms": 500.0,
                "recall_at_10": 0.50,
            },
            "B2": {
                "citation_validity": 1.0,
                "latency_p95_ms": 1200.0,
                "recall_at_10": 0.45,
            },
            "B3": {
                "citation_validity": 1.0,
                "latency_p95_ms": 1400.0,
                "recall_at_10": 0.45,
            },
        }
    }
    violations = evaluate_regression_gate(valid_report)
    assert len(violations) == 0


def test_regression_gate_flags_citation_invalidity() -> None:
    bad_report = {
        "variants": {
            "B3": {
                "citation_validity": 0.95,  # Less than 1.0
                "latency_p95_ms": 500.0,
                "recall_at_10": 0.60,
            }
        }
    }
    violations = evaluate_regression_gate(bad_report)
    assert len(violations) == 1
    assert "citation validity" in violations[0]


def test_regression_gate_flags_excessive_latency() -> None:
    slow_report = {
        "variants": {
            "B2": {
                "citation_validity": 1.0,
                "latency_p95_ms": 3500.0,  # Exceeds 2500ms
                "recall_at_10": 0.55,
            }
        }
    }
    violations = evaluate_regression_gate(slow_report)
    assert len(violations) == 1
    assert "p95 latency" in violations[0]
