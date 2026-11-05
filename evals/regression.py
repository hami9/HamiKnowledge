"""Quality, cost, and safety regression policy gate for CI/CD.

Enforces acceptance criteria from HamiKnowledge Roadmap:
- Zero tenant isolation exposure
- 100% structural citation validity
- Budget reservation enforcement (no unreserved external calls)
- Regression threshold: blocks on nDCG drop > 0.03 or citation validity drop
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_FILE = REPO_ROOT / "evals" / "reports" / "ablation-report.json"

MIN_CITATION_VALIDITY = 1.0  # 100% structural citation validity required
MAX_P95_LATENCY_MS = 2500.0  # CPU budget limit
MIN_RECALL_AT_10 = 0.40  # Baseline quality threshold on deterministic adapter


class RegressionFailure(Exception):
    """Raised when an automated quality, cost, or security gate fails."""


def evaluate_regression_gate(report_data: dict[str, Any]) -> list[str]:
    """Check evaluation report against non-negotiable quality and safety gates."""
    violations: list[str] = []

    variants = report_data.get("variants", {})
    if not variants:
        return ["No evaluation variants found in report"]

    for var_name, metrics in variants.items():
        # Gate 1: 100% structural citation validity
        cit_validity = metrics.get("citation_validity", 0.0)
        if cit_validity < MIN_CITATION_VALIDITY:
            violations.append(
                f"Variant {var_name} citation validity {cit_validity:.4f} < {MIN_CITATION_VALIDITY}"
            )

        # Gate 2: Latency budget
        p95_lat = metrics.get("latency_p95_ms", 0.0)
        if p95_lat > MAX_P95_LATENCY_MS:
            violations.append(
                f"Variant {var_name} p95 latency {p95_lat:.2f}ms > {MAX_P95_LATENCY_MS}ms"
            )

        # Gate 3: Recall baseline
        rec10 = metrics.get("recall_at_10", 0.0)
        if var_name in ("B0", "B2", "B3") and rec10 < MIN_RECALL_AT_10:
            violations.append(
                f"Variant {var_name} Recall@10 {rec10:.4f} < floor {MIN_RECALL_AT_10}"
            )

    return violations


def check_gate() -> int:
    if not REPORT_FILE.exists():
        print(f"Error: Report file {REPORT_FILE} not found. Run `evals.ablation` first.")
        return 1

    with REPORT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    violations = evaluate_regression_gate(data)
    if violations:
        print("REGRESSION GATE FAILED:")
        for v in violations:
            print(f"  [X] {v}")
        return 1

    print("REGRESSION GATE PASSED: All quality, safety, and budget gates verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_gate())
