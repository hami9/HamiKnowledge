"""Thread-safe budget reservation, accounting, and usage ledger.

Enforces roadmap financial and rate contracts:
- Atomically reserves an upper bound before any provider call.
- Raises BudgetExhaustedError when remaining funds are insufficient.
- Reconciles actual consumption after call completion.
- Scoped strictly per workspace.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from uuid import UUID, uuid4


class BudgetError(Exception):
    """Base exception for budget and accounting operations."""


class BudgetExhaustedError(BudgetError):
    """Raised when workspace balance cannot cover the reservation request."""


@dataclass
class ReservationRecord:
    reservation_id: str
    workspace_id: UUID
    amount_usd: float
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    reconciled: bool = False


@dataclass
class WorkspaceBudget:
    workspace_id: UUID
    total_budget_usd: float = 10.00  # Default starter allowance
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    request_count: int = 0
    total_tokens: int = 0

    @property
    def available_usd(self) -> float:
        return max(0.0, round(self.total_budget_usd - self.spent_usd - self.reserved_usd, 8))


class BudgetLedger:
    """Thread-safe ledger enforcing atomic reservation and post-call reconciliation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._budgets: dict[UUID, WorkspaceBudget] = {}
        self._reservations: dict[str, ReservationRecord] = {}

    def get_or_create_budget(
        self,
        workspace_id: UUID,
        initial_budget_usd: float = 10.00,
    ) -> WorkspaceBudget:
        with self._lock:
            if workspace_id not in self._budgets:
                self._budgets[workspace_id] = WorkspaceBudget(
                    workspace_id=workspace_id,
                    total_budget_usd=initial_budget_usd,
                )
            return self._budgets[workspace_id]

    def reserve(
        self,
        workspace_id: UUID,
        estimated_cost_usd: float,
    ) -> str:
        """Atomically reserve funds before calling external inference/embedding services."""
        if estimated_cost_usd <= 0.0:
            estimated_cost_usd = 0.0001  # Minimum nominal reserve

        with self._lock:
            budget = self.get_or_create_budget(workspace_id)
            if budget.available_usd < estimated_cost_usd:
                raise BudgetExhaustedError(
                    f"Workspace {workspace_id} has insufficient budget "
                    f"({budget.available_usd:.4f} USD available, "
                    f"{estimated_cost_usd:.4f} USD required)"
                )

            reservation_id = f"res_{uuid4().hex[:12]}"
            budget.reserved_usd = round(budget.reserved_usd + estimated_cost_usd, 8)

            self._reservations[reservation_id] = ReservationRecord(
                reservation_id=reservation_id,
                workspace_id=workspace_id,
                amount_usd=estimated_cost_usd,
            )
            return reservation_id

    def reconcile(
        self,
        workspace_id: UUID,
        reservation_id: str,
        actual_cost_usd: float,
        tokens_used: int = 0,
    ) -> None:
        """Reconcile reserved amount with actual usage reported or estimated."""
        with self._lock:
            record = self._reservations.get(reservation_id)
            if not record or record.reconciled or record.workspace_id != workspace_id:
                return

            budget = self.get_or_create_budget(workspace_id)
            # Remove from reserved
            budget.reserved_usd = max(0.0, round(budget.reserved_usd - record.amount_usd, 8))
            # Add to spent
            budget.spent_usd = round(budget.spent_usd + actual_cost_usd, 8)
            budget.request_count += 1
            budget.total_tokens += tokens_used

            record.reconciled = True

    def release(
        self,
        workspace_id: UUID,
        reservation_id: str,
    ) -> None:
        """Release reservation if operation was aborted before any provider execution."""
        with self._lock:
            record = self._reservations.get(reservation_id)
            if not record or record.reconciled or record.workspace_id != workspace_id:
                return

            budget = self.get_or_create_budget(workspace_id)
            budget.reserved_usd = max(0.0, round(budget.reserved_usd - record.amount_usd, 8))
            record.reconciled = True

    def add_funds(self, workspace_id: UUID, amount_usd: float) -> WorkspaceBudget:
        if amount_usd <= 0:
            raise ValueError("Amount must be positive")
        with self._lock:
            budget = self.get_or_create_budget(workspace_id)
            budget.total_budget_usd = round(budget.total_budget_usd + amount_usd, 8)
            return budget


# Global ledger instance
global_ledger = BudgetLedger()
