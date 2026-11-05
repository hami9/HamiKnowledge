"""FastAPI dependency injection utilities."""

from __future__ import annotations

from db.budget_ledger import BudgetLedger, global_ledger
from db.memory_store import MemoryStore, global_store
from ingestion.chunker import HeadingAwareChunker
from ingestion.service import IngestionService


def get_store() -> MemoryStore:
    return global_store


def get_ledger() -> BudgetLedger:
    return global_ledger


def get_ingestion_service() -> IngestionService:
    return IngestionService(store=global_store, chunker=HeadingAwareChunker())
