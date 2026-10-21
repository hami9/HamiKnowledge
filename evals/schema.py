"""Evaluation dataset schema.

Relevance labels are anchored to verbatim quotes in pinned corpus documents, not to
chunk IDs, so labels stay valid when chunking parameters change. The retrieval
evaluator maps a quote to every chunk whose span overlaps it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Split = Literal["dev", "test", "adversarial"]
Language = Literal["en", "fa", "mixed"]
Category = Literal[
    "fact_lookup",
    "multi_passage",
    "exact_term",
    "ambiguous",
    "unanswerable",
    "prompt_injection",
    "misleading_citation",
    "deleted_content",
    "acl_change",
    "tool_misuse",
]
ExpectedBehavior = Literal["answer", "abstain", "ignore_injection", "exclude_revoked"]

SPLIT_PREFIX: dict[str, str] = {"dev": "dev", "test": "test", "adversarial": "adv"}
SPLIT_FILES: dict[str, str] = {
    "dev": "dev.jsonl",
    "test": "test.jsonl",
    "adversarial": "adversarial.jsonl",
}
DEV_POOL_PERCENT = 40


class Relevance(BaseModel):
    doc_path: str = Field(min_length=1)
    quote: str = Field(min_length=12, max_length=600)
    grade: int = Field(ge=1, le=3, description="3 answers, 2 partially answers, 1 related")


class Review(BaseModel):
    status: Literal["draft", "reviewed", "rejected"]
    reviewer: str | None = None
    reviewed_at: dt.date | None = None

    @model_validator(mode="after")
    def _reviewed_has_reviewer(self) -> Review:
        if self.status != "draft" and (not self.reviewer or not self.reviewed_at):
            raise ValueError("reviewed/rejected items need reviewer and reviewed_at")
        return self


class Fixture(BaseModel):
    """A synthetic document uploaded only for an adversarial/lifecycle scenario."""

    title: str = Field(min_length=1)
    text: str = Field(min_length=20)


class EvalItem(BaseModel):
    id: str = Field(pattern=r"^(dev|test|adv)-\d{3}$")
    split: Split
    family_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,80}$")
    language: Language
    category: Category
    question: str = Field(min_length=8, max_length=600)
    answerable: bool
    expected_behavior: ExpectedBehavior
    relevant: list[Relevance] = Field(default_factory=list)
    answer_facts: list[str] = Field(default_factory=list)
    fixture: Fixture | None = None
    scenario: str | None = Field(default=None, max_length=800)
    rubric_version: str = Field(pattern=r"^rubric-v\d+$")
    review: Review
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EvalItem:
        if not self.id.startswith(SPLIT_PREFIX[self.split] + "-"):
            raise ValueError(f"id {self.id} does not match split {self.split}")
        if self.answerable:
            if self.expected_behavior not in ("answer", "ignore_injection"):
                raise ValueError("answerable items must expect answer or ignore_injection")
            if not any(r.grade == 3 for r in self.relevant):
                raise ValueError("answerable items need at least one grade-3 passage")
            if not self.answer_facts:
                raise ValueError("answerable items need answer_facts")
        else:
            if self.expected_behavior == "answer":
                raise ValueError("unanswerable items cannot expect an answer")
            if self.category not in ("deleted_content", "acl_change") and any(
                r.grade > 1 for r in self.relevant
            ):
                raise ValueError("unanswerable items may only list grade-1 related passages")
        if self.split != "adversarial":
            if self.category in (
                "prompt_injection",
                "misleading_citation",
                "deleted_content",
                "acl_change",
                "tool_misuse",
            ):
                raise ValueError(f"category {self.category} belongs to the adversarial split")
            if self.fixture or self.scenario:
                raise ValueError("fixture/scenario are only allowed in the adversarial split")
            if self.category == "unanswerable" and self.answerable:
                raise ValueError("unanswerable category must set answerable=false")
        elif not self.scenario:
            raise ValueError("adversarial items need a scenario")
        return self


def normalize_for_match(text: str) -> str:
    """NFC plus whitespace collapse; used only to locate quotes, never for display."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def pool_key(doc_path: str) -> str:
    """Translations share a pool with their English source to keep near-duplicates together."""
    if doc_path.startswith("content/fa/"):
        return "content/en/" + doc_path[len("content/fa/") :]
    return doc_path


def assigned_pool(doc_path: str) -> Literal["dev", "test"]:
    digest = hashlib.sha256(pool_key(doc_path).encode("utf-8")).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 100 < DEV_POOL_PERCENT else "test"


def load_items(path: Path) -> list[EvalItem]:
    if not path.exists():
        return []
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                items.append(EvalItem.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path.name}:{lineno}: {exc}") from exc
    return items


class SplitAccessError(RuntimeError):
    """Raised when frozen evaluation data is requested for tuning."""


def load_split(
    datasets_dir: Path, split: Split, purpose: Literal["tuning", "final", "lifecycle"]
) -> list[EvalItem]:
    """Single entry point for experiment code; refuses frozen splits for tuning."""
    if purpose == "tuning" and split != "dev":
        raise SplitAccessError(f"split '{split}' is frozen and cannot be used for tuning")
    if purpose == "lifecycle" and split != "adversarial":
        raise SplitAccessError("lifecycle runs only use the adversarial split")
    return load_items(datasets_dir / SPLIT_FILES[split])
