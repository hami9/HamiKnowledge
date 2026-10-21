from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import ValidationError

from evals.corpus import CorpusDocument, extract_title
from evals.schema import (
    EvalItem,
    SplitAccessError,
    assigned_pool,
    load_split,
    normalize_for_match,
    pool_key,
)
from evals.validate import check_items, check_targets


def _doc(path: str, lang: Literal["en", "fa"] = "en") -> CorpusDocument:
    return CorpusDocument(
        doc_id=f"c:{path}", path=path, lang=lang, title="t", bytes=1, sha256="0" * 64
    )


def _paths_in_pool(pool: str, n: int, prefix: str = "content/en/docs/x") -> list[str]:
    found: list[str] = []
    i = 0
    while len(found) < n:
        path = f"{prefix}{i}.md"
        if assigned_pool(path) == pool:
            found.append(path)
        i += 1
    return found


def _item(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "dev-001",
        "split": "dev",
        "family_id": "fam-one",
        "language": "en",
        "category": "fact_lookup",
        "question": "What does the kubelet do?",
        "answerable": True,
        "expected_behavior": "answer",
        "relevant": [{"doc_path": "p.md", "quote": "The kubelet is an agent", "grade": 3}],
        "answer_facts": ["kubelet is a node agent"],
        "rubric_version": "rubric-v1",
        "review": {"status": "draft"},
    }
    base.update(overrides)
    return base


def test_pool_assignment_is_deterministic_and_pairs_translations() -> None:
    path = "content/en/docs/concepts/overview/_index.md"
    assert assigned_pool(path) == assigned_pool(path)
    fa = "content/fa/docs/concepts/overview/_index.md"
    assert pool_key(fa) == path
    assert assigned_pool(fa) == assigned_pool(path)


def test_answerable_item_requires_grade_three_and_facts() -> None:
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(relevant=[{"doc_path": "p", "quote": "x" * 20, "grade": 2}]))
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(answer_facts=[]))


def test_unanswerable_item_rejects_strong_labels() -> None:
    with pytest.raises(ValidationError):
        EvalItem.model_validate(
            _item(category="unanswerable", answerable=False, expected_behavior="abstain")
        )
    item = EvalItem.model_validate(
        _item(
            category="unanswerable",
            answerable=False,
            expected_behavior="abstain",
            relevant=[],
            answer_facts=[],
        )
    )
    assert not item.answerable


def test_adversarial_categories_stay_in_adversarial_split() -> None:
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(category="prompt_injection"))
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(id="adv-001", split="adversarial"))  # missing scenario


def test_id_prefix_must_match_split() -> None:
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(id="test-001"))


def test_reviewed_status_needs_reviewer() -> None:
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(review={"status": "reviewed"}))


def test_frozen_splits_cannot_be_loaded_for_tuning(tmp_path: Any) -> None:
    with pytest.raises(SplitAccessError):
        load_split(tmp_path, "test", "tuning")
    with pytest.raises(SplitAccessError):
        load_split(tmp_path, "adversarial", "tuning")
    assert load_split(tmp_path, "dev", "tuning") == []


def test_check_items_enforces_pools_quotes_and_families() -> None:
    dev_path = _paths_in_pool("dev", 1)[0]
    test_path = _paths_in_pool("test", 1)[0]
    corpus = {p: _doc(p) for p in (dev_path, test_path)}
    texts = {
        dev_path: normalize_for_match("The kubelet is an agent that runs on each node."),
        test_path: normalize_for_match("The scheduler assigns Pods to nodes."),
    }
    good_dev = EvalItem.model_validate(
        _item(relevant=[{"doc_path": dev_path, "quote": "The kubelet is an  agent", "grade": 3}])
    )
    assert check_items({"dev": [good_dev]}, corpus, texts) == []

    wrong_pool = EvalItem.model_validate(
        _item(relevant=[{"doc_path": test_path, "quote": "The scheduler assigns Pods", "grade": 3}])
    )
    errors = check_items({"dev": [wrong_pool]}, corpus, texts)
    assert any("test pool" in e for e in errors)

    bad_quote = EvalItem.model_validate(
        _item(relevant=[{"doc_path": dev_path, "quote": "The kubelet schedules Pods", "grade": 3}])
    )
    assert any("quote not found" in e for e in check_items({"dev": [bad_quote]}, corpus, texts))

    test_item = EvalItem.model_validate(
        _item(
            id="test-001",
            split="test",
            relevant=[{"doc_path": test_path, "quote": "The scheduler assigns Pods", "grade": 3}],
        )
    )
    errors = check_items({"dev": [good_dev], "test": [test_item]}, corpus, texts)
    assert any("spans splits" in e for e in errors)


def test_check_targets_reports_size_gaps() -> None:
    assert any("dev: 0 items" in e for e in check_targets({}))


def test_extract_title_prefers_front_matter() -> None:
    assert extract_title("---\ntitle: Pods\n---\n# Other\n", "f") == "Pods"
    assert extract_title("# Heading\nbody", "f") == "Heading"
    assert extract_title("body", "fallback") == "fallback"
