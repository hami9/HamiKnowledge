"""Validate evaluation datasets and manage the frozen test manifest.

Usage:
    python -m evals.validate check            # schema, quotes, splits, frozen hashes
    python -m evals.validate check --strict   # also enforce target split sizes
    python -m evals.validate freeze           # write frozen-manifest.json
    python -m evals.validate pools --out .cache/pools   # doc lists per split pool
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from evals.corpus import CorpusDocument, load_lock, read_document
from evals.schema import (
    SPLIT_FILES,
    EvalItem,
    assigned_pool,
    load_items,
    normalize_for_match,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"
CORPUS_MANIFEST = DATASETS_DIR / "corpus" / "k8s-docs-v1.yaml"
FROZEN_MANIFEST = DATASETS_DIR / "frozen-manifest.json"
FROZEN_SPLITS = ("test", "adversarial")
RUBRIC_FILE = REPO_ROOT / "docs" / "evaluation-rubric.md"

TARGETS: dict[str, int] = {"dev": 60, "test": 100, "adversarial": 40}
TEST_ANSWERABLE = 70
TEST_UNANSWERABLE = 30


class FrozenManifest(BaseModel):
    frozen_at: dt.date
    corpus_lock_sha256: str
    rubric_sha256: str
    files: dict[str, str]
    counts: dict[str, int]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_index(manifest_path: Path) -> dict[str, CorpusDocument]:
    return {d.path: d for d in load_lock(manifest_path)}


def check_items(
    items_by_split: dict[str, list[EvalItem]],
    corpus: dict[str, CorpusDocument],
    read_text: dict[str, str] | None = None,
    manifest_path: Path = CORPUS_MANIFEST,
) -> list[str]:
    errors: list[str] = []
    texts: dict[str, str] = dict(read_text or {})
    ids: Counter[str] = Counter()
    family_splits: dict[str, set[str]] = {}
    doc_splits: dict[str, set[str]] = {}

    for split, items in items_by_split.items():
        for item in items:
            ids[item.id] += 1
            if item.split != split:
                errors.append(f"{item.id}: stored in {split} file but split={item.split}")
            family_splits.setdefault(item.family_id, set()).add(split)
            for rel in item.relevant:
                doc = corpus.get(rel.doc_path)
                if doc is None:
                    errors.append(f"{item.id}: unknown doc_path {rel.doc_path}")
                    continue
                pool = assigned_pool(rel.doc_path)
                if split == "dev" and pool != "dev":
                    errors.append(f"{item.id}: {rel.doc_path} is in the test pool")
                if split in FROZEN_SPLITS and pool != "test":
                    errors.append(f"{item.id}: {rel.doc_path} is in the dev pool")
                if rel.grade >= 2:
                    doc_splits.setdefault(rel.doc_path, set()).add(split)
                if rel.doc_path not in texts:
                    texts[rel.doc_path] = normalize_for_match(read_document(manifest_path, doc))
                if normalize_for_match(rel.quote) not in texts[rel.doc_path]:
                    errors.append(f"{item.id}: quote not found verbatim in {rel.doc_path}")

    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    errors += [
        f"family {f} spans splits {sorted(s)}"
        for f, s in family_splits.items()
        if len(s) > 1
    ]
    errors += [
        f"source {p} used by dev and a frozen split"
        for p, s in doc_splits.items()
        if "dev" in s and s & set(FROZEN_SPLITS)
    ]
    return errors


def check_targets(items_by_split: dict[str, list[EvalItem]]) -> list[str]:
    errors = [
        f"{split}: {len(items_by_split.get(split, []))} items, target {target}"
        for split, target in TARGETS.items()
        if len(items_by_split.get(split, [])) != target
    ]
    test = items_by_split.get("test", [])
    answerable = sum(1 for i in test if i.answerable)
    if test and (answerable, len(test) - answerable) != (TEST_ANSWERABLE, TEST_UNANSWERABLE):
        errors.append(
            f"test: {answerable} answerable / {len(test) - answerable} unanswerable, "
            f"target {TEST_ANSWERABLE}/{TEST_UNANSWERABLE}"
        )
    return errors


def check_frozen(datasets_dir: Path = DATASETS_DIR) -> list[str]:
    manifest_file = datasets_dir / FROZEN_MANIFEST.name
    if not manifest_file.exists():
        return []
    frozen = FrozenManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    errors = []
    for name, digest in frozen.files.items():
        path = datasets_dir / name
        if not path.exists() or sha256_file(path) != digest:
            errors.append(f"frozen file changed or missing: {name}")
    lock = CORPUS_MANIFEST.with_name(f"{CORPUS_MANIFEST.stem}.lock.jsonl")
    if sha256_file(lock) != frozen.corpus_lock_sha256:
        errors.append("corpus lock changed after freeze")
    if RUBRIC_FILE.exists() and sha256_file(RUBRIC_FILE) != frozen.rubric_sha256:
        errors.append("rubric changed after freeze")
    return errors


def summarize(items_by_split: dict[str, list[EvalItem]]) -> str:
    lines = []
    for split, items in items_by_split.items():
        langs = Counter(i.language for i in items)
        cats = Counter(i.category for i in items)
        reviewed = sum(1 for i in items if i.review.status == "reviewed")
        answerable = sum(1 for i in items if i.answerable)
        lines.append(
            f"{split}: {len(items)} items, answerable={answerable}, reviewed={reviewed}, "
            f"lang={dict(langs)}, categories={dict(cats)}"
        )
    return "\n".join(lines)


def freeze(datasets_dir: Path = DATASETS_DIR) -> FrozenManifest:
    lock = CORPUS_MANIFEST.with_name(f"{CORPUS_MANIFEST.stem}.lock.jsonl")
    files = {SPLIT_FILES[s]: sha256_file(datasets_dir / SPLIT_FILES[s]) for s in FROZEN_SPLITS}
    counts = {s: len(load_items(datasets_dir / SPLIT_FILES[s])) for s in FROZEN_SPLITS}
    manifest = FrozenManifest(
        frozen_at=dt.date.today(),
        corpus_lock_sha256=sha256_file(lock),
        rubric_sha256=sha256_file(RUBRIC_FILE),
        files=files,
        counts=counts,
    )
    (datasets_dir / FROZEN_MANIFEST.name).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def write_pools(out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[CorpusDocument]] = {"dev": [], "test": []}
    for doc in load_lock(CORPUS_MANIFEST):
        pools[assigned_pool(doc.path)].append(doc)
    for name, docs in pools.items():
        with (out_dir / f"{name}-pool.tsv").open("w", encoding="utf-8", newline="\n") as fh:
            for d in docs:
                fh.write(f"{d.path}\t{d.lang}\t{d.bytes}\t{d.title}\n")
    return {k: len(v) for k, v in pools.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.validate")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--strict", action="store_true")
    sub.add_parser("freeze")
    pools = sub.add_parser("pools")
    pools.add_argument("--out", type=Path, default=REPO_ROOT / ".cache" / "pools")
    args = parser.parse_args(argv)

    if args.command == "pools":
        print(write_pools(args.out))
        return 0

    try:
        items = {s: load_items(DATASETS_DIR / f) for s, f in SPLIT_FILES.items()}
    except ValueError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 1
    errors = check_items(items, _corpus_index(CORPUS_MANIFEST))
    if args.command == "check":
        errors += check_frozen()
        if args.strict:
            errors += check_targets(items)
        print(summarize(items))
        for err in errors:
            print(f"ERROR {err}", file=sys.stderr)
        print("OK" if not errors else f"{len(errors)} error(s)")
        return 1 if errors else 0

    errors += check_targets(items)
    if errors:
        for err in errors:
            print(f"ERROR {err}", file=sys.stderr)
        print("refusing to freeze")
        return 1
    print(freeze().model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
