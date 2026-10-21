"""Build and verify the pinned evaluation corpus.

The corpus text is never committed. A manifest (YAML) pins the upstream commit and
include paths; ``build`` writes a lock file with one content hash per document so
any later run can prove it used the same bytes.

Usage:
    python -m evals.corpus fetch evals/datasets/corpus/k8s-docs-v1.yaml
    python -m evals.corpus build evals/datasets/corpus/k8s-docs-v1.yaml
    python -m evals.corpus verify evals/datasets/corpus/k8s-docs-v1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


class SourceInfo(BaseModel):
    name: str
    repository: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    license_url: str
    attribution: str


class IncludeRule(BaseModel):
    lang: Literal["en", "fa"]
    paths: list[str] = Field(min_length=1)


class CorpusManifest(BaseModel):
    corpus_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    source: SourceInfo
    checkout_dir: str
    include: list[IncludeRule] = Field(min_length=1)
    min_bytes: int = Field(default=0, ge=0)


class CorpusDocument(BaseModel):
    doc_id: str
    path: str
    lang: Literal["en", "fa"]
    title: str
    bytes: int
    sha256: str


def load_manifest(path: Path) -> CorpusManifest:
    with path.open(encoding="utf-8") as fh:
        return CorpusManifest.model_validate(yaml.safe_load(fh))


def lock_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.lock.jsonl")


def checkout_root(manifest: CorpusManifest) -> Path:
    return (REPO_ROOT / manifest.checkout_dir).resolve()


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def fetch(manifest: CorpusManifest) -> Path:
    """Create or update a sparse, blobless checkout at the pinned commit."""
    root = checkout_root(manifest)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--",
             manifest.source.repository, str(root)],
            check=True,
        )
    patterns = ["/LICENSE"] + [f"/{p.strip('/')}/" for r in manifest.include for p in r.paths]
    _git(["sparse-checkout", "set", "--no-cone", *patterns], root)
    commit = manifest.source.commit
    try:
        _git(["cat-file", "-e", f"{commit}^{{commit}}"], root)
    except subprocess.CalledProcessError:
        _git(["fetch", "--filter=blob:none", "origin", commit], root)
    _git(["checkout", "--detach", commit], root)
    return root


def _verify_head(manifest: CorpusManifest, root: Path) -> None:
    if not (root / ".git").exists():
        raise RuntimeError(f"checkout missing at {root}; run `fetch` first")
    head = _git(["rev-parse", "HEAD"], root)
    if head != manifest.source.commit:
        raise RuntimeError(
            f"checkout HEAD {head} does not match pinned commit {manifest.source.commit}"
        )


def extract_title(text: str, fallback: str) -> str:
    match = _FRONT_MATTER.match(text)
    if match:
        try:
            meta = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            meta = None
        if isinstance(meta, dict):
            title = meta.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def iter_documents(manifest: CorpusManifest, root: Path) -> Iterator[CorpusDocument]:
    seen: set[str] = set()
    for rule in manifest.include:
        for rel in rule.paths:
            base = (root / rel).resolve()
            if not base.is_relative_to(root):
                raise ValueError(f"include path escapes checkout: {rel}")
            if not base.is_dir():
                raise FileNotFoundError(f"include path not found: {rel}")
            for file in sorted(base.rglob("*.md")):
                data = file.read_bytes()
                if len(data) < manifest.min_bytes:
                    continue
                relpath = file.relative_to(root).as_posix()
                if relpath in seen:
                    continue
                seen.add(relpath)
                text = data.decode("utf-8")
                yield CorpusDocument(
                    doc_id=f"{manifest.corpus_id}:{relpath}",
                    path=relpath,
                    lang=rule.lang,
                    title=extract_title(text, fallback=file.stem),
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )


def build(manifest_path: Path) -> list[CorpusDocument]:
    manifest = load_manifest(manifest_path)
    root = checkout_root(manifest)
    _verify_head(manifest, root)
    docs = sorted(iter_documents(manifest, root), key=lambda d: d.path)
    with lock_path(manifest_path).open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.model_dump(), ensure_ascii=False) + "\n")
    return docs


def load_lock(manifest_path: Path) -> list[CorpusDocument]:
    with lock_path(manifest_path).open(encoding="utf-8") as fh:
        return [CorpusDocument.model_validate_json(line) for line in fh if line.strip()]


def verify(manifest_path: Path) -> list[str]:
    """Return a list of mismatches between the lock file and the checkout."""
    manifest = load_manifest(manifest_path)
    root = checkout_root(manifest)
    _verify_head(manifest, root)
    locked = {d.path: d for d in load_lock(manifest_path)}
    current = {d.path: d for d in iter_documents(manifest, root)}
    problems = [f"missing from checkout: {p}" for p in sorted(locked.keys() - current.keys())]
    problems += [f"not in lock: {p}" for p in sorted(current.keys() - locked.keys())]
    problems += [
        f"hash mismatch: {p}"
        for p in sorted(locked.keys() & current.keys())
        if locked[p].sha256 != current[p].sha256
    ]
    return problems


def read_document(manifest_path: Path, doc: CorpusDocument) -> str:
    """Read a locked document and fail if its bytes changed."""
    manifest = load_manifest(manifest_path)
    data = (checkout_root(manifest) / doc.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != doc.sha256:
        raise RuntimeError(f"hash mismatch for {doc.path}")
    return data.decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.corpus")
    parser.add_argument("command", choices=["fetch", "build", "verify"])
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    manifest_path: Path = args.manifest

    if args.command == "fetch":
        root = fetch(load_manifest(manifest_path))
        print(f"checkout ready: {root}")
        return 0
    if args.command == "build":
        docs = build(manifest_path)
        by_lang: dict[str, int] = {}
        for d in docs:
            by_lang[d.lang] = by_lang.get(d.lang, 0) + 1
        total = sum(d.bytes for d in docs)
        print(f"locked {len(docs)} documents ({by_lang}), {total} bytes")
        return 0
    problems = verify(manifest_path)
    for problem in problems:
        print(problem, file=sys.stderr)
    print("corpus verified" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
