"""R0 CPU feasibility and model selection benchmark (development split only).

Measures, per pinned candidate model, load time, resident memory, passage
throughput, single-query latency and dev-split retrieval quality with exact
dense search. Rerankers are measured on the top-20 candidates of one embedding
model. The chunker here is a fixed-size feasibility chunker, not the production
chunker.

Usage:
    python -m evals.feasibility --embeddings all --rerank-with intfloat/multilingual-e5-small
    python -m evals.feasibility --embeddings BAAI/bge-m3 --max-chunks 128   # throughput only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

from evals.corpus import load_lock, read_document
from evals.schema import EvalItem, load_split, normalize_for_match

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"
CORPUS_MANIFEST = DATASETS_DIR / "corpus" / "k8s-docs-v1.yaml"
MODELS_FILE = REPO_ROOT / "config" / "models.yaml"
CACHE_DIR = REPO_ROOT / ".cache" / "feasibility"
REPORTS_DIR = REPO_ROOT / "evals" / "reports"
_FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)

CHUNK_CHARS = 2000
OVERLAP_CHARS = 300
TOP_K = 10
RERANK_CANDIDATES = 20


@dataclass(frozen=True)
class Chunk:
    doc_path: str
    start: int
    end: int
    text: str


def chunk_text(doc_path: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + CHUNK_CHARS)
        if end < n:
            space = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            end = space if space > start else end
        chunks.append(Chunk(doc_path, start, end, text[start:end]))
        if end >= n:
            break
        nxt = end - OVERLAP_CHARS
        space = text.find(" ", nxt, end)
        start = space + 1 if space != -1 else nxt
    return chunks


def build_chunks() -> tuple[list[Chunk], dict[str, str]]:
    texts: dict[str, str] = {}
    chunks: list[Chunk] = []
    for doc in load_lock(CORPUS_MANIFEST):
        raw = read_document(CORPUS_MANIFEST, doc)
        text = normalize_for_match(_FRONT_MATTER.sub("", raw, count=1))
        texts[doc.path] = text
        chunks.extend(chunk_text(doc.path, text))
    return chunks, texts


def relevance_spans(
    item: EvalItem, texts: dict[str, str]
) -> list[tuple[str, int, int, int]]:
    spans = []
    for rel in item.relevant:
        quote = normalize_for_match(rel.quote)
        text = texts.get(rel.doc_path, "")
        pos = text.find(quote)
        if pos == -1:
            # The quote may sit inside stripped front matter; skip rather than mislabel.
            continue
        spans.append((rel.doc_path, pos, pos + len(quote), rel.grade))
    return spans


def chunk_gain(chunk: Chunk, spans: list[tuple[str, int, int, int]]) -> int:
    best = 0
    for path, start, end, grade in spans:
        if path == chunk.doc_path and chunk.start < end and start < chunk.end:
            best = max(best, grade)
    return best


def score_ranking(
    ranked: list[int], chunks: list[Chunk], spans: list[tuple[str, int, int, int]]
) -> tuple[float, float]:
    strong = [s for s in spans if s[3] >= 2]
    top = ranked[:TOP_K]
    found = sum(
        1
        for s in strong
        if any(chunk_gain(chunks[i], [s]) > 0 for i in top)
    )
    recall = found / len(strong) if strong else 0.0
    gains = [chunk_gain(chunks[i], spans) for i in top]
    dcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(gains))
    ideal_gains = sorted(
        (chunk_gain(c, spans) for c in chunks if chunk_gain(c, spans) > 0), reverse=True
    )[:TOP_K]
    idcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(ideal_gains))
    return recall, (dcg / idcg if idcg else 0.0)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


def rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss) / 2**20


def aggregate(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lang in ("all", "en", "fa", "mixed"):
        rows = [r for r in per_item if lang == "all" or r["language"] == lang]
        if rows:
            out[lang] = {
                "n": len(rows),
                "recall@10": round(statistics.fmean(r["recall"] for r in rows), 4),
                "ndcg@10": round(statistics.fmean(r["ndcg"] for r in rows), 4),
            }
    return out


def run_embedding(
    spec: dict[str, Any],
    chunks: list[Chunk],
    items: list[EvalItem],
    texts: dict[str, str],
    max_chunks: int | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    from sentence_transformers import SentenceTransformer

    base_rss = rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device="cpu")
    model.max_seq_length = min(model.max_seq_length or 512, 512)
    load_s = time.perf_counter() - t0
    loaded_rss = rss_mb()

    subset = chunks[:max_chunks] if max_chunks else chunks
    cache = CACHE_DIR / f"{spec['id'].replace('/', '__')}-{spec['revision'][:12]}-{len(subset)}.npy"
    passages = [spec["passage_prefix"] + c.text for c in subset]
    if cache.exists():
        matrix = np.load(cache)
        encode_s = None
    else:
        t0 = time.perf_counter()
        matrix = model.encode(
            passages, batch_size=16, normalize_embeddings=spec["normalize"],
            convert_to_numpy=True, show_progress_bar=False,
        )
        encode_s = time.perf_counter() - t0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, matrix)
    peak_rss = rss_mb()

    result: dict[str, Any] = {
        "model": spec["id"],
        "revision": spec["revision"],
        "dimension": int(matrix.shape[1]),
        "load_seconds": round(load_s, 2),
        "rss_model_mb": round(loaded_rss - base_rss, 1),
        "rss_peak_mb": round(peak_rss, 1),
        "chunks_encoded": len(subset),
        "encode_seconds": round(encode_s, 2) if encode_s is not None else "cached",
        "passages_per_second": round(len(subset) / encode_s, 2) if encode_s else "cached",
    }

    queries = [spec["query_prefix"] + i.question for i in items]
    latencies = []
    for q in queries[:30]:
        t0 = time.perf_counter()
        model.encode([q], normalize_embeddings=spec["normalize"], show_progress_bar=False)
        latencies.append((time.perf_counter() - t0) * 1000)
    result["query_latency_ms"] = {
        "p50": round(percentile(latencies, 50), 1),
        "p95": round(percentile(latencies, 95), 1),
        "n": len(latencies),
    }

    if max_chunks:
        return result, None

    qmat = model.encode(
        queries, batch_size=16, normalize_embeddings=spec["normalize"],
        convert_to_numpy=True, show_progress_bar=False,
    )
    scores = qmat @ matrix.T
    rankings = np.argsort(-scores, axis=1)[:, :RERANK_CANDIDATES]
    per_item = []
    for item, ranked in zip(items, rankings, strict=True):
        spans = relevance_spans(item, texts)
        recall, ndcg = score_ranking(ranked.tolist(), chunks, spans)
        per_item.append({"id": item.id, "language": item.language,
                         "recall": recall, "ndcg": ndcg})
    result["dev_quality"] = aggregate(per_item)
    del model
    return result, rankings


def run_reranker(
    spec: dict[str, Any],
    rankings: np.ndarray,
    chunks: list[Chunk],
    items: list[EvalItem],
    texts: dict[str, str],
) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder

    base_rss = rss_mb()
    t0 = time.perf_counter()
    model = CrossEncoder(spec["id"], revision=spec["revision"], device="cpu", max_length=512)
    load_s = time.perf_counter() - t0
    loaded_rss = rss_mb()
    latencies = []
    per_item = []
    for item, ranked in zip(items, rankings, strict=True):
        candidates = ranked.tolist()
        pairs = [(item.question, chunks[i].text) for i in candidates]
        t0 = time.perf_counter()
        scores = model.predict(pairs, batch_size=20, show_progress_bar=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        order = [candidates[i] for i in np.argsort(-np.asarray(scores))]
        recall, ndcg = score_ranking(order, chunks, relevance_spans(item, texts))
        per_item.append({"id": item.id, "language": item.language,
                         "recall": recall, "ndcg": ndcg})
    return {
        "model": spec["id"],
        "revision": spec["revision"],
        "load_seconds": round(load_s, 2),
        "rss_model_mb": round(loaded_rss - base_rss, 1),
        "rerank20_latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "n": len(latencies),
        },
        "dev_quality": aggregate(per_item),
    }


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(prog="python -m evals.feasibility")
    parser.add_argument("--embeddings", nargs="+", default=["all"])
    parser.add_argument("--rerank-with", default=None,
                        help="embedding model whose top-20 feeds the rerankers")
    parser.add_argument("--rerankers", nargs="+", default=["all"])
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="throughput-only run on the first N chunks")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    config = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    emb_specs = [s for s in config["embedding_candidates"]
                 if "all" in args.embeddings or s["id"] in args.embeddings]
    rr_specs = [s for s in config["reranker_candidates"]
                if "all" in args.rerankers or s["id"] in args.rerankers]

    chunks, texts = build_chunks()
    items = [i for i in load_split(DATASETS_DIR, "dev", "tuning") if i.answerable]
    if not items and not args.max_chunks:
        raise SystemExit("dev split has no answerable items")

    report: dict[str, Any] = {
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "split": "dev",
        "answerable_items": len(items),
        "environment": {
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": args.threads,
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
        },
        "corpus": {"manifest": CORPUS_MANIFEST.name, "documents": len(texts),
                   "chunks": len(chunks), "chunk_chars": CHUNK_CHARS,
                   "overlap_chars": OVERLAP_CHARS},
        "embeddings": [],
        "rerankers": [],
    }
    rankings_by_model: dict[str, np.ndarray] = {}
    for spec in emb_specs:
        print(f"embedding: {spec['id']}", flush=True)
        result, rankings = run_embedding(spec, chunks, items, texts, args.max_chunks)
        report["embeddings"].append(result)
        if rankings is not None:
            rankings_by_model[spec["id"]] = rankings
        print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.rerank_with:
        if args.rerank_with not in rankings_by_model:
            raise SystemExit(f"--rerank-with {args.rerank_with} was not run in full mode")
        report["rerank_input"] = args.rerank_with
        for spec in rr_specs:
            print(f"reranker: {spec['id']}", flush=True)
            result = run_reranker(spec, rankings_by_model[args.rerank_with], chunks, items, texts)
            report["rerankers"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or REPORTS_DIR / f"feasibility-{dt.date.today().isoformat()}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
