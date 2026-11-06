"""Ablation benchmark runner comparing B0, B1, B2, and B3 retrieval variants.

Implements all Milestone A3 evaluation contracts:
- Variant B0: Lexical baseline (PostgreSQL FTS reference)
- Variant B1: Vector baseline (Dense cosine similarity)
- Variant B2: Hybrid RRF (Lexical + Vector combined via Reciprocal Rank Fusion)
- Variant B3: Hybrid RRF + Cross-Encoder Reranking

Usage:
    python -m evals.ablation --split dev
    python -m evals.ablation --split test --purpose final
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from evals.corpus import checkout_root, load_lock, load_manifest, read_document
from evals.metrics import (
    calculate_percentile,
    compute_abstention_rates,
    extract_relevance_spans,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from evals.schema import load_split
from ingestion.chunker import HeadingAwareChunker
from retrieval.embeddings import DeterministicEmbeddingAdapter, EmbeddingModel
from retrieval.fusion import ReciprocalRankFuser
from retrieval.generation import DeterministicGenerationAdapter, GeneratedAnswer
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.pipeline import RetrievalPipeline, RetrievalVariant
from retrieval.protocols import Chunk, RetrievalFilter
from retrieval.reranker import DeterministicRerankerAdapter
from retrieval.vector import InMemoryVectorRetriever

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"
CORPUS_MANIFEST_PATH = DATASETS_DIR / "corpus" / "k8s-docs-v1.yaml"
REPORTS_DIR = REPO_ROOT / "evals" / "reports"


@dataclass
class VariantMetrics:
    variant: str
    recall_at_10: float
    ndcg_at_10: float
    mrr: float
    citation_validity: float
    true_abstention_rate: float
    false_abstention_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    avg_tokens_per_query: float
    avg_cost_per_query_usd: float
    total_cost_usd: float


@dataclass
class AblationReport:
    timestamp: str
    split: str
    purpose: str
    total_items: int
    answerable_items: int
    unanswerable_items: int
    corpus_document_count: int
    corpus_chunk_count: int
    variants: dict[str, VariantMetrics]
    chosen_default: str
    justification: str


def load_and_chunk_corpus(
    chunker: HeadingAwareChunker,
    workspace_id: UUID,
) -> tuple[dict[str, str], list[Chunk], dict[str, str]]:
    """Load corpus documents from lock file and chunk them with production chunker."""
    manifest = load_manifest(CORPUS_MANIFEST_PATH)
    lock_items = load_lock(CORPUS_MANIFEST_PATH)

    root = checkout_root(manifest)
    if not (root / ".git").exists():
        raise RuntimeError("Corpus checkout missing; run `python -m evals.corpus fetch` first")

    document_texts: dict[str, str] = {}
    all_chunks: list[Chunk] = []
    chunk_to_doc_path: dict[str, str] = {}

    for doc in lock_items:
        raw_text = read_document(CORPUS_MANIFEST_PATH, doc)
        document_texts[doc.path] = raw_text

        doc_chunks = chunker.chunk(
            text=raw_text,
            document_id=uuid4(),
            workspace_id=workspace_id,
            source_version="v1",
            language=doc.lang,
        )
        for c in doc_chunks:
            all_chunks.append(c)
            chunk_to_doc_path[c.chunk_id] = doc.path

    return document_texts, all_chunks, chunk_to_doc_path


def run_evaluation(
    split: Literal["dev", "test"] = "dev",
    purpose: Literal["tuning", "final"] = "tuning",
    variants_to_run: list[RetrievalVariant] | None = None,
    embedder: EmbeddingModel | None = None,
) -> AblationReport:
    """Run full B0-B3 evaluation matrix on the specified split."""
    if variants_to_run is None:
        variants_to_run = ["B0", "B1", "B2", "B3"]

    embedder = embedder or DeterministicEmbeddingAdapter()
    eval_workspace_id = uuid4()
    chunker = HeadingAwareChunker()

    print(f"Loading and chunking corpus from {CORPUS_MANIFEST_PATH}...")
    document_texts, chunks, chunk_to_doc_path = load_and_chunk_corpus(chunker, eval_workspace_id)
    print(f"Loaded {len(document_texts)} documents, produced {len(chunks)} production chunks.")

    print("Embedding chunks with pinned model adapter...")
    chunk_embeddings = {
        c.chunk_id: embedder.embed_query(c.search_text)
        for c in chunks
    }

    print(f"Loading {split} split items (purpose={purpose})...")
    items = load_split(DATASETS_DIR, split, purpose)
    answerable_items = [it for it in items if it.answerable]
    unanswerable_items = [it for it in items if not it.answerable]

    # Initialize retrievers
    lexical_retriever = InMemoryLexicalRetriever(chunks)
    vector_retriever = InMemoryVectorRetriever(chunks, chunk_embeddings)
    reranker = DeterministicRerankerAdapter()
    generator = DeterministicGenerationAdapter()

    pipeline = RetrievalPipeline(
        lexical_retriever=lexical_retriever,
        vector_retriever=vector_retriever,
        embedder=embedder,
        fuser=ReciprocalRankFuser(),
        reranker=reranker,
    )

    retrieval_filter = RetrievalFilter(workspace_id=eval_workspace_id)
    variant_results: dict[str, VariantMetrics] = {}

    for var in variants_to_run:
        print(f"\nEvaluating Variant {var} on {len(items)} items...")
        recalls: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        latencies: list[float] = []
        answers: list[GeneratedAnswer] = []
        valid_citations_count = 0
        total_citations_count = 0

        for item in items:
            t0 = time.perf_counter()
            pipe_res = pipeline.execute(
                query=item.question,
                retrieval_filter=retrieval_filter,
                variant=var,
                limit=10,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            # Generate answer
            ans = generator.generate(
                query=item.question,
                candidates=pipe_res.candidates,
                workspace_id=eval_workspace_id,
                variant=var,
            )
            answers.append(ans)

            # Structural citation validity
            for cit in ans.citations:
                total_citations_count += 1
                if any(c.chunk_id == cit.chunk_id for c in pipe_res.candidates):
                    valid_citations_count += 1

            # Quality metrics on answerable items
            if item.answerable:
                spans = extract_relevance_spans(item, document_texts)
                rec = recall_at_k(pipe_res.candidates, chunk_to_doc_path, spans, k=10)
                nd = ndcg_at_k(pipe_res.candidates, chunk_to_doc_path, spans, k=10)
                mr = mrr_at_k(pipe_res.candidates, chunk_to_doc_path, spans, k=10)
                recalls.append(rec)
                ndcgs.append(nd)
                mrrs.append(mr)

        avg_rec = sum(recalls) / len(recalls) if recalls else 0.0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
        cit_validity = (
            (valid_citations_count / total_citations_count)
            if total_citations_count > 0
            else 1.0
        )

        abst_rates = compute_abstention_rates(items, answers)
        p50 = calculate_percentile(latencies, 50)
        p95 = calculate_percentile(latencies, 95)

        total_tokens = sum(a.usage.total_tokens for a in answers)
        total_cost = sum(a.usage.estimated_cost_usd for a in answers)

        metrics = VariantMetrics(
            variant=var,
            recall_at_10=round(avg_rec, 4),
            ndcg_at_10=round(avg_ndcg, 4),
            mrr=round(avg_mrr, 4),
            citation_validity=round(cit_validity, 4),
            true_abstention_rate=abst_rates["true_abstention_rate"],
            false_abstention_rate=abst_rates["false_abstention_rate"],
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            avg_tokens_per_query=round(total_tokens / len(items), 1),
            avg_cost_per_query_usd=round(total_cost / len(items), 6),
            total_cost_usd=round(total_cost, 6),
        )
        variant_results[var] = metrics

    # Decide chosen default based on empirical evidence
    # Comparing B2 (hybrid) vs B3 (hybrid + reranker)
    b2 = variant_results.get("B2")
    b3 = variant_results.get("B3")

    if b3 and b2 and b3.ndcg_at_10 >= b2.ndcg_at_10 and b3.latency_p95_ms <= 1000.0:
        chosen = "B3"
        justification = (
            f"B3 (Hybrid RRF + Reranker) achieved highest quality "
            f"(nDCG@10: {b3.ndcg_at_10:.4f} vs B2: {b2.ndcg_at_10:.4f}) "
            f"with citation validity {b3.citation_validity * 100:.1f}%."
        )
    else:
        chosen = "B2"
        justification = (
            "B2 (Hybrid RRF) provides the optimal trade-off between retrieval recall and latency."
        )

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = AblationReport(
        timestamp=timestamp,
        split=split,
        purpose=purpose,
        total_items=len(items),
        answerable_items=len(answerable_items),
        unanswerable_items=len(unanswerable_items),
        corpus_document_count=len(document_texts),
        corpus_chunk_count=len(chunks),
        variants=variant_results,
        chosen_default=chosen,
        justification=justification,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_filename = "release-report.json" if split == "test" else "ablation-report.json"
    report_path = REPORTS_DIR / report_filename
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print(f"\nReport saved to {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HamiKnowledge B0-B3 ablation evaluation.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--purpose", choices=["tuning", "final"], default="tuning")
    args = parser.parse_args()

    report = run_evaluation(split=args.split, purpose=args.purpose)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"ABLATION REPORT SUMMARY ({report.split.upper()} SPLIT)")
    print("=" * 80)
    header = (
        f"{'Variant':<8} | {'Recall@10':<10} | {'nDCG@10':<10} | "
        f"{'MRR':<8} | {'TAR':<8} | {'FAR':<8} | {'p95 Lat (ms)':<12}"
    )
    print(header)
    print("-" * 80)
    for _v_name, m in report.variants.items():
        print(
            f"{m.variant:<8} | {m.recall_at_10:<10.4f} | {m.ndcg_at_10:<10.4f} | "
            f"{m.mrr:<8.4f} | {m.true_abstention_rate:<8.2f} | {m.false_abstention_rate:<8.2f} | "
            f"{m.latency_p95_ms:<12.2f}"
        )
    print("=" * 80)
    print(f"Chosen Default: {report.chosen_default}")
    print(f"Justification:  {report.justification}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
