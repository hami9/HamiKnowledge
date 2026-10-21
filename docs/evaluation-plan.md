# Evaluation plan (A0)

## Task boundary

HamiKnowledge answers English, Persian and mixed-language questions about engineering documentation: deployment, configuration, operations and troubleshooting. In scope: fact lookup, multi-passage synthesis, exact technical terms and ambiguous questions. Out of scope: questions that need images, tables rendered from code, live cluster state, or content outside the corpus.

## Corpus

`evals/datasets/corpus/k8s-docs-v1.yaml` pins the Kubernetes documentation (CC BY 4.0) at one upstream commit:

- English: `concepts`, `tasks`, `tutorials`, `setup`.
- Persian: all of `content/fa/docs` (mostly glossary, setup and tutorials).

Source text is not committed. `python -m evals.corpus fetch|build|verify` recreates the checkout and checks every document hash against `k8s-docs-v1.lock.jsonl`. Attribution is required in any published demo.

## Splits

| Split | Target | File | Use |
| --- | --- | --- | --- |
| Development | 60 | `evals/datasets/dev.jsonl` | Tune chunking, fusion, candidate counts, prompts, abstention threshold, model selection |
| Frozen test | 100 (70 answerable, 30 unanswerable) | `evals/datasets/test.jsonl` | Final comparison and release checks only |
| Adversarial/lifecycle | 40 | `evals/datasets/adversarial.jsonl` | Injection, misleading citations, deletion, ACL changes, tool misuse |

Language target for the frozen test set: about 30 English, 30 Persian, 40 mixed. The actual distribution is reported by `python -m evals.validate check`.

## Leakage controls

1. **Source pools.** Every corpus document is assigned to a dev or test pool by `sha256(pool_key(path)) % 100 < 40`. A Persian translation shares the pool of its English source. Dev items cite only dev-pool documents; test and adversarial items cite only test-pool documents. The corpus itself stays fully indexed.
2. **Families.** Near-duplicate questions share a `family_id`; a family may not span splits.
3. **Not indexed.** The corpus manifest only includes upstream documentation paths. Dataset files live under `evals/datasets/` and are never ingested.
4. **Not tuned.** Experiment code loads data through `evals.schema.load_split`, which raises `SplitAccessError` for `test`/`adversarial` with `purpose="tuning"`.
5. **Frozen hashes.** `python -m evals.validate freeze` records hashes of the test and adversarial files, the corpus lock and the rubric. `check` fails if any of them change afterwards.

## Labels

Relevance labels are verbatim quotes plus a grade (see `docs/evaluation-rubric.md`). They do not depend on chunk IDs, so chunking can change without relabeling. The validator fails if a quote is not found in the pinned document.

Drafts may be machine-assisted. They enter results only after manual review (`review.status = reviewed`).

## Metrics

Recall@10 and nDCG@10 per language, citation validity, citation support precision, grounded answer pass rate, appropriate and false abstention, latency and cost. Targets are listed in the roadmap; they are targets, not results. Retrieval deltas are reported with paired bootstrap confidence intervals; small pass/fail samples are reported as raw counts.
