# Model and License Selection (A0)

## Objectives & Constraints

HamiKnowledge requires multilingual support (English, Persian, and mixed-language technical terms). Under the project requirements:
1. **License**: Commercially permissible without non-commercial riders (e.g. MIT, Apache-2.0).
2. **Resource Feasibility**: Must execute on CPU-based target environments (Intel i5 quad-core, 16 GB RAM) within latency targets (retrieval p95 <= 1s, reranking p95 <= 2s).
3. **Reproducibility**: Model revisions are pinned by full git commit SHA. Silent replacement is forbidden.

## Candidate Evaluation

### Embedding Models

| Candidate | Revision (Pinned) | License | Params | Dim | Load (s) | Peak RSS | CPU Throughput |
|---|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | MIT | 117M | 384 | 44.2s | 1.29 GB | 3.66 pass/s |
| `intfloat/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | MIT | 278M | 768 | 98.6s | 2.13 GB | 1.25 pass/s |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | MIT | 567M | 1024 | 205.5s | 1.91 GB | 0.45 pass/s |

**Selection for v1 baseline**:
- **Default Baseline**: `intfloat/multilingual-e5-small`. At 3.66 passages/sec on 4 CPU threads, full demo corpus indexing (3,059 chunks) takes ~14 minutes.
- **Comparison / High-Capacity**: `intfloat/multilingual-e5-base` (768 dim) retained for evaluation in A2/A3 ablations.

### Reranker Models

| Candidate | Revision (Pinned) | License | Params | Status |
|---|---|---|---|---|
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | Apache-2.0 | 117M | Selected candidate for RERANK (20 candidates) |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Apache-2.0 | 568M | Secondary evaluation candidate (CPU latency risk) |
| `jinaai/jina-reranker-v2-base-multilingual` | `9cfeff2df7d40d1b78e75e5e9cebec92a99813c9` | CC-BY-NC-4.0 | 278M | **REJECTED**: Non-commercial license prohibited for production deployment |

## Pinning Policy

Changing model revision or vector dimensions requires a recorded ADR and a versioned reindex. Vector spaces must never be mixed.
