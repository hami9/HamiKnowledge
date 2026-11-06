# HamiKnowledge

> Multi-user RAG knowledge platform for technical engineering teams querying English, Persian, and mixed-language technical documentation, with structurally verifiable source citations, calibrated abstention, and per-query budget accounting.

Status: **v1.0.0 — All Milestones (A0–A5) Passed.**  
Repository: [https://github.com/hami9/HamiKnowledge](https://github.com/hami9/HamiKnowledge)

---

## Architecture Overview

```mermaid
flowchart TD
    User["User / Client (RTL & LTR)"] -->|HTTP / Web Dashboard| API["FastAPI Gateway (/v1)"]
    
    subgraph Security & Financial Boundaries
        Auth["Workspace & Document ACL Check"]
        Ledger["Atomic Budget Ledger (PostgreSQL RLock)"]
        Cache["Isolated Query Cache (Tenant & Version Scoped)"]
    end
    
    API --> Auth
    Auth --> Ledger
    Ledger --> Cache
    
    subgraph Data Lifecycle & Ingestion
        Ingest["Ingestion Service (10MB Limit, MIME, UTF-8)"]
        Chunker["Heading-Aware Chunker (Offset Preservation)"]
        DenseEmbed["Pinned Multilingual Embeddings (intfloat/multilingual-e5-small)"]
        Store["PostgreSQL Storage (pgvector + Native FTS)"]
    end
    
    API --> Ingest
    Ingest --> Chunker
    Chunker --> DenseEmbed
    DenseEmbed --> Store
    
    subgraph Retrieval & Synthesis Pipeline
        Lex["B0: Lexical FTS"]
        Vec["B1: Dense Vector Cosine"]
        Fuser["B2: Reciprocal Rank Fusion (Default)"]
        Rerank["B3: Cross-Encoder Reranker"]
        Generator["Deterministic & Grounded Generator"]
        Citations["Structural Citation Validator & Abstention"]
    end
    
    Cache -->|Cache Miss| Lex & Vec
    Store --> Lex & Vec
    Lex & Vec --> Fuser
    Fuser --> Rerank
    Rerank --> Generator
    Generator --> Citations
    Citations -->|Reconcile Spend| Ledger
    Citations -->|Cited Answer + Usage + Excerpts| User

    subgraph Bounded Read-Only Agent (R4)
        Agent["Bounded Agent Loop"]
        T1["search_documents (Server ACL Enforced)"]
        T2["get_document_excerpt (Server ACL Enforced)"]
    end
    API -.->|Agent Mode| Agent
    Agent --> T1 & T2
```

---

## Key Features & Invariants

- **Multi-Tenant Isolation:** Workspace boundaries and document ACLs are enforced server-side inside every SQL predicate and vector filter. The model never enforces tenant permissions.
- **Traceable Source Citations:** Every claim links to an exact character offset (`start_char` to `end_char`) and heading path. Cited chunk IDs are structurally validated against retrieved context before response delivery.
- **Calibrated Abstention:** When evidence is insufficient or contradictory, the system declines to hallucinate and provides an explicit abstention with unverified passages.
- **Untrusted Content Barrier:** Retrieved document text is treated purely as data, never instruction. Injected prompt instructions, shell commands, or malicious URLs inside documents are ignored.
- **Bounded Read-Only Agent:** Permits exactly two safe tools (`search_documents` and `get_document_excerpt`), with a hard ceiling of 3 calls, a 20-second deadline, per-request budget reservations, and loop termination. Zero shell execution, zero arbitrary URLs, zero write tools.
- **Atomic Budget Ledger:** Thread-safe ledger that atomically reserves conservative cost ceilings before external execution and reconciles actual token usage afterward, preventing overages even under concurrent load.
- **RTL-Correct UI:** Native right-to-left layout for Persian documentation and mixed Latin technical terms using logical CSS properties.

---

## Empirical Benchmark & Quality Results

### 1. Frozen Test Set Comparison (100 Items, 6,550 Chunks)

Evaluated against the frozen 100-item test split (70 answerable, 30 unanswerable; 30 English, 30 Persian, 40 mixed-language) on the Kubernetes v1 corpus (601 documents, 6,550 chunks):

| Variant | Strategy | Recall@10 | nDCG@10 | MRR | Cit. Validity | p95 Latency | Avg Cost/Query |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Pure Lexical (PostgreSQL FTS) | 0.3857 | 0.2852 | 0.2555 | **100.0%** | 478.62 ms | $0.000167 |
| **B1** | Pure Vector (Dense Cosine) | 0.0929 | 0.0651 | 0.0571 | **100.0%** | 677.66 ms | $0.000194 |
| **B2** | Hybrid RRF (Lexical + Vector) | 0.3500 | 0.2338 | 0.1893 | **100.0%** | 1093.24 ms | $0.000194 |
| **B3** | Hybrid RRF + Cross-Encoder | 0.3500 | **0.3028** | **0.2871** | **100.0%** | 1127.95 ms | $0.000215 |

- **Default Selection:** **B2 (Hybrid RRF)** is the recommended baseline balance of recall and predictable latency.
- **High-Precision Option:** **B3 (Hybrid RRF + Reranker)** provides a +0.069 nDCG@10 uplift for applications prioritizing ranking precision over 34 ms of latency.

### 2. Adversarial & Lifecycle Benchmark (40 Items)

Evaluated with `python -m evals.adversarial` across 5 threat categories:

| Category | Items | Passed | Pass Rate | Key Invariant Enforced |
| :--- | :---: | :---: | :---: | :--- |
| **Prompt Injection** | 12 | 12 | **100.0%** | Injections ignored; zero prompt leaks or fake tokens |
| **Misleading Citations** | 8 | 8 | **100.0%** | False fixture claims rejected; authentic docs cited |
| **Deleted Content** | 8 | 8 | **100.0%** | Immediate exclusion; queries abstain with 0 citations |
| **ACL Changes** | 6 | 6 | **100.0%** | Server-side boundaries enforced; restricted content excluded |
| **Tool Misuse** | 6 | 6 | **100.0%** | Shell/network requests blocked; max 3 tool calls enforced |
| **Total** | **40** | **40** | **100.0%** | Full report: `evals/reports/adversarial-report.json` |

---

## Quickstart

### Prerequisites

- Python 3.12+ (or Docker 24+)
- 4 vCPUs, 8 GB RAM minimum (measured on Intel Core i5-6400, 16 GB RAM)

### 1. Local Development Setup

```powershell
# Clone repository
git clone https://github.com/hami9/HamiKnowledge.git
cd HamiKnowledge

# Create virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[models,dev]"

# Fetch and verify evaluation corpus
.\.venv\Scripts\python.exe -m evals.corpus fetch evals/datasets/corpus/k8s-docs-v1.yaml
.\.venv\Scripts\python.exe -m evals.corpus verify evals/datasets/corpus/k8s-docs-v1.yaml

# Run test suite
.\.venv\Scripts\pytest -q
```

### 2. Start API Server & Web Dashboard

```powershell
.\.venv\Scripts\uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at [http://localhost:8000/](http://localhost:8000/) to access the interactive RTL dashboard.

### 3. Docker Compose Deployment

```powershell
cd deploy
docker compose up -d
```

Starts PostgreSQL 16 (pgvector), Redis 7, MinIO S3, and the HamiKnowledge API on port 8000.

---

## Verification & Quality Gates

Run all automated quality gates:

```powershell
# Lint check
.\.venv\Scripts\ruff check .

# Strict type check
.\.venv\Scripts\mypy evals ingestion retrieval apps db agent tests

# Full unit and integration test suite (61 tests)
.\.venv\Scripts\pytest

# Dataset split validation
.\.venv\Scripts\python -m evals.validate check --strict

# Automated CI regression gate
.\.venv\Scripts\python -m evals.regression

# Full 40-item adversarial benchmark
.\.venv\Scripts\python -m evals.adversarial
```

---

## Repository Structure

```text
HamiKnowledge/
├── apps/
│   ├── api/                 # FastAPI application, routers (query, agent, documents, budget)
│   └── web/                 # RTL-correct interactive web dashboard (index.html)
├── agent/                   # Bounded read-only agent loop, tool limits, trace audit
├── db/                      # In-memory store, budget ledger, multi-tenant query cache
├── deploy/                  # Docker Compose, Dockerfiles, HamiCloud application descriptor
├── docs/
│   ├── evidence/            # Milestone evidence records (A0, A1, A2, A3, A4, A5)
│   ├── evaluation-plan.md   # Evaluation methodology and metrics definition
│   ├── evaluation-rubric.md # Groundedness and citation scoring rubrics
│   └── model-selection.md   # Model benchmarks and licensing documentation
├── evals/
│   ├── datasets/            # dev.jsonl, test.jsonl, adversarial.jsonl, corpus lock
│   ├── reports/             # ablation-report.json, release-report.json, adversarial-report.json
│   ├── ablation.py          # B0-B3 ablation runner
│   ├── adversarial.py       # 40-item adversarial test harness
│   └── regression.py        # Automated CI regression gate
├── ingestion/               # Heading-aware chunker, text normalizer, lifecycle service
├── retrieval/               # Lexical FTS, dense vector, RRF fusion, reranker, generation
└── tests/                   # 61 comprehensive unit, integration, concurrency & safety tests
```

---

## Documentation & Evidence Records

- [Evidence Record A0 — Defined Evidence](docs/evidence/A0.md)
- [Evidence Record A1 — Searchable Workspace](docs/evidence/A1.md)
- [Evidence Record A2 — RAG MVP Candidate](docs/evidence/A2.md)
- [Evidence Record A3 — Evaluated MVP](docs/evidence/A3.md)
- [Evidence Record A4 — Multi-User v1 Candidate](docs/evidence/A4.md)
- [Evidence Record A5 — HamiKnowledge v1.0 Release](docs/evidence/A5.md)
- [Engineering Roadmap](HamiKnowledge-Roadmap.md)
- [HamiCloud Platform Runtime](https://github.com/hami9/hamicloud)

---

## Attribution & License

- **Evaluation Corpus:** Kubernetes documentation, (c) The Kubernetes Authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source text is not redistributed in this repository.
- **License:** MIT License. See [LICENSE](LICENSE) for details.
