# hamiknowledge

Multi-user RAG workspace for English and Persian technical documents, with checkable citations and visible cost.

Status: **A0 — evaluation baseline passed.**

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[models,dev]" psutil
```

## Usage

Fetch and verify the pinned evaluation corpus (Kubernetes docs, CC BY 4.0):

```powershell
.\.venv\Scripts\python.exe -m evals.corpus fetch evals/datasets/corpus/k8s-docs-v1.yaml
.\.venv\Scripts\python.exe -m evals.corpus verify evals/datasets/corpus/k8s-docs-v1.yaml
```

Validate the datasets and the frozen manifest:

```powershell
.\.venv\Scripts\python.exe -m evals.validate check --strict
```

Run the CPU feasibility and model selection benchmark on the dev split:

```powershell
.\.venv\Scripts\python.exe -m evals.feasibility --embeddings all --rerank-with intfloat/multilingual-e5-small
```

## Documentation

- [Evaluation plan](docs/evaluation-plan.md)
- [Evaluation rubric](docs/evaluation-rubric.md)
- [Model selection](docs/model-selection.md)
- [Evidence record A0](docs/evidence/A0.md)
- [Engineering roadmap](HamiKnowledge-Roadmap.md)
- [Platform runtime (HamiCloud)](https://github.com/hami9/hamicloud)

## Attribution

Evaluation corpus: Kubernetes documentation, (c) The Kubernetes Authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source text is not redistributed in this repository.

## License

MIT
