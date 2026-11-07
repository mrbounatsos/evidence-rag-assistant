# EvidenceRAG

<div align="center">

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)
![CI](https://github.com/mrbounatsos/evidence-rag-assistant/actions/workflows/ci.yml/badge.svg)
![mypy](https://img.shields.io/badge/types-mypy-2A6DB2)
![Ollama](https://img.shields.io/badge/LLM-Qwen3%208B%20via%20Ollama-000000?logo=ollama&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20development-orange)

</div>

EvidenceRAG is a local retrieval-augmented generation (RAG) system for answering questions about biomedical research papers. It retrieves relevant passages from a curated literature corpus, generates answers with source citations, and abstains when the retrieved evidence is insufficient.

The project covers the full pipeline, from acquiring and processing scientific papers to retrieval, answer generation, and benchmark evaluation. Inference runs locally using open-source models; no hosted LLM API is required.

## System overview

```mermaid
flowchart LR
    A[Europe PMC] --> B[Licence verification and ingestion]
    B --> C[Text extraction and chunking]
    C --> D[Embeddings and search index]
    D --> E[Passage retrieval]
    E --> F[Qwen3 via Ollama]
    F --> G[Cited answer or abstention]
```

| Component | Implementation |
| --- | --- |
| Data source | Europe PMC open-access biomedical articles |
| Ingestion | Article-level licence verification, full-text XML parsing, and file provenance tracking |
| Preprocessing | Section-aware text extraction and approximately 300-word chunks |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Retrieval | FAISS dense search; hybrid keyword and semantic retrieval evaluated |
| Generation | Qwen3 8B served locally through Ollama |
| Grounding | Passage-level source references, citation validation, and abstention handling |
| Evaluation | LitQA2 benchmark with retrieval and no-source controls |

The ingestion pipeline verifies article licences against the full-text XML rather than relying solely on search metadata. It excludes references and administrative sections from the searchable text and records provenance for the documents used to build the index. Article text and derived indexes are stored locally and are not committed to the repository.

During generation, retrieved passages are passed to the model as numbered sources. The system validates citation references in the response and distinguishes answers from abstentions and ungrounded outputs. A retrieval-score threshold handles clearly off-topic questions; the model is also instructed to abstain when topically relevant passages do not contain the requested information.

## Evaluation

The system was evaluated using the 38-question tuning split of **LitQA2**, part of the [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) benchmark for scientific literature question answering.

| Configuration | Accuracy | Precision |
| --- | ---: | ---: |
| Hybrid retrieval, 8 passages | **90%** | **96%** |
| Initial dense-retrieval baseline | 78% | — |
| Closed-book model, without retrieved sources | 16% | 36% |

Combining keyword and embedding-based retrieval and increasing the number of passages from five to eight improved measured accuracy from 78% to 90%. The rate of retrieving an answer-containing passage increased from 78% to 96%.

In a separate no-source experiment, the source paper was removed from the corpus. The system abstained on all **51 of 51** evaluated questions.

**Evaluation scope:** These experiments use a closed corpus of 138 papers containing the source article for each answerable benchmark question. Results therefore measure retrieval and answer generation within a known collection, not the ability to discover relevant papers across the entire biomedical literature. The sample is also small, so these figures should not be interpreted as general-purpose performance estimates.

Experimental setup, analysis, and limitations are documented in the [project report](docs/REPORT.md).

## Getting started

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with the `qwen3:8b` model
- A GPU is recommended for local inference

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mrbounatsos/evidence-rag-assistant.git
cd evidence-rag-assistant
uv sync --extra gpu
ollama pull qwen3:8b
```

### Prepare the corpus

Run the ingestion scripts to discover papers, resolve benchmark references, and fetch licence-verified full text:

```bash
uv run python scripts/discover_europe_pmc.py
uv run python scripts/download_litqa2.py
uv run python scripts/resolve_litqa2.py
uv run python scripts/fetch_fulltext.py
```

Extract the article text, build chunks, and generate the search index:

```bash
uv run python scripts/extract_text.py
uv run python scripts/build_chunks.py
uv run python scripts/build_index.py
```

### Ask a question

Ensure Ollama is running, then use the command-line interface:

```bash
uv run python scripts/ask.py "How does DyAb predict antibody properties with little data?"
```

If the retrieved passages do not support an answer, the system returns an abstention instead of a generated response presented as fact.

### Reproduce the evaluation

```bash
uv run python scripts/build_litqa2_benchmark.py
uv run python scripts/evaluate_litqa2.py --split tune
```

For long evaluation runs under WSL, Ollama's prompt cache may cause excessive memory use. The project documents the `LLAMA_ARG_CACHE_RAM=0` setting and the reason for using it in the [report](docs/REPORT.md).

## Docker

The repository includes a Docker Compose configuration with separate containers for the API and Ollama. Build the corpus and index locally before starting the services; the `data/` directory is mounted read-only inside the API container.

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen3:8b
curl http://localhost:8000/health
```

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs). The supplied Compose configuration requests an NVIDIA GPU for Ollama.

## Repository structure

```text
src/evidence_rag_assistant/
    ingestion/      Paper discovery, licence checks, and text extraction
    indexing/       Chunking and embedding-based indexing
    retrieval/      Passage search
    generation/     Answer generation and citation handling
    evaluation/     LitQA2 evaluation
scripts/            Pipeline and CLI entry points
tests/              Automated tests
notebooks/          Data inspection and implementation decisions
experiments/        Evaluation results
docs/               Technical report and development roadmap
```

## Documentation

- [Technical report](docs/REPORT.md) — design decisions, experiments, evaluation methodology, and limitations
- [Roadmap](docs/ROADMAP.md) — implementation status and planned work
- [Ingestion analysis](notebooks/01_ingestion_decisions.ipynb) — evidence supporting corpus and preprocessing decisions

## Data and attribution

Scientific articles are retrieved from [Europe PMC](https://europepmc.org) and remain subject to their respective licences. Full-text articles and derived data are not redistributed through this repository.

LitQA2 is provided by [FutureHouse](https://www.futurehouse.org) as part of [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) (CC BY-SA 4.0). The project uses the [BGE embedding model](https://huggingface.co/BAAI/bge-small-en-v1.5) and [Qwen3](https://github.com/QwenLM) for local inference.
