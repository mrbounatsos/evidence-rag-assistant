# Roadmap

This is the source of truth for EvidenceRAG's progress. The README shows a summary; detailed reasoning behind each decision is in [REPORT.md](REPORT.md).

**Status legend:** ✅ done · 🚧 in progress · 📋 planned · 💤 optional

_Last updated: 21 October 2026_

---

## Overview

| # | Module | Status |
|---|---|---|
| 0 | Environment and repository | ✅ |
| 1 | Data ingestion | ✅ |
| 2 | Chunking and indexing | ✅ |
| 3 | Retrieval and benchmarking | 🚧 |
| 4 | Grounded answers with citations and abstention | ✅ |
| 5 | Evaluation | 🚧 |
| 6 | API (FastAPI + Pydantic) | 📋 |
| 7 | User interface (Streamlit) | 📋 |
| 8 | Packaging, Docker, CI, documentation | 📋 |
| 9 | Cloud deployment (GCP) | 💤 |

### Why the order changed

The original plan finished retrieval benchmarking (Module 3) before building answer generation (Module 4). After the extraction work, the project switched to **building a complete system first**, then measuring and improving where the measurements point. Generation was therefore built before retrieval was fully optimized, and Modules 3 and 5 now advance together on the same benchmark.

---

## 0 · Environment and repository ✅

- [x] Python 3.12 project managed with `uv`
- [x] Local LLM via Ollama (`qwen3:8b`) on the GPU
- [x] Quality gate: Ruff, mypy, pytest
- [x] Public repository with isolated personal Git identity
- [x] Repository hygiene: data, indexes and results never committed
- [x] `CLAUDE.md` with project ground rules for AI coding assistants

## 1 · Data ingestion ✅

- [x] Candidate discovery from Europe PMC
- [x] Licence policy: code-only publication, all Creative Commons licences accepted for local use
- [x] Full-text retrieval with licence verification against each article's own XML
- [x] Fail-closed licence detection, tested against real publisher patterns
- [x] Manifest of every fetch attempt, with SHA-256 fingerprints
- [x] Text extraction from JATS XML: section classification, citation-marker removal, formula handling
- [x] Extraction audit of every skipped element
- [x] Decisions notebook: eight findings, each as question → evidence → decision

## 2 · Chunking and indexing ✅

- [x] Section-aware chunking (~300 words), captions as separate chunks
- [x] Title and section prefix for each chunk
- [x] BGE embeddings and exact FAISS index
- [x] Stale-index detection

## 3 · Retrieval and benchmarking 🚧

- [x] Dense (embedding) retrieval
- [x] Exclusion of selected papers, used for unanswerable-question evaluation
- [x] Keyword search (BM25) — next
- [x] Hybrid retrieval combining keyword and embedding search
- [x] Limit on chunks per paper, for more diverse results
- [x] Retrieve 8 passages instead of 5 (experiment 1)
- [ ] Optional reranking of retrieved passages

## 4 · Grounded answers with citations and abstention ✅

- [x] Prompt with numbered sources and strict grounding rules
- [x] Two layers of abstention: retrieval threshold and model decision
- [x] Citation validation; answers without valid citations labelled ungrounded
- [x] Retries and error reporting for model calls
- [x] Command-line tools to search and ask

## 5 · Evaluation 🚧

- [x] LitQA2 selected as human-validated benchmark
- [x] LitQA2 source papers verified and added to the corpus (89 questions usable)
- [x] Permanent tune/test split (38 / 51 questions)
- [x] Three evaluation modes: normal, source removed, closed book
- [x] Retrieval and answer metrics; full configuration recorded with every run
- [x] **Baseline on the tune split** — 84% accuracy, 97% precision
- [x] **Baseline on the test split**
- [ ] Hybrid retrieval vs. baseline, on tune, then confirmed on test
- [ ] Automatic check that cited passages support their claims
- [ ] Harder setting with similar distractor papers

## 6 · API 📋

- [ ] FastAPI service exposing search and question answering
- [ ] Pydantic models for requests, answers, citations and statuses
- [ ] Health check and configuration through environment variables

## 7 · User interface 📋

- [ ] Streamlit app: ask a question, read the answer, inspect cited passages
- [ ] Clear display of abstentions and retrieval scores

## 8 · Packaging, Docker, CI, documentation 📋

- [ ] Docker images for the API and the model server; Docker Compose for local use
- [ ] GitHub Actions: lint, type check and tests on every push
- [ ] Offline mode by default after first download
- [ ] Troubleshooting notes (including the WSL memory issue)
- [ ] Code licence and contribution notes

## 9 · Cloud deployment (GCP) 💤

Optional, to explore production tooling and use more compute. The project stays fully runnable locally.

- [ ] Container images in Artifact Registry
- [ ] Cloud Run service with a GPU for the model, scaling to zero
- [ ] Cloud Run jobs for evaluation runs
- [ ] Private Cloud Storage bucket for corpus and index
- [ ] Budget alerts from day one
- [ ] Optional: Kubernetes (GKE) variant with GPU node pool

---

## Experiment backlog

Each experiment is measured on the tune split against the current baseline.

| Experiment | Question it answers |
|---|---|
| Hybrid retrieval | Does keyword search recover passages that hinge on rare exact terms? |
| Chunk size and overlap | Do smaller or overlapping chunks improve passage retrieval? |
| Number of passages | Does giving the model more passages help or distract? |
| Chunks per paper limit | Does diversity help broad questions? |
| Reasoning mode | Does letting the model reason first improve answers, and at what speed cost? |
| Retrieval threshold | Is the off-topic threshold useful, and at what value? |
| Larger model | How much does model size matter, given more compute? |

## Known limitations

See [REPORT.md, section 15](REPORT.md#15-known-limitations).