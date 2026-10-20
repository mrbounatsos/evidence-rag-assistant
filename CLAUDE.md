# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

EvidenceRAG is a local-first biomedical research assistant built as a public portfolio project. It answers questions from open-access research papers (initially protein structure prediction and antibody design), cites its sources, and abstains when the evidence is insufficient.

The project should demonstrate real engineering quality: reproducible data ingestion, measurable retrieval and answer quality, clean code, and honest documentation. Prefer simple, well-tested solutions over impressive-looking ones.

## Environment

- Python 3.12, managed with `uv` (project `.venv`). Add dependencies with `uv add`, never `pip install`.
- Runs under WSL2 on Linux.
- Local LLM via Ollama at `http://localhost:11434`, model `qwen3:8b`.
- No paid APIs. The core workflow must run offline once data and models are downloaded.
- Do not install the CUDA toolkit; Ollama handles GPU use on its own.

## Commands

Run all of these before proposing a commit:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run pytest -q
```

Use `uv run ruff format .` to fix formatting.

## Git rules

- **Never modify global Git configuration** (`git config --global`, `~/.gitconfig`). Other GitHub identities exist on this machine and must not be affected.
- The remote uses the SSH alias `github-personal`. Do not change the remote URL or SSH config.
- Author identity is set at repository level only. Do not change it.
- Do not commit, push, or rewrite history without explicit approval. Show `git status --short` and the proposed commit message first.
- Never stage anything under `data/` (except `.gitkeep` files), `experiments/results/`, `.env`, model files, or other local artifacts.

## Data and licensing rules

These rules are part of the project's integrity and must not be weakened:

- Do not assume that open access or public availability means content can be reused. Check each article's licence.
- The licence in an article's own JATS XML is authoritative. Search-index metadata is only a first filter.
- Only write full text to disk when its licence is in the allowed set. Exclude articles whose licence is missing or cannot be parsed.
- Preserve provenance on every record: source, identifiers (PMCID, PMID, DOI), licence, licence evidence, retrieval query, timestamps, and content hashes.
- Article text and anything derived from it (chunks, embeddings, indexes) stay local and out of Git. The public repository contains code, configuration, and benchmarks that reference articles by identifier, not copied text.
- Respect API usage: identify with a `User-Agent`, pause between requests, cache results, and make scripts safe to re-run.

## Code conventions

- Type hints everywhere; code must pass mypy.
- Reusable logic (licence parsing, XML extraction, chunking, retrieval) belongs in the `src` package with tests. `scripts/` holds thin entry points.
- Tests must not hit the network. Use small hand-made fixtures (e.g. minimal JATS XML files) instead.
- Keep dependencies minimal. Use LangChain or other frameworks only where they clearly add value.
- Configuration via environment variables (see `.env.example`); never hard-code secrets.

## Working style

- Work in small, verifiable steps. After each step, run the relevant commands and report the actual output rather than assuming success.
- Briefly explain significant engineering decisions, including trade-offs.
- Follow the roadmap order. Do not start the API or UI before ingestion, retrieval, and evaluation are solid.
- Never overstate results in documentation or the README. Reported metrics must come from runs that can be reproduced.

## Roadmap

0. Environment and repository setup (done)
1. Data ingestion: discovery, licence verification, full-text retrieval, text extraction (in progress)
2. Chunking and indexing
3. Retrieval (FAISS + BM25 hybrid) and benchmarking
4. Grounded RAG answers with citations and abstention
5. Evaluation
6. FastAPI + Pydantic API
7. Streamlit UI
8. Packaging, Docker, CI, documentation, public release