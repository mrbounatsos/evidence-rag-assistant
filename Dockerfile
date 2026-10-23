# syntax=docker/dockerfile:1

ARG UV_VERSION=0.13.3

# --- uv, pinned to the version used locally -------------------------------
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- Builder: install dependencies and bake in the embedding model --------
FROM python:3.12-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    HF_HOME=/app/hf-cache
WORKDIR /app

# Dependencies first: this layer is reused as long as the lock file is unchanged.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra cpu --no-install-project

# Then the project itself, installed into the environment (not as editable).
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra cpu --no-editable

# Download the embedding model at build time, so the container never does.
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# --- Runtime: only the environment and the model, run as a non-root user ---
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/hf-cache /app/hf-cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EVIDENCERAG_DATA_DIR=/app/data

USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn evidence_rag_assistant.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]