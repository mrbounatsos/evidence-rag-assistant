"""HTTP API for EvidenceRAG.

The embedding model, indexes, chunks and model client are loaded once at startup
and shared by all requests. Configuration comes from Settings, so the same app
runs locally and in a container. Run with:

    uv run uvicorn evidence_rag_assistant.api.app:app --port 8000
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from evidence_rag_assistant.api.log import configure_logging
from evidence_rag_assistant.api.schemas import (
    AskRequest,
    AskResponse,
    CitedSource,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    ServiceConfig,
    Source,
)
from evidence_rag_assistant.generation.answer import answer_question
from evidence_rag_assistant.generation.llm import LLM, GenerationError, create_llm
from evidence_rag_assistant.retrieval.retriever import RetrievedChunk, Retriever
from evidence_rag_assistant.settings import Settings, get_settings

logger = logging.getLogger("evidence_rag_assistant.api")

EUROPE_PMC_ARTICLE_URL = "https://europepmc.org/articles/{pmcid}"


@dataclass
class Resources:
    """Everything a request needs, loaded once at startup."""

    settings: Settings
    retriever: Retriever
    llm: LLM


def load_resources(settings: Settings) -> Resources:
    retriever = Retriever(settings.chunks_path, settings.index_dir, mode=settings.retrieval_mode)
    if retriever.stale:
        # Fail closed: better no service than one that cites the wrong text.
        raise RuntimeError(
            "The index is stale: chunks changed since it was built. Rebuild the index."
        )
    return Resources(settings, retriever, create_llm(settings))


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]


def service_config(resources: Resources, k: int) -> ServiceConfig:
    return ServiceConfig(
        retrieval=resources.settings.retrieval_mode,
        k=k,
        min_retrieval_score=resources.settings.min_retrieval_score,
        llm=resources.llm.model,
        embedding_model=resources.retriever.meta["model"],
    )


def to_source(number: int, item: RetrievedChunk) -> Source:
    chunk = item.chunk
    return Source(
        number=number,
        chunk_id=chunk["chunk_id"],
        pmcid=chunk["pmcid"],
        title=chunk["title"],
        section=" > ".join(chunk["section_path"]) or "Main text",
        text=chunk["text"],
        score=item.score,
        licence=chunk.get("licence"),
        source_url=EUROPE_PMC_ARTICLE_URL.format(pmcid=chunk["pmcid"]),
    )


def create_app(
    load: Callable[[Settings], Resources] = load_resources,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own `load` to avoid real models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = settings or get_settings()
        configure_logging(config.log_level)
        app.state.resources = load(config)
        logger.info("service started")
        yield

    app = FastAPI(
        title="EvidenceRAG",
        description="Answers from biomedical papers, with citations, or not at all.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000),
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(resources: ResourcesDep) -> HealthResponse:
        return HealthResponse(
            status="ok",
            chunks=len(resources.retriever.chunk_ids),
            config=service_config(resources, resources.settings.k),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, resources: ResourcesDep) -> SearchResponse:
        k = body.k or resources.settings.k
        sources = resources.retriever.retrieve(body.question, k)
        return SearchResponse(
            question=body.question,
            top_score=max((s.score for s in sources), default=0.0),
            sources=[to_source(n, s) for n, s in enumerate(sources, start=1)],
            config=service_config(resources, k),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, resources: ResourcesDep) -> AskResponse:
        k = body.k or resources.settings.k
        try:
            answer = answer_question(
                body.question,
                resources.retriever,
                resources.llm,
                k=k,
                min_score=resources.settings.min_retrieval_score,
            )
        except GenerationError as error:
            logger.exception("language model unavailable")
            raise HTTPException(
                status_code=503, detail="The language model is unavailable. Try again later."
            ) from error

        logger.info("answer", extra={"answer_status": answer.status})
        return AskResponse(
            question=body.question,
            status=answer.status,  # type: ignore[arg-type]
            answer=answer.text,
            top_score=answer.top_score,
            sources=[
                CitedSource(**to_source(n, s).model_dump(), cited=n in answer.cited)
                for n, s in enumerate(answer.sources, start=1)
            ],
            invalid_citations=answer.invalid_citations,
            config=service_config(resources, k),
        )

    return app


app = create_app()
