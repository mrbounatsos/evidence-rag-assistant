"""Request and response formats of the HTTP API.

These models are the API's contract: FastAPI validates requests against them,
serializes responses through them, and generates the interactive documentation
from them.
"""

from typing import Literal

from pydantic import BaseModel, Field

AnswerStatus = Literal["answered", "ungrounded", "abstained_by_model", "abstained_low_retrieval"]


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="The question to search for")
    k: int | None = Field(
        default=None, ge=1, le=20, description="Passages to retrieve; server default if omitted"
    )


class AskRequest(SearchRequest):
    pass


class Source(BaseModel):
    """A retrieved passage, with what is needed to attribute it."""

    number: int = Field(description="Number used for citations in the answer, e.g. [1]")
    chunk_id: str
    pmcid: str
    title: str
    section: str
    text: str
    score: float = Field(description="Embedding similarity to the question")
    licence: str | None = Field(description="Creative Commons licence of the source article")
    source_url: str = Field(description="The source article on Europe PMC")


class CitedSource(Source):
    cited: bool = Field(description="Whether the answer cites this source")


class ServiceConfig(BaseModel):
    """The configuration that produced a response."""

    retrieval: str
    k: int
    min_retrieval_score: float
    llm: str
    embedding_model: str


class SearchResponse(BaseModel):
    question: str
    top_score: float
    sources: list[Source]
    config: ServiceConfig


class AskResponse(BaseModel):
    question: str
    status: AnswerStatus
    answer: str
    top_score: float
    sources: list[CitedSource]
    invalid_citations: list[int]
    config: ServiceConfig


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks: int
    config: ServiceConfig
