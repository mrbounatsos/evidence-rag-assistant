"""Tests for the HTTP API, using fake resources instead of real models."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from evidence_rag_assistant.api.app import Resources, create_app
from evidence_rag_assistant.generation.answer import ABSTAIN_MESSAGE
from evidence_rag_assistant.generation.llm import GenerationError
from evidence_rag_assistant.retrieval.retriever import RetrievedChunk
from evidence_rag_assistant.settings import Settings


class FakeRetriever:
    meta = {"model": "fake-embedder"}
    chunk_ids = ["PMC1:0000", "PMC1:0001"]

    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        chunk = {
            "chunk_id": "PMC1:0000",
            "pmcid": "PMC1",
            "title": "A paper",
            "section_path": ["Results"],
            "text": "A finding.",
            "licence": "cc by",
        }
        return [RetrievedChunk(chunk, self.score)]


class ScriptedLLM:
    model = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


class FailingLLM:
    model = "failing"

    def generate(self, system: str, user: str) -> str:
        raise GenerationError("connection refused at 10.0.0.5:11434")


def client_for(retriever: object, llm: object) -> TestClient:
    settings = Settings(_env_file=None)
    resources = Resources(settings, retriever, llm)  # type: ignore[arg-type]
    return TestClient(create_app(load=lambda _: resources, settings=settings))


@pytest.fixture
def scripted() -> ScriptedLLM:
    return ScriptedLLM("The finding holds [1].")


@pytest.fixture
def client(scripted: ScriptedLLM) -> Iterator[TestClient]:
    with client_for(FakeRetriever(), scripted) as test_client:
        yield test_client


def test_health_reports_configuration(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2
    assert body["config"]["retrieval"] == "hybrid"


def test_ask_returns_cited_sources_with_attribution(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What was found?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    source = body["sources"][0]
    assert source["cited"] is True
    assert source["licence"] == "cc by"
    assert source["source_url"] == "https://europepmc.org/articles/PMC1"
    assert "x-request-id" in response.headers


def test_search_does_not_call_the_model(client: TestClient, scripted: ScriptedLLM) -> None:
    response = client.post("/search", json={"question": "What was found?", "k": 3})
    assert response.status_code == 200
    assert response.json()["config"]["k"] == 3
    assert scripted.calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Hi"},
        {"question": "x" * 1001},
        {"question": "What was found?", "k": 50},
        {},
    ],
    ids=["too-short", "too-long", "k-too-large", "missing-question"],
)
def test_invalid_requests_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_model_abstention_is_reported() -> None:
    with client_for(FakeRetriever(), ScriptedLLM(ABSTAIN_MESSAGE)) as test_client:
        body = test_client.post("/ask", json={"question": "What was found?"}).json()
    assert body["status"] == "abstained_by_model"
    assert body["sources"][0]["cited"] is False


def test_model_failure_returns_503_without_internal_details() -> None:
    with client_for(FakeRetriever(), FailingLLM()) as test_client:
        response = test_client.post("/ask", json={"question": "What was found?"})
    assert response.status_code == 503
    assert "10.0.0.5" not in response.text
