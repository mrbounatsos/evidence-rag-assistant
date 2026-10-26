"""Tests for the Ollama client's request handling and authentication."""

import io
import json

import pytest

from evidence_rag_assistant.generation import llm as llm_module
from evidence_rag_assistant.generation.llm import OllamaLLM, create_llm
from evidence_rag_assistant.settings import Settings


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace urlopen; record each request and reply with a fixed answer."""
    requests: list = []

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        requests.append(request)
        body = {"message": {"content": "<think>hidden</think>An answer [1]."}}
        return FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(llm_module, "urlopen", fake_urlopen)
    return requests


def test_request_without_auth_has_no_authorization_header(sent: list) -> None:
    reply = OllamaLLM("http://localhost:11434", "qwen3:8b").generate("system", "user")
    assert reply == "An answer [1]."
    assert sent[0].get_header("Authorization") is None
    payload = json.loads(sent[0].data)
    assert payload["options"] == {"temperature": 0, "num_ctx": 8192}


def test_request_with_auth_sends_header(sent: list) -> None:
    client = OllamaLLM("https://llm.example", "qwen3:8b", auth=lambda: "Bearer token-123")
    client.generate("system", "user")
    assert sent[0].get_header("Authorization") == "Bearer token-123"
    assert sent[0].full_url == "https://llm.example/api/chat"


def test_google_auth_is_only_used_when_configured() -> None:
    local = create_llm(Settings(_env_file=None))
    cloud = create_llm(
        Settings(_env_file=None, llm_auth="google_id_token", llm_base_url="https://llm.example")
    )
    assert local.auth is None  # type: ignore[attr-defined]
    assert cloud.auth is not None  # type: ignore[attr-defined]
