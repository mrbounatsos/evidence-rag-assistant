"""Tests for configuration defaults, overrides and validation."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag_assistant.settings import Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore any EVIDENCERAG_ variables set on the machine running the tests."""
    for name in list(os.environ):
        if name.startswith("EVIDENCERAG_"):
            monkeypatch.delenv(name)


def test_defaults_match_evaluated_configuration() -> None:
    settings = Settings(_env_file=None)
    assert settings.retrieval_mode == "hybrid"
    assert settings.k == 8
    assert settings.llm_model == "qwen3:8b"
    assert settings.chunks_path == Path("data/processed/chunks.jsonl")


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCERAG_K", "5")
    monkeypatch.setenv("EVIDENCERAG_DATA_DIR", "/mnt/data")
    settings = Settings(_env_file=None)
    assert settings.k == 5
    assert settings.index_dir == Path("/mnt/data/index")


def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCERAG_K", "50")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
