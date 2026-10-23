"""Language-model backends behind one small interface.

The rest of the system depends only on LLM.generate(system, user) -> text, so the
model server can change (Ollama locally or on a GPU service today; vLLM or a
managed model later) by adding one implementation here and selecting it in
create_llm.
"""

import json
import re
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evidence_rag_assistant.settings import Settings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class GenerationError(RuntimeError):
    """Raised when the model request still fails after retries."""


class LLM(Protocol):
    model: str

    def generate(self, system: str, user: str) -> str: ...


class OllamaLLM:
    """Ollama chat API: deterministic (temperature 0), reasoning mode off."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300,
        retries: int = 2,
        context_tokens: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.context_tokens = context_tokens

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": self.context_tokens},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                content: str = data["message"]["content"]
                # Defensive: drop any reasoning block if the model emits one anyway.
                return _THINK_BLOCK.sub("", content).strip()
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {error.code}: {detail}"
            except (URLError, TimeoutError) as error:
                last_error = str(error)
            if attempt < self.retries:
                time.sleep(2 * (attempt + 1))
        raise GenerationError(
            f"Model request failed after {self.retries + 1} attempts: {last_error}"
        )


def create_llm(settings: Settings) -> LLM:
    """Build the configured language-model backend."""
    if settings.llm_backend == "ollama":
        return OllamaLLM(
            settings.llm_base_url, settings.llm_model, timeout=settings.llm_timeout_seconds
        )
    raise ValueError(f"Unknown LLM backend {settings.llm_backend!r}")
