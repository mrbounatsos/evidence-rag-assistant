"""Language-model backends behind one small interface.

The rest of the system depends only on LLM.generate(system, user) -> text, so the
model server can change (Ollama locally or on a GPU service today; vLLM or a
managed model later) by adding one implementation here and selecting it in
create_llm.
"""

import json
import re
import time
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token

from evidence_rag_assistant.settings import Settings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class GenerationError(RuntimeError):
    """Raised when the model request still fails after retries."""


class LLM(Protocol):
    model: str

    def generate(self, system: str, user: str) -> str: ...


class GoogleIdTokenAuth:
    """Authorization header for calling a private Cloud Run service.

    On Cloud Run the token comes from the instance's service account via the
    metadata server: no keys are involved. Tokens are fetched on first use and
    refreshed automatically when they expire.
    """

    def __init__(self, audience: str) -> None:
        self._audience = audience
        self._request = GoogleAuthRequest()
        self._credentials: Credentials | None = None

    def header(self) -> str:
        credentials = self._credentials
        if credentials is None:
            credentials = id_token.fetch_id_token_credentials(self._audience, request=self._request)
            self._credentials = credentials
        if not credentials.valid:
            credentials.refresh(self._request)
        return f"Bearer {credentials.token}"


class OllamaLLM:
    """Ollama chat API: deterministic (temperature 0), reasoning mode off."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300,
        retries: int = 2,
        context_tokens: int = 8192,
        auth: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.context_tokens = context_tokens
        self.auth = auth

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
        data = json.dumps(payload).encode("utf-8")
        last_error = ""
        for attempt in range(self.retries + 1):
            headers = {"Content-Type": "application/json"}
            if self.auth is not None:
                headers["Authorization"] = self.auth()
            request = Request(f"{self.base_url}/api/chat", data=data, headers=headers)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data_out = json.load(response)
                content: str = data_out["message"]["content"]
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
        auth = (
            GoogleIdTokenAuth(settings.llm_base_url).header
            if settings.llm_auth == "google_id_token"
            else None
        )
        return OllamaLLM(
            settings.llm_base_url,
            settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            auth=auth,
        )
    raise ValueError(f"Unknown LLM backend {settings.llm_backend!r}")
