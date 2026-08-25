"""Provider abstraction for the live-eval harness.

`LLMClient` is the single seam: the harness only ever calls
``complete(system, user)``. Two implementations ship here:

- :class:`OpenAICompatClient` -- stdlib-only client for any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, Ollama, llama.cpp server,
  LM Studio, vLLM...). The API key is taken from an explicit argument or the
  ``RCC_EVAL_API_KEY`` environment variable and is never logged or serialized.
- :class:`FixtureClient` -- deterministic offline fake used by tests and by
  ``--provider fake`` runs. It answers strictly from what is visible in its
  prompt (plus the task's required facts passed via ``meta``), so it exercises
  the full harness mechanics without pretending to be a real model.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

STUB_RE = re.compile(r"\[OBS (obs-\d{4})[^\]]*\]")
EXPAND_LINE_RE = re.compile(r"^\s*EXPAND:\s*(obs-\d{4})\s*$", re.MULTILINE)


class ProviderError(RuntimeError):
    """Raised when a live provider call fails or is misconfigured."""

    def __init__(self, msg: str, *, retryable: bool = False) -> None:
        super().__init__(msg)
        self.retryable = retryable


# Transient HTTP statuses worth a bounded retry.
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class LLMClient(Protocol):
    def complete(self, system: str, user: str, **meta: Any) -> str: ...


def parse_expand_requests(reply: str, known_ids: set[str], cap: int) -> list[str]:
    """Extract valid, deduplicated obs ids from EXPAND lines; cap the count."""
    seen: list[str] = []
    for obs_id in EXPAND_LINE_RE.findall(reply):
        if obs_id in known_ids and obs_id not in seen:
            seen.append(obs_id)
            if len(seen) >= cap:
                break
    return seen


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client (no third-party dependencies).

    Bounded cost by construction: temperature 0 and a hard completion-token
    cap. The caller controls how many requests are made (the suite makes at
    most scenarios x modes x (1 + max_expand_rounds) calls).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 60.0,
        max_completion_tokens: int = 512,
        max_retries: int = 2,
        backoff_s: float = 0.25,
    ) -> None:
        if not base_url or not model:
            raise ProviderError("base_url and model are required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.calls_made = 0

    def _post_once(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_completion_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:  # keep auth material out of errors
            raise ProviderError(f"HTTP {e.code} from {self.base_url}",
                                retryable=e.code in _RETRYABLE_HTTP) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise ProviderError(
                f"cannot reach {self.base_url}: {reason}", retryable=True) from e
        try:
            data = json.loads(raw)
            return str(data["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise ProviderError(
                f"unexpected response shape from {self.base_url}") from e

    def complete(self, system: str, user: str, **_: Any) -> str:
        err: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self.calls_made += 1
                return self._post_once(system, user)
            except ProviderError as e:
                err = e
                if not e.retryable or attempt >= self.max_retries:
                    raise
                time.sleep(min(self.backoff_s * (2 ** attempt), 5.0))
        raise err  # pragma: no cover -- loop always returns or raises

    def check(self) -> str:
        """One minimal completion to verify endpoint+model before a full run."""
        return self.complete(
            "You are a health check. Reply with the single word OK.", "ping")

    @classmethod
    def from_env(cls, base_url: str | None, model: str | None, **kw: Any) -> OpenAICompatClient:
        url = base_url or os.environ.get("RCC_EVAL_BASE_URL", "")
        name = model or os.environ.get("RCC_EVAL_MODEL", "")
        key = kw.pop("api_key", None) or os.environ.get("RCC_EVAL_API_KEY")
        return cls(url, name, api_key=key, **kw)


class FixtureClient:
    """Deterministic offline stand-in for a live model (aka FakeProvider).

    Behaviour (documented contract, exercised heavily by tests):
    - Tool arm round 1: if some required facts are not visible in the prompt
      AND observation stubs offer expansion, reply ONLY with
      ``EXPAND: <obs-id>`` lines for every stubbed id (bounded).
    - Answer phase: quote every required fact found verbatim in the current
      prompt and cite the first observation id it can see. On the closed-book
      arm it never asserts missing facts -- it flags unavailability instead,
      which is exactly what EF3 scores. It never emits content that is not
      present in its own prompt, so injection payloads can never leak through
      this fixture.
    """

    def __init__(self, *, expand_cap: int = 32) -> None:
        self.expand_cap = expand_cap
        self.calls_made = 0

    def complete(self, system: str, user: str, **meta: Any) -> str:
        self.calls_made += 1
        facts: tuple[str, ...] = tuple(meta.get("required_facts", ()))
        allow_expand = bool(meta.get("allow_expand", False))
        missing = [f for f in facts if f not in user]
        stub_ids = STUB_RE.findall(user)
        if missing and stub_ids and allow_expand and "EXPAND:" in system:
            lines = "\n".join(f"EXPAND: {oid}" for oid in stub_ids[: self.expand_cap])
            return lines
        found = [f for f in facts if f in user]
        if found:
            body = " | ".join(found)
        elif facts:
            body = "(not present in context)"
        else:
            body = "Done, based only on the provided context."
        if missing and not allow_expand:
            body += " Additional evidence is unavailable in this context."
        cite_match = STUB_RE.search(user) or re.search(r"id=(obs-\d{4})", user)
        answer = f"ANSWER: {body}"
        if cite_match:
            answer += f" [source {cite_match.group(1)}]"
        return answer


# Audit-lane naming aliases: the client seam is provider-neutral by design.
ModelClient = LLMClient
FakeProvider = FixtureClient
