"""Query-aware break-even gate (Phase 7) + compressor plug point (Phase 5.3).

Masking is a decision, not an automatic win. Per-turn amortised comparison:

    keep-inline cost : D * N * (1 + cache_penalty)
    mask+offload cost: S * N + q * N * (s + c)

    D content tokens, S stub tokens, N remaining turns,
    q expected re-reads per turn, s tokens actually re-read on expansion,
    c expansion overhead (tool envelope, rendering).

Defaults are deliberately conservative; q and N must come from measured
workloads, not vibes (documented in the plan). Fail-open: when in doubt,
keep inline.

Compressor protocol: model-assisted distillation (Phase 5 step 3) plugs in
here behind the same gate. NullCompressor keeps the system fully
deterministic offline; SafeCompressor converts any compressor failure into
None -> callers fall back verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Usage:
    content_tokens: int            # D
    stub_tokens: int               # S
    remaining_turns: int           # N
    expected_reads_per_turn: float = 0.05   # q -- conservative default
    read_slice_tokens: int = 400            # s
    expand_overhead_tokens: int = 50        # c
    cache_penalty: float = 0.0              # extra multiplier on inline churn


@dataclass(frozen=True)
class Decision:
    mask: bool
    window_cost: int
    offload_cost: int
    reason: str


def break_even(u: Usage) -> Decision:
    if u.remaining_turns <= 0 or u.content_tokens <= 0:
        return Decision(False, 0, 0, "no future to amortise into")
    window = round(u.content_tokens * u.remaining_turns * (1 + u.cache_penalty))
    offload = round(
        u.stub_tokens * u.remaining_turns
        + u.expected_reads_per_turn * u.remaining_turns
        * (u.read_slice_tokens + u.expand_overhead_tokens)
    )
    if offload < window:
        return Decision(True, window, offload,
                        f"offload {offload} < inline {window}")
    return Decision(False, window, offload,
                    f"offload {offload} >= inline {window}; keep verbatim")


class Compressor(Protocol):
    def compress(self, text: str) -> str | None: ...


class NullCompressor:
    """Deterministic no-op: never distils anything."""

    def compress(self, text: str) -> str | None:
        return None


class SafeCompressor:
    """Wraps any compressor; failures degrade to None (fail-open), never raise."""

    def __init__(self, inner: Compressor):
        self._inner = inner

    def compress(self, text: str) -> str | None:
        try:
            out = self._inner.compress(text)
        except Exception:  # noqa: BLE001 -- fail-open boundary for 3rd-party code
            return None
        if not isinstance(out, str) or not out.strip():
            return None
        return out
