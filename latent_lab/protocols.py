"""Core protocols for latent-cognition backends.

A LatentBackend exposes the exact control points the localized-recurrence
architecture needs. Structural rule: `latent_step` maps model state to model
state only — there is no vocabulary decode, no string round-trip, no
tokenization inside the loop. Decoding exists as a separate method that is
only legal OUTSIDE a latent loop; `recurrence.run_latent_loop` enforces this
by counting backend decode calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .state import RCCModelState, Vec

LATENT_LOOP_DEPTH_UNLIMITED = -1


@dataclass(frozen=True)
class ModelInfo:
    """Identity of the loaded model/runtime (feeds RCCModelState ABI header)."""

    repo_id: str
    revision: str
    config_hash: str
    tokenizer_hash: str
    runtime: str                      # "mock" | "hf-pt" | "mlx" | ...
    runtime_version: str
    dtype: str                        # "fp32" | "fp16" | "bf16" | "int4" | ...
    quantization: str | None = None
    layer_types: tuple[str, ...] = ()  # e.g. ("gdn","gdn","gdn","attn") * 6


@dataclass(frozen=True)
class ProblemInput:
    """One problem instance entering the latent pipeline.

    `payload` is backend-specific machine content (token ids / embeddings /
    structured records) — never free-form telemetry text.
    """

    problem_id: str
    payload: tuple[Vec, ...]          # sequence of vectors (mock: tiny dims)
    evidence_refs: tuple[str, ...] = ()  # provenance handles into raw store


@dataclass(frozen=True)
class Readout:
    """Fixed readout slots after contextualization/recurrence.

    `features` feed the discrete controller; `answer_handle` is an opaque
    handle the backend can decode AFTER the loop. No strings here.
    """

    features: Vec
    answer_handle: object


@runtime_checkable
class LatentBackend(Protocol):
    """Minimal ABI every backend (mock, hf_qwen, mlx_qwen) implements."""

    def info(self) -> ModelInfo: ...

    def workspace_dims(self) -> tuple[int, int, int]:
        """(n_memory_slots M, n_working_slots H, n_readout_slots R)."""
        ...

    def contextualize(self, problem: ProblemInput) -> RCCModelState:
        """Run the lower decoder prefix ONCE; return problem-conditioned
        M0/H0/R0 workspace plus initial cache/recurrent state."""
        ...

    def latent_step(self, state: RCCModelState) -> RCCModelState:
        """One continuous recurrence step over the selected layer interval.

        Contract: consumes and returns RCCModelState; performs no vocabulary
        decode and no string conversion. Implementations must be instrumented
        via `decode_calls` for loop-time enforcement."""
        ...

    def readout(self, state: RCCModelState) -> Readout: ...

    def decode(self, state: RCCModelState, answer_handle: object) -> str:
        """Materialize text. Legal ONLY outside a latent loop."""
        ...

    @property
    def decode_calls(self) -> int:
        """Monotonic counter of decode invocations (loop guard reads it)."""
        ...


def check_backend(b: object) -> None:
    """Fail fast when an object does not satisfy the structural protocol."""
    if not isinstance(b, LatentBackend):
        raise TypeError(f"{type(b).__name__} does not satisfy LatentBackend")
    m, h, r = b.workspace_dims()
    if min(m, h, r) < 1:
        raise ValueError("workspace requires at least one slot each (M,H,R)")
