"""latent_lab — latent-cognition experiments for RCC.

Experimental layer: never required for importing the stdlib-only `rcc`
evidence substrate. Heavy dependencies (torch/transformers/mlx) are optional
and guarded at import time inside `latent_lab.backends.*` and
`latent_lab.bench.state_probe`.

Maturity note (2026-08-22): everything here is SCAFFOLDED + UNIT_VERIFIED on
the deterministic mock backend. No real-model claim is made until a
state-probe JSON report exists (see bench/state_probe.py).
"""

from .controller import ActionKind, ControllerAction, RuleController
from .intervals import LayerInterval, candidate_intervals, natural_groups
from .protocols import (
    LatentBackend,
    ModelInfo,
    ProblemInput,
    Readout,
)
from .provenance import ProvenanceLink, link_answer_to_evidence
from .recurrence import DecodeInsideLatentLoopError, LoopResult, run_latent_loop
from .state import STATE_SCHEMA_VERSION, RCCModelState, Vec
from .telemetry import TelemetryEvent, TelemetrySink

__all__ = [
    "STATE_SCHEMA_VERSION",
    "ActionKind",
    "ControllerAction",
    "DecodeInsideLatentLoopError",
    "LatentBackend",
    "LayerInterval",
    "LoopResult",
    "ModelInfo",
    "ProblemInput",
    "ProvenanceLink",
    "RCCModelState",
    "Readout",
    "RuleController",
    "TelemetryEvent",
    "TelemetrySink",
    "Vec",
    "candidate_intervals",
    "link_answer_to_evidence",
    "natural_groups",
    "run_latent_loop",
]
