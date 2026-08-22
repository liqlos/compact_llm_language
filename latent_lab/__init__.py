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
from .state import RCCModelState, STATE_SCHEMA_VERSION, Vec
from .telemetry import TelemetryEvent, TelemetrySink

__all__ = [
    "ActionKind",
    "ControllerAction",
    "RuleController",
    "LayerInterval",
    "candidate_intervals",
    "natural_groups",
    "LatentBackend",
    "LoopResult",
    "ModelInfo",
    "ProblemInput",
    "Readout",
    "ProvenanceLink",
    "link_answer_to_evidence",
    "DecodeInsideLatentLoopError",
    "run_latent_loop",
    "RCCModelState",
    "STATE_SCHEMA_VERSION",
    "Vec",
    "TelemetryEvent",
    "TelemetrySink",
]
