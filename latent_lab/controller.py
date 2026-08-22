"""Discrete controller over the latent loop.

Actions: CONTINUE | ACT | EXPAND | ANSWER | ABORT/UNCERTAIN.
Natural language is NOT required inside the controller. The rule-based
implementation reads only the Readout features (fixed-size vector) and
emits ControllerAction + telemetry; it never sees slot contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .protocols import Readout
from .telemetry import TelemetrySink


class ActionKind(str, Enum):
    CONTINUE = "CONTINUE"
    ACT = "ACT"
    EXPAND = "EXPAND"
    ANSWER = "ANSWER"
    ABORT = "ABORT"


@dataclass(frozen=True)
class ControllerAction:
    kind: ActionKind
    tool_id: str | None = None        # for ACT
    evidence_refs: tuple[str, ...] = ()  # for EXPAND
    halt_confidence: float = 0.0


class RuleController:
    """Deterministic halting policy: stop when readout confidence >= tau.

    features[0] is the backend's scalar confidence estimate. This is the
    fixed-depth baseline controller; learned halting is Stage T5.
    """

    def __init__(self, tau: float = 0.5, max_steps: int = 4,
                 sink: TelemetrySink | None = None):
        self.tau = tau
        self.max_steps = max_steps
        self.sink = sink or TelemetrySink()

    def decide(self, readout: Readout, step_index: int) -> ControllerAction:
        conf = float(readout.features[0]) if readout.features else 0.0
        if step_index >= self.max_steps and conf < self.tau:
            self.sink.emit("latent_loop", "halt", f"step_{step_index}", "aborted",
                           confidence=conf)
            return ControllerAction(ActionKind.ABORT, halt_confidence=conf)
        if conf >= self.tau:
            self.sink.emit("latent_loop", "halt", f"step_{step_index}", "ok",
                           confidence=conf)
            return ControllerAction(ActionKind.ANSWER, halt_confidence=conf)
        self.sink.emit("latent_loop", "continue", f"step_{step_index}", "running",
                       confidence=conf)
        return ControllerAction(ActionKind.CONTINUE, halt_confidence=conf)
