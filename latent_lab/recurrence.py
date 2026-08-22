"""Latent recurrence loop with structural no-decode enforcement.

run_latent_loop executes K continuous steps H[t+1] = F_interval(H[t], ...).
Between steps there is no argmax, no vocabulary decode, no string
conversion, no re-tokenization: the loop counts backend.decode_calls before
and after and raises DecodeInsideLatentLoopError if it changed. This makes
"hidden textual CoT" structurally detectable, not merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .controller import ActionKind, ControllerAction, RuleController
from .state import RCCModelState
from .telemetry import TelemetrySink


class DecodeInsideLatentLoopError(RuntimeError):
    """Backend decoded tokens during a latent step — not latent reasoning."""


@dataclass
class LoopResult:
    final_state: RCCModelState
    steps_executed: int
    halted: bool
    action: ControllerAction | None = None
    telemetry_events: list = field(default_factory=list)


def run_latent_loop(
    backend,
    state: RCCModelState,
    *,
    k_steps: int,
    controller: RuleController | None = None,
    sink: TelemetrySink | None = None,
) -> LoopResult:
    """Run up to k_steps continuous recurrence steps.

    Halting: if a controller is provided it decides CONTINUE vs ANSWER/ABORT
    from readout features; otherwise exactly k_steps run (fixed depth).
    """
    if k_steps < 0:
        raise ValueError("k_steps must be >= 0")
    sink = sink or TelemetrySink()
    controller = controller or RuleController(max_steps=k_steps)
    decode_before = backend.decode_calls

    cur = state
    steps = 0
    action: ControllerAction | None = None

    sink.emit("latent_loop", "start", f"layers_{cur.interval[0]}_{cur.interval[1]}",
              "running", k_steps=k_steps)

    while steps < k_steps:
        cur = backend.latent_step(cur)   # pure state -> state; no decode legal
        steps += 1
        ro = backend.readout(cur)
        action = controller.decide(ro, steps)
        if action.kind is not ActionKind.CONTINUE:
            break

    decode_after = backend.decode_calls
    if decode_after != decode_before:
        raise DecodeInsideLatentLoopError(
            f"backend performed {decode_after - decode_before} decode call(s) "
            "inside the latent loop — this is token generation, not latent reasoning"
        )

    halted = action is not None and action.kind is not ActionKind.CONTINUE
    sink.emit("latent_loop", "end", f"steps_{steps}", "ok",
              halted=halted,
              final_action=action.kind.value if action else "CONTINUE")
    return LoopResult(
        final_state=cur,
        steps_executed=steps,
        halted=halted,
        action=action,
        telemetry_events=sink.events(),
    )
