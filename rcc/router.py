"""Deterministic scratch-mode router (Phase 5 step 2).

Chooses how much machine state to surface from cheap, observable signals —
no model call involved:

    FULL     safety-critical: conflicts present, repeated failures, or
             suspected injection in recent observations
    EXPERT   long structured investigations (many observations)
    SYMBOLIC default working state once several atoms exist
    DRAFT    light note-taking during early exploration
    DIRECT   nothing worth surfacing (trivial ops)

The mode only controls VISIBILITY of the RIR block; atoms are always
recorded in state, journal and timeline, so observability never degrades.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("DIRECT", "DRAFT", "SYMBOLIC", "EXPERT", "FULL")


@dataclass(frozen=True)
class TaskSignals:
    n_observations: int
    n_atoms: int
    n_conflicts: int = 0
    recent_failures: int = 0
    injection_suspected: bool = False


def route(signals: TaskSignals) -> str:
    if signals.injection_suspected or signals.recent_failures >= 2:
        return "FULL"
    if signals.n_conflicts > 0:
        return "FULL"
    if signals.n_observations >= 8:
        return "EXPERT"
    if signals.n_atoms >= 3:
        return "SYMBOLIC"
    if signals.n_observations >= 1 or signals.n_atoms >= 1:
        return "DRAFT"
    return "DIRECT"
