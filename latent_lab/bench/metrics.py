"""Benchmark metrics — wall-clock first, honesty enforced by schema.

Primary metric: wall-clock time per successful task. Synthetic token
estimates from bench/ are NEVER mixed into these numbers.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import asdict, dataclass, field


@dataclass
class TaskMetrics:
    task_id: str
    config: str                    # e.g. "direct", "latent_k4_interval_early"
    success: bool | None           # None = scorer not applicable (recorded, not faked)
    wall_seconds: float
    latent_steps: int = 0
    decode_calls_in_loop: int = 0  # must be 0 for latent configs
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelfCheckReport:
    """Mechanical correctness of the harness/backend BEFORE any benchmarking.

    All checks must pass on a real backend before its quality numbers mean
    anything (see EVALUATION_PROTOCOL.md anti-cheating controls).
    """

    backend: str
    k_causal_differs: bool = False          # K=0 vs K>0 change the state
    zero_state_changes_readout: bool = False
    swapped_state_changes_readout: bool = False
    truncated_depth_changes_readout: bool = False
    no_decode_inside_loop: bool = True      # loop raises if violated
    deterministic_rerun: bool = False       # same input -> identical state ids
    interval_sensitivity: dict = field(default_factory=dict)  # name->differs
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all([
            self.k_causal_differs,
            self.zero_state_changes_readout,
            self.swapped_state_changes_readout,
            self.truncated_depth_changes_readout,
            self.no_decode_inside_loop,
            self.deterministic_rerun,
            bool(self.interval_sensitivity) and any(self.interval_sensitivity.values()),
        ])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def manifest(extra: dict | None = None) -> dict:
    """Environment manifest every saved result must embed."""
    base = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if extra:
        base.update(extra)
    return base
