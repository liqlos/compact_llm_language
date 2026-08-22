"""Structured telemetry — operational events only.

Rules enforced here:
- Telemetry carries phase/action/subject/status plus small scalar details.
  It NEVER decodes hidden reasoning and NEVER carries slot contents.
- Telemetry objects are deliberately NOT accepted anywhere in the cognition
  input path (ProblemInput/RCCModelState); the type system is the guard.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TelemetryEvent:
    ts: float
    phase: str          # "latent_probe" | "latent_loop" | "bench" | ...
    action: str         # "benchmark_interval" | "step" | "halt" | ...
    subject: str        # "layers_16_24" | task id | ...
    status: str         # "running" | "ok" | "error" | "aborted"
    detail: dict = field(default_factory=dict)

    def human(self) -> str:
        """One short operator-facing line (UI may localize)."""
        return f"{self.phase}: {self.action} {self.subject} -> {self.status}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class TelemetrySink:
    """Append-only sink; jsonl file optional. Never read by the model."""

    def __init__(self) -> None:
        self._events: list[TelemetryEvent] = []

    def emit(self, phase: str, action: str, subject: str, status: str,
             **detail) -> TelemetryEvent:
        ev = TelemetryEvent(
            ts=time.time(), phase=phase, action=action,
            subject=subject, status=status, detail=dict(detail),
        )
        self._events.append(ev)
        return ev

    def events(self) -> list[TelemetryEvent]:
        return list(self._events)

    def dump_jsonl(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(ev.to_json() + "\n" for ev in self._events)
