"""Telemetry isolation: operational events never re-enter cognition."""

import json

import pytest

from latent_lab.backends.mock import MockHybridBackend
from latent_lab.protocols import ProblemInput
from latent_lab.recurrence import run_latent_loop
from latent_lab.telemetry import TelemetrySink


def test_events_are_operational_only():
    sink = TelemetrySink()
    b = MockHybridBackend()
    prob = ProblemInput("t", (tuple(i / 8 for i in range(8)),))
    run_latent_loop(b, b.contextualize(prob), k_steps=2, sink=sink)
    for ev in sink.events():
        d = json.loads(ev.to_json())
        assert set(d) == {"ts", "phase", "action", "subject", "status", "detail"}
        assert "hidden" not in ev.human().lower()
        # no workspace contents leak into telemetry
        for v in d["detail"].values():
            assert not isinstance(v, list) or len(str(v)) < 200


def test_telemetry_cannot_be_fed_to_contextualize():
    """Type system is the guard: ProblemInput takes vectors, not strings."""
    b = MockHybridBackend()
    with pytest.raises((TypeError, ValueError)):
        b.contextualize("telemetry text that must never become model input")


def test_loop_result_does_not_expose_workspace_text():
    b = MockHybridBackend()
    prob = ProblemInput("t", (tuple(i / 8 for i in range(8)),))
    r = run_latent_loop(b, b.contextualize(prob), k_steps=1)
    payload = repr(r.final_state.workspace)[:400]
    assert "ANSWER" not in payload          # no decoded text anywhere in state
    assert r.final_state.runtime == "mock"  # identity header, not content


def test_sink_dump_roundtrip(tmp_path):
    sink = TelemetrySink()
    sink.emit("latent_probe", "check", "x", "ok", n=1)
    p = tmp_path / "tel.jsonl"
    sink.dump_jsonl(p)
    lines = p.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["action"] == "check"
