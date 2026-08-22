"""Mock-backend recurrence loop: no-decode contract, causality, controller."""

import pytest

from latent_lab.backends.mock import MockHybridBackend
from latent_lab.controller import RuleController
from latent_lab.intervals import LayerInterval, candidate_intervals, natural_groups
from latent_lab.protocols import ProblemInput
from latent_lab.recurrence import (
    DecodeInsideLatentLoopError,
    run_latent_loop,
)
from latent_lab.telemetry import TelemetrySink


@pytest.fixture
def prob():
    vec = tuple(i / 9 for i in range(8))
    return ProblemInput("p", (vec,), ("blob-1", "blob-2"))


def test_fixed_depth_loop_no_decode(prob):
    b = MockHybridBackend(LayerInterval(0, 4, "early"))
    st = b.contextualize(prob)
    r = run_latent_loop(b, st, k_steps=4)
    assert r.steps_executed == 4
    assert b.decode_calls == 0          # nothing decoded anywhere


def test_decode_inside_loop_is_detected(prob):
    class DecodingBackend(MockHybridBackend):
        def latent_step(self, state):
            self._decode_calls += 1     # simulate hidden token generation
            return super().latent_step(state)

    b = DecodingBackend()
    st = b.contextualize(prob)
    with pytest.raises(DecodeInsideLatentLoopError):
        run_latent_loop(b, st, k_steps=3)


def test_k_zero_versus_positive_differs_causally(prob):
    b = MockHybridBackend()
    st = b.contextualize(prob)
    r0 = run_latent_loop(b, st, k_steps=0)
    r3 = run_latent_loop(b, st, k_steps=3)
    assert r0.final_state.workspace.fingerprint() != \
        r3.final_state.workspace.fingerprint()


def test_interval_choice_changes_result(prob):
    outs = []
    for iv in (LayerInterval(0, 4, "early"), LayerInterval(4, 8, "output")):
        b = MockHybridBackend(iv)
        st = b.contextualize(prob)
        r = run_latent_loop(b, st, k_steps=2)
        outs.append(r.final_state.workspace.fingerprint())
    assert outs[0] != outs[1]


def test_controller_halts_on_confidence(prob):
    sink = TelemetrySink()
    b = MockHybridBackend()
    ctl = RuleController(tau=0.01, max_steps=10, sink=sink)  # halts immediately
    st = b.contextualize(prob)
    r = run_latent_loop(b, st, k_steps=10, controller=ctl, sink=sink)
    assert r.halted and r.steps_executed == 1
    assert any(e.action == "halt" for e in sink.events())


def test_determinism_same_input_same_state(prob):
    b1, b2 = MockHybridBackend(), MockHybridBackend()
    r1 = run_latent_loop(b1, b1.contextualize(prob), k_steps=2)
    r2 = run_latent_loop(b2, b2.contextualize(prob), k_steps=2)
    assert r1.final_state.state_id() == r2.final_state.state_id()


def test_natural_groups_on_hybrid_layout():
    lt = ("gdn", "gdn", "gdn", "attn") * 2
    assert natural_groups(lt) == [(0, 4), (4, 8)]
    # 2-group layout: middle == output_side, so dedup leaves 3 candidates
    ivs = candidate_intervals(lt)
    names = {iv.name for iv in ivs}
    assert {"early", "full_decoder_control"} <= names
    spans = {(iv.lo, iv.hi) for iv in ivs}
    assert len(spans) == len(ivs)          # no duplicate candidates


def test_negative_k_rejected(prob):
    b = MockHybridBackend()
    with pytest.raises(ValueError):
        run_latent_loop(b, b.contextualize(prob), k_steps=-1)
