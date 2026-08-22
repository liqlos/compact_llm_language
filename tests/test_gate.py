"""Break-even gate (Phase 7) + compressor protocol (Phase 5.3) tests."""

import pytest

from rcc import Policy, RawStore, ResearchSession
from rcc.gate import NullCompressor, SafeCompressor, Usage, break_even

BIG = "g" * 1200  # ~300 approx tokens


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "s")


def _u(**kw):
    fields = {
        "content_tokens": 300, "stub_tokens": 15, "remaining_turns": 20,
        "expected_reads_per_turn": 0.05, "read_slice_tokens": 400,
        "expand_overhead_tokens": 50,
    }
    fields.update(kw)
    return Usage(**fields)


def test_masks_when_offload_clearly_wins():
    d = break_even(_u())
    assert d.mask and d.offload_cost < d.window_cost


def test_keeps_when_reread_every_turn():
    d = break_even(_u(expected_reads_per_turn=1.0))
    assert not d.mask


def test_keeps_at_zero_horizon():
    assert not break_even(_u(remaining_turns=0)).mask


def test_boundary_monotonicity():
    """Raising q must never flip keep->mask."""
    prev = True
    for q100 in range(0, 101, 10):
        d = break_even(_u(expected_reads_per_turn=q100 / 100))
        if not d.mask:
            prev = False
        else:
            assert prev  # once it flips to mask=False it stays False


def test_cache_penalty_pushes_towards_masking():
    mild = break_even(_u(cache_penalty=0.0))
    harsh = break_even(_u(cache_penalty=0.5))
    assert harsh.window_cost > mild.window_cost


# ---- session wiring ----------------------------------------------------------

def _session(store, policy):
    s = ResearchSession("run-g", store, policy)
    oid = s.observe("big", BIG)
    for i in range(5):
        s.observe(f"f{i}", BIG + str(i))
    return s, oid


def test_gate_breakeven_still_masks_bulky_old_obs(store):
    s, _ = _session(store, Policy(gate="breakeven"))
    c = s.compile()
    assert c.metrics.observations_masked >= 1


def test_gate_breakeven_never_masks_hot_data(store):
    s, _ = _session(store, Policy(gate="breakeven", expected_reads_per_turn=1.0,
                                  min_mask_tokens=150))
    c = s.compile()
    assert c.metrics.observations_masked == 0   # everything re-read every turn


def test_window_and_breakeven_agree_on_default_q(store):
    """With conservative default expected-reads both gates mask identically."""
    sw, _ = _session(store, Policy())
    sb, _ = _session(store, Policy(gate="breakeven"))
    cw, cb = sw.compile(), sb.compile()
    assert cw.metrics.observations_masked >= 1
    assert cw.metrics.observations_masked == cb.metrics.observations_masked


# ---- compressor protocol -----------------------------------------------------

class ExplodingCompressor:
    def compress(self, text):
        raise RuntimeError("provider down")


class GreedyCompressor:
    def compress(self, text):
        return text[:10]


def test_null_compressor_is_noop():
    assert NullCompressor().compress("abc") is None


def test_safe_compressor_fails_open():
    assert SafeCompressor(ExplodingCompressor()).compress("abc") is None
    assert SafeCompressor(GreedyCompressor()).compress("abcdef") == "abcdef"
    assert SafeCompressor(NullCompressor()).compress("abc") is None
