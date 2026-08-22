"""Integration: benchmark harness produces sane before/after results."""

from bench.harness import run_scenario
from bench.scenarios import SCENARIOS


def _by_name(name):
    return next(s for s in SCENARIOS if s.name == name)


def test_compiled_reduces_tokens_on_long_research(tmp_path):
    sc = _by_name("long_research")
    base = run_scenario(sc, enabled=False, root=tmp_path)
    comp = run_scenario(sc, enabled=True, root=tmp_path)
    assert comp.final_tokens < base.final_tokens * 0.5
    assert comp.peak_active_tokens < base.peak_active_tokens
    assert comp.quality["constraint_inline"]


def test_repeated_sql_dedups(tmp_path):
    sc = _by_name("repeated_sql")
    comp = run_scenario(sc, enabled=True, root=tmp_path)
    assert comp.duplicates_stubbed >= 5
    base = run_scenario(sc, enabled=False, root=tmp_path)
    assert comp.total_tokens_all_turns < base.total_tokens_all_turns * 0.6


def test_exact_facts_recoverable_after_masking(tmp_path):
    sc = _by_name("exact_facts")
    comp = run_scenario(sc, enabled=True, root=tmp_path)
    assert comp.quality["facts_recoverable"] == 1.0
    # facts must NOT all be inline anymore (they were masked away cheaply)
    assert comp.quality["facts_inline"] < 0.5


def test_constraints_retained_under_compaction(tmp_path):
    sc = _by_name("constraints")
    comp = run_scenario(sc, enabled=True, root=tmp_path)
    assert comp.quality["policy_inline"]
    assert comp.quality["constraint_inline"]


def test_injection_quarantined(tmp_path):
    sc = _by_name("injection")
    comp = run_scenario(sc, enabled=True, root=tmp_path)
    assert comp.quality["payload_not_promoted"]
    assert comp.quality["payload_quarantined_or_wrapped"]


def test_baseline_mode_is_pure_passthrough(tmp_path):
    sc = _by_name("constraints")
    base = run_scenario(sc, enabled=False, root=tmp_path)
    assert base.stub_observation_tokens == 0
    assert base.failopen_inline == 0
