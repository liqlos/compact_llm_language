"""Exact-tokenizer tests: skipped automatically if tiktoken is unavailable."""

import pytest

from rcc.tokens import available_encoders, count_tokens, count_tokens_exact

tiktoken = pytest.importorskip("tiktoken")

SAMPLE_STRUCTURED = "id|user|status|amount|ts\n1|u1|paid|13.00|2026-07-01"
SAMPLE_PROSE = "The quick brown fox jumps over the lazy dog near the river bank today."


def test_exact_counts_deterministic():
    assert count_tokens_exact(SAMPLE_PROSE) == count_tokens_exact(SAMPLE_PROSE)


def test_approx_within_sane_ratio_of_exact():
    for text in (SAMPLE_PROSE, SAMPLE_STRUCTURED, "x" * 1000, "2024-05-09 v9.2.1"):
        approx = count_tokens(text)
        exact = count_tokens_exact(text)
        ratio = approx / exact
        assert 0.4 < ratio < 2.5, f"{text[:30]!r}: approx={approx} exact={exact}"


def test_structured_costs_more_than_prose_exactly():
    assert count_tokens_exact(SAMPLE_STRUCTURED) > count_tokens_exact(SAMPLE_PROSE) // 2


def test_available_encoders_lists_o200k():
    assert "o200k_base" in available_encoders()


def test_compiled_beats_baseline_under_exact_counting(tmp_path):
    from bench.harness import run_scenario
    from bench.scenarios import SCENARIOS

    sc = next(s for s in SCENARIOS if s.name == "long_research")
    base = run_scenario(sc, enabled=False, root=tmp_path,
                        tokenizer=count_tokens_exact)
    comp = run_scenario(sc, enabled=True, root=tmp_path,
                        tokenizer=count_tokens_exact)
    assert comp.final_tokens < base.final_tokens * 0.6
    assert comp.peak_active_tokens < base.peak_active_tokens * 0.6
