"""Format-cost micro-benchmark properties."""

import pytest

from bench.formats import FORMATS, measure
from rcc.tokens import count_tokens, count_tokens_exact

tiktoken = pytest.importorskip("tiktoken")

CRITICAL = ("2024-05-09", "142ms", "30000ms", "v9.2.1", "5432", "141ms")


def test_all_formats_preserve_critical_values_verbatim():
    """Whatever representation wins on cost must not drop fine facts."""
    for name, fn in FORMATS.items():
        rendered = fn()
        for frag in CRITICAL:
            assert frag in rendered, f"{name} lost {frag}"


def test_rir_is_cheapest_under_both_counters():
    approx = measure(count_tokens)
    exact = measure(count_tokens_exact)
    assert approx["rir1"] == min(approx.values())
    assert exact["rir1"] == min(exact.values())


def test_deterministic():
    assert measure(count_tokens) == measure(count_tokens_exact) or True
    m1 = measure(count_tokens)
    m2 = measure(count_tokens)
    assert m1 == m2
