"""Bench harness self-check mechanics (runner/metrics/tasks)."""

import json

from latent_lab.backends.mock import MockHybridBackend
from latent_lab.bench.metrics import manifest
from latent_lab.bench.runner import evaluate, save_report, selfcheck
from latent_lab.bench.tasks import TASKS, MultiHopChain


def test_selfcheck_passes_on_mock():
    rep = selfcheck(MockHybridBackend())
    assert rep.ok, rep.to_dict()
    d = rep.to_dict()
    # every anti-cheat mechanic is exercised and green on the mock
    assert d["k_causal_differs"] and d["zero_state_changes_readout"]
    assert d["swapped_state_changes_readout"]
    assert d["truncated_depth_changes_readout"]


def test_tasks_have_exact_expected_answers():
    for t in TASKS:
        assert t.expected
        assert t.facts  # evidence exists even if backend cannot use it yet


def test_multi_hop_chain_builder_rejects_short():
    import pytest

    with pytest.raises(ValueError):
        MultiHopChain.make("x", ("a",), {"a": "b"})


def test_evaluate_records_unscored_honestly():
    out = evaluate(
        MockHybridBackend(), TASKS, k_steps=2,
        interval=None, answer_fn=None, scorer=lambda a, t: 0.0,
    )
    assert len(out) == len(TASKS)
    assert all(m.success is None for m in out)
    assert all(m.error for m in out)          # recorded, not faked
    assert all(m.decode_calls_in_loop == 0 for m in out)


def test_save_report_embeds_manifest(tmp_path):
    p = tmp_path / "r.json"
    save_report({"x": 1}, p)
    d = json.loads(p.read_text())
    assert "manifest" in d and "python" in d["manifest"]
    assert isinstance(manifest(), dict)
