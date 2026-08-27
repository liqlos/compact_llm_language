"""Deterministic tests for gold scoring in latent_run.evaluate.

Gold candidate index must be derived from ex.candidates.index(ex.answer) and
rank_of_gold read off the returned order — never assuming candidate zero is
gold. Tests place gold at a nonzero candidate position.
"""

from types import SimpleNamespace

import pytest

from latent_lab.bench.latent_run import evaluate
from latent_lab.bench.suite import Example


class _Ids:
    """Minimal stand-in for a token-id tensor (only .to is used)."""

    def to(self, device):
        return self


class _StubRec:
    """rank_candidates stub returning scripted candidate orders."""

    def __init__(self, orders):
        self.model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cpu")]))
        self.orders = list(orders)

    def rank_candidates(self, input_ids, candidate_ids, k_steps, *,
                        ablate=None, partner_input_ids=None):
        order = self.orders.pop(0)
        scores = [float(len(order) - order.index(i)) for i in range(len(order))]
        return order, scores, None


def _ex(gold_pos, n_cands=4, ex_id="e0"):
    cands = tuple(f"c{i}" for i in range(n_cands))
    return Example(
        ex_id=ex_id, family="fsm", depth=3, prompt="p",
        answer=cands[gold_pos], candidates=cands, content_key="k",
    )


def _data(examples):
    return SimpleNamespace(
        examples=examples,
        prompt_ids=[_Ids() for _ in examples],
        cand_ids=[[_Ids() for _ in ex.candidates] for ex in examples],
    )


def test_gold_at_nonzero_position_top_ranked_is_correct():
    # Gold sits at candidate index 2 and the model ranks it first.
    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[2, 0, 3, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    assert len(res["records"]) == 1
    rec = res["records"][0]
    assert rec["correct"] == 1.0
    assert rec["rank_of_gold"] == 0
    assert rec["n_candidates"] == 4
    assert res["accuracy"] == 1.0


def test_gold_at_nonzero_position_non_top_ranked_is_incorrect():
    # Candidate 0 ranks first (old buggy assumption would score this as a
    # hit) but gold lives at candidate index 2 and lands at rank 1 only.
    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[0, 2, 3, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    rec = res["records"][0]
    assert rec["correct"] == 0.0
    assert rec["rank_of_gold"] == 1
    assert rec["n_candidates"] == 4
    assert res["accuracy"] == 0.0


def test_mixed_batch_aggregates_per_example_gold_position():
    # Two examples, both with gold at nonzero positions: one ranked second,
    # one whose gold lands at the last rank of a full permutation.
    exs = [_ex(gold_pos=1, ex_id="rank1"), _ex(gold_pos=3, ex_id="last")]
    orders = [[3, 1, 0, 2], [2, 1, 0, 3]]
    res = evaluate(_StubRec(orders), _data(exs), k_steps=1,
                   indices=[0, 1], tag="t")
    by_id = {r["ex_id"]: r for r in res["records"]}
    assert by_id["rank1"]["rank_of_gold"] == 1
    assert by_id["rank1"]["correct"] == 0.0
    assert by_id["last"]["rank_of_gold"] == 3
    assert by_id["last"]["correct"] == 0.0
    assert res["accuracy"] == 0.0

    # Same batch, orders flipped so each gold is ranked top: all correct.
    res2 = evaluate(_StubRec([[1, 3, 0, 2], [3, 0, 1, 2]]),
                    _data(exs), k_steps=1, indices=[0, 1], tag="t")
    assert res2["accuracy"] == 1.0
    assert all(r["rank_of_gold"] == 0 for r in res2["records"])


def test_records_retain_raw_scores_and_are_independently_rescorable():
    """Lossless raw evidence: scores_raw + score_order + candidate/gold
    identity must survive, and accuracy must be recomputable after an
    arbitrary consistent candidate permutation."""
    from latent_lab.bench.latent_run import rescore_records

    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[0, 3, 2, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    rec = res["records"][0]
    assert rec["scores_raw"] == [4.0, 1.0, 2.0, 3.0]
    assert rec["score_order"] == [0, 3, 2, 1]
    assert rec["candidates"] == ["c0", "c1", "c2", "c3"]
    assert rec["answer"] == "c2"
    assert rec["gold_candidate_index"] == 2

    # independent rescore agrees without touching derived fields first
    assert rescore_records([rec]) == 0.0

    # permute the candidate set consistently (scores/order/gold move with
    # the candidates): rescored accuracy is IDENTICAL — raw evidence is
    # sufficient to redo scoring under any corrected scorer
    m = {j: p for p, j in enumerate([3, 1, 0, 2])}  # old idx -> new position
    moved = {"candidates": [None] * 4, "scores_raw": [None] * 4}
    for j in range(4):
        moved["candidates"][m[j]] = rec["candidates"][j]
        moved["scores_raw"][m[j]] = rec["scores_raw"][j]
    moved["answer"] = rec["answer"]
    moved["gold_candidate_index"] = m[rec["gold_candidate_index"]]
    moved["score_order"] = [m[i] for i in rec["score_order"]]
    merged = {**rec, **moved}
    assert rescore_records([merged]) == rescore_records([rec])


def test_rescore_rejects_tampered_derived_fields():
    from latent_lab.bench.latent_run import rescore_records

    ex = _ex(gold_pos=2)
    res = evaluate(_StubRec([[0, 3, 2, 1]]), _data([ex]), k_steps=1,
                   indices=[0], tag="t")
    rec = dict(res["records"][0])
    assert rec["correct"] == 0.0
    tampered = {**rec, "correct": 1.0}  # claim correct against raw evidence
    import pytest
    with pytest.raises(ValueError):
        rescore_records([tampered])
    tampered_order = {**rec, "score_order": [1, 0, 2, 3]}
    with pytest.raises(ValueError):
        rescore_records([tampered_order])


# ---------------------------------------------------------------------------
# gold identity is re-derived from answer/candidates — a supplied index is
# never trusted (missing / duplicated / substituted gold fails closed)
# ---------------------------------------------------------------------------

def _gold_pos2_record():
    ex = _ex(gold_pos=2)
    return dict(evaluate(_StubRec([[0, 3, 2, 1]]), _data([ex]), k_steps=1,
                         indices=[0], tag="t")["records"][0])


def test_rescore_rejects_substituted_gold_index_even_when_self_consistent():
    """Stealthy substitution: the persisted index names a DIFFERENT
    candidate while rank_of_gold/correct/accuracy are rewritten to match
    it — index-trusting scoring would accept this; deriving gold from
    answer/candidates must not."""
    import pytest

    from latent_lab.bench.latent_run import rescore_records

    rec = _gold_pos2_record()          # candidates c0..c3, answer c2
    assert rec["answer"] == "c2"
    rec["gold_candidate_index"] = 0    # substitute the top-scored rival
    rec["rank_of_gold"] = 0            # ... and stay internally consistent
    rec["correct"] = 1.0
    with pytest.raises(ValueError, match="gold_candidate_index"):
        rescore_records([rec])


def test_rescore_rejects_missing_and_duplicated_gold():
    import pytest

    from latent_lab.bench.latent_run import rescore_records

    rec = _gold_pos2_record()
    missing = {**rec, "answer": "nope"}
    with pytest.raises(ValueError, match="missing from"):
        rescore_records([missing])

    dup = {**_gold_pos2_record(),
           "candidates": ["c0", "c1", "c2", "c2"],
           "scores_raw": [4.0, 1.0, 3.0, 2.0],
           "score_order": [0, 2, 3, 1]}
    with pytest.raises(ValueError, match="duplicated"):
        rescore_records([dup])


def test_rescore_rejects_missing_answer_field_and_bad_index_types():
    import pytest

    from latent_lab.bench.latent_run import rescore_records

    rec = _gold_pos2_record()
    no_answer = {k: v for k, v in rec.items() if k != "answer"}
    with pytest.raises(ValueError, match="missing from"):
        rescore_records([no_answer])

    bad_idx = {**rec, "gold_candidate_index": "2"}
    with pytest.raises(ValueError, match="gold_candidate_index"):
        rescore_records([bad_idx])


def _v3_ex(ex_id="v3-0"):
    return SimpleNamespace(
        ex_id=ex_id, family="fsm", prompt="v3 prompt", answer="B",
        candidates=("A", "B", "C"), candidate_permutation_seed=71,
        candidate_permutation=(2, 0, 1),
    )


def _v3_compute():
    return {
        "prefill_layers": 12,
        "recurrence_interval_applications": 4,
        "k_loops": 2,
        "candidate_tail_layers": 9,
        "lm_head_calls": 3,
        "tokenizer_calls": 4,
        "decode_calls": 0,
        "wall_seconds": 0.1,
        "peak_memory_bytes": None,
        "successful_task": True,
    }


def _v3_details(rows=None):
    rows = rows or ((-0.8,), (-0.2, -0.2), (-1.5,))
    sums = [sum(row) for row in rows]
    normalized = [total / len(row) for total, row in zip(sums, rows)]
    return {
        "candidate_token_logprobs": [list(row) for row in rows],
        "candidate_token_counts": [len(row) for row in rows],
        "candidate_raw_sum_logprobs": sums,
        "candidate_length_normalized_logprobs": normalized,
        "primary_score_definition": "mean_candidate_token_logprob_v1",
        "primary_scores": normalized,
        "exact_top_tie_indices": [],
        "compute": _v3_compute(),
    }


def _v3_identity():
    return {
        "run_id": "run-v3",
        "recipe_hash": "a" * 64,
        "model_id": "tiny-model",
        "model_revision": "d" * 40,
        "adapter_id": "adapter-1",
        "checkpoint_id": "step-10",
        "checkpoint_content_hash": "b" * 64,
        "suite_id": "behavioral-v3",
        "suite_version": 3,
        "suite_hash": "c" * 64,
        "recurrence_config": {
            "interval": [2, 4], "gradient_semantics": "truncated_cache",
        },
    }


class _V3Rec:
    def __init__(self, details):
        self.model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cpu")]))
        self.details = details

    def rank_candidates(self, *args, **kwargs):
        normalized = self.details["candidate_length_normalized_logprobs"]
        order = sorted(range(len(normalized)), key=lambda i: -normalized[i])
        return order, self.details["candidate_raw_sum_logprobs"], \
            SimpleNamespace(extra=self.details)


def test_v3_runtime_converter_online_offline_and_selector_are_one_path():
    from latent_lab.bench.eval_v3 import checkpoint_history_entry
    from latent_lab.bench.latent_run import (
        evaluate_v3, rescore_v3_records, select_v3_checkpoint)

    ex = _v3_ex()
    result = evaluate_v3(
        _V3Rec(_v3_details()), _data([ex]), 2, [0],
        evidence_identity=_v3_identity(), split="validation", tag="v3")
    record = result["records"][0]
    assert record["predicted_answer"] == "B"
    assert record["correctness"] is True
    assert result["metrics"] == rescore_v3_records([record])
    selected = select_v3_checkpoint([
        {"step": 1, "accuracy": 1.0},
        checkpoint_history_entry(10, [record]),
    ])
    assert selected.step == 10
    assert selected.n_rejected == 1


def test_v3_runtime_converter_rejects_online_offline_disagreement():
    from latent_lab.bench.latent_run import build_v3_runtime_record

    details = _v3_details()
    details["candidate_raw_sum_logprobs"][1] = -999.0
    with pytest.raises(ValueError, match="raw_sum"):
        build_v3_runtime_record(
            _v3_ex(), split="validation", k_steps=2,
            evidence_identity=_v3_identity(), details=details,
            runtime_order=[1, 0, 2], runtime_raw_sums=[-0.8, -0.4, -1.5])


def test_v3_runtime_converter_persists_exact_top_tie_as_error():
    from latent_lab.bench.latent_run import build_v3_runtime_record

    details = _v3_details(rows=((-0.4,), (-0.4, -0.4), (-1.5,)))
    details["exact_top_tie_indices"] = [0, 1]
    record = build_v3_runtime_record(
        _v3_ex(), split="validation", k_steps=2,
        evidence_identity=_v3_identity(), details=details,
        runtime_raw_sums=[-0.4, -0.8, -1.5])
    assert record["status"] == "error"
    assert record["error_status"] == "AMBIGUOUS_TOP_TIE"
    assert record["predicted_answer"] is None


def test_v3_runtime_converter_rejects_false_tie_claim():
    from latent_lab.bench.latent_run import build_v3_runtime_record

    details = _v3_details()
    details["exact_top_tie_indices"] = [0, 1]
    with pytest.raises(ValueError, match="tie claim disagrees"):
        build_v3_runtime_record(
            _v3_ex(), split="validation", k_steps=2,
            evidence_identity=_v3_identity(), details=details,
            runtime_raw_sums=[-0.8, -0.4, -1.5])


def test_evaluate_v3_consumes_runtime_ambiguous_top_tie_shape():
    from latent_lab.bench.latent_run import evaluate_v3

    try:
        from latent_lab.backends.localized import AmbiguousTopTie
    except ImportError:  # pre-runtime-integration branch compatibility
        class AmbiguousTopTie(RuntimeError):
            def __init__(self, message, *, candidate_details=(), report=None):
                super().__init__(message)
                self.candidate_details = tuple(candidate_details)
                self.report = report
                self.details = dict(report.extra)
                self.raw_sums = tuple(
                    detail.raw_sum_logprob for detail in candidate_details)

    details = _v3_details(rows=((-0.4,), (-0.4, -0.4), (-1.5,)))
    details["exact_top_tie_indices"] = [0, 1]
    report = SimpleNamespace(extra=details)
    candidate_details = tuple(
        SimpleNamespace(raw_sum_logprob=value)
        for value in details["candidate_raw_sum_logprobs"])

    class _TieRec:
        model = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cpu")]))

        def rank_candidates(self, *args, **kwargs):
            raise AmbiguousTopTie(
                "exact top tie", candidate_details=candidate_details,
                report=report)

    result = evaluate_v3(
        _TieRec(), _data([_v3_ex()]), 2, [0],
        evidence_identity=_v3_identity(), split="validation", tag="v3")
    assert result["records"][0]["error_status"] == "AMBIGUOUS_TOP_TIE"
    assert result["metrics"]["nontermination_error_rate"] == 1.0


def test_v3_checkpoint_selection_recomputes_raw_history():
    from latent_lab.bench.latent_run import (
        build_v3_runtime_record, canonical_v3_history_entry,
        select_v3_checkpoint_from_raw_history,
        selected_v3_adapter_state_sha256)

    losing = build_v3_runtime_record(
        _v3_ex("v3-losing"), split="validation", k_steps=2,
        evidence_identity=_v3_identity(),
        details=_v3_details(rows=((-0.1,), (-1.0,), (-2.0,))),
        runtime_order=[0, 1, 2], runtime_raw_sums=[-0.1, -1.0, -2.0])
    winning_identity = {**_v3_identity(), "checkpoint_id": "step-20",
                        "checkpoint_content_hash": "e" * 64}
    winning = build_v3_runtime_record(
        _v3_ex("v3-winning"), split="validation", k_steps=2,
        evidence_identity=winning_identity, details=_v3_details(),
        runtime_order=[1, 0, 2], runtime_raw_sums=[-0.8, -0.4, -1.5])
    history = [
        canonical_v3_history_entry(10, [losing]),
        canonical_v3_history_entry(20, [winning]),
    ]
    expected_identity = {
        "run_id": "run-v3",
        "recipe_hash": "a" * 64,
        "model_id": "tiny-model",
        "model_revision": "d" * 40,
        "adapter_id": "adapter-1",
        "suite_id": "behavioral-v3",
        "suite_version": 3,
        "suite_hash": "c" * 64,
        "split": "validation",
        "k": 2,
        "recurrence_config": {
            "interval": [2, 4], "gradient_semantics": "truncated_cache",
        },
    }
    selected = select_v3_checkpoint_from_raw_history(
        history, expected_identity=expected_identity)
    assert selected.step == 20
    assert selected.metric == 1.0
    assert selected_v3_adapter_state_sha256(
        history, expected_step=20, expected_metric=1.0) == "e" * 64

    with pytest.raises(ValueError, match="run_id disagrees"):
        select_v3_checkpoint_from_raw_history(
            history,
            expected_identity={**expected_identity, "run_id": "other-run"})

    history[0]["metrics"]["micro_accuracy"] = 1.0
    with pytest.raises(ValueError, match="disagrees with raw"):
        select_v3_checkpoint_from_raw_history(history)
    with pytest.raises(ValueError, match="no raw v3 records"):
        select_v3_checkpoint_from_raw_history([
            {"step": 1, "metrics": history[1]["metrics"]},
        ])


def test_cmd_eval_wiring_and_persisted_envelope_are_v3(tmp_path):
    import inspect
    import json

    from latent_lab.bench.artifacts import validate_eval
    from latent_lab.bench.latent_run import (
        _train_inner, build_v3_eval_payload, cmd_eval, evaluate_v3)

    source = inspect.getsource(cmd_eval)
    assert "evaluate_v3(" in source
    assert "build_v3_eval_payload(" in source
    assert "rescore_records(" not in source
    assert source.index("state = load_adapter_bundle(") < source.index(
        "    validate_selected_adapter_state_binding(") < source.index(
            "model, tok = load_model(")
    train_source = inspect.getsource(_train_inner)
    assert "latent_lab.bench.suite_v3" in train_source
    assert "select_v3_checkpoint_from_raw_history(" in train_source
    assert "evaluate_v3(" in train_source
    assert "selected_adapter_state_sha256" in train_source

    result = evaluate_v3(
        _V3Rec(_v3_details()), _data([_v3_ex()]), 2, [0],
        evidence_identity=_v3_identity(), split="test_id",
        tag="E-localized|test_id|clean|K=2")
    cfg = {
        "mode": "E-localized", "model": "tiny-model",
        "revision": "d" * 40, "suite_sha256": "c" * 64,
        "interval": [2, 4], "max_k": 8, "k": 2, "seed": 7,
    }
    payload = build_v3_eval_payload(
        adapter="runs/tiny", split="test_id", config=cfg,
        model_id="tiny-model", revision="d" * 40,
        suite_hash="c" * 64, tokenizer_class="TinyTokenizer",
        interval=[2, 4], k_steps=2, ablation=None, seed=7,
        checkpoint_content_digest="b" * 64, result=result, device="cpu")
    path = tmp_path / "eval_v3.json"
    path.write_text(json.dumps(payload))
    loaded = validate_eval(path)
    record = loaded["results"]["clean"]["records"][0]
    assert record["schema_version"] == "latent_eval.v3"
    assert loaded["results"]["clean"]["metrics"] == result["metrics"]
