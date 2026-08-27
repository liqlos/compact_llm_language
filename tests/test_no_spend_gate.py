"""Focused tests for the bounded no-spend integrity gate.

Covers the acceptance properties that must hold BEFORE any GPU spend:

* corrected gold-aware scorer FAIL-CLOSED contract: exactly one raw
  representation, full unique permutations only, finite real scores
  (NaN/Inf invalid, never sunk to -inf), top-score ties between different
  candidates ambiguous (array position never decides), duplicated candidate
  sets invalid, record flags/status preserved through aggregation;
* deterministic corrected checkpoint selection with non-negative integer
  steps and finite metrics;
* artifact discovery/hash/duplicate detection;
* train-report schema blockers (missing/non-finite fields are blockers,
  not defaults) and strict JSON everywhere (NaN literals fail closed);
* safe identity-bound checkpoint classification with FP32 validated from
  the bundle PAYLOAD (bf16 bundles must NOT READY);
* symmetric discovery: orphan checkpoints / orphan eval files /
  duplicate checkpoint bindings are explicit blockers;
* eval->checkpoint joins: exact model/revision/suite binding and FULL
  test_id + test_ood coverage per retained loadable checkpoint; one
  unrelated eval proves nothing;
* 4B quarantine completeness: any known-invalid or byte-duplicate live
  artifact blocks READY even when a sibling file differs; marker-only
  empty quarantine proves nothing;
* integrity: --out/input-root overlap rejection before writing, streaming
  before/after source fingerprints proving inputs unchanged, exit codes
  (0 READY / 1 NOT_READY / 2 execution error), canonical byte-stability
  without wall-clock values.
"""

from __future__ import annotations

import json
import math
import os
import re
from types import SimpleNamespace

import pytest

from latent_lab.bench import no_spend_gate as g
from latent_lab.bench.corrected_scoring import (
    FLAG_AMBIGUOUS_TIE,
    FLAG_CONFLICTING_INPUTS,
    FLAG_DUPLICATE_CANDIDATES,
    FLAG_GOLD_ABSENT,
    FLAG_NONFINITE_SCORES,
    FLAG_NORMALIZED_MATCH,
    FLAG_ORDER_NOT_PERMUTATION,
    CURRENT_EVIDENCE_ELIGIBLE,
    INVALID_RECORDS,
    LEGACY_EVIDENCE_SCOPE,
    LEGACY_RAW_RESCORED,
    MISSING_RAW_PREDICTION,
    corrected_score,
    normalize_answer,
    rescore_records,
    select_best_checkpoint,
)

ISO_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _strict_loads(blob: bytes) -> object:
    """Strict JSON parse: rejects NaN/Infinity tokens entirely."""
    return json.loads(
        blob,
        parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"),
    )


# ---------------------------------------------------------------------------
# legacy scorer salvage/classification fail-closed contract
# ---------------------------------------------------------------------------

class TestCorrectedScorer:
    def test_gold_not_at_index_zero_top_rank_is_correct(self):
        cands = ("a", "b", "gold", "c")
        cs = corrected_score(cands, "gold", order=[2, 0, 3, 1])
        assert cs.valid and cs.correct and cs.rank_of_gold == 0

    def test_gold_not_at_index_zero_other_ranked_first_incorrect(self):
        cands = ("a", "b", "gold", "c")
        cs = corrected_score(cands, "gold", order=[0, 2, 3, 1])
        assert cs.valid and not cs.correct and cs.rank_of_gold == 1

    def test_permutation_invariance_with_scores_attached_to_content(self):
        cands = ("a", "b", "gold", "c")
        scores = [0.1, 0.4, 0.9, -0.3]
        base = corrected_score(cands, "gold", scores=scores)
        perm = [3, 1, 0, 2]
        pc = [cands[i] for i in perm]
        ps = [scores[i] for i in perm]
        moved = corrected_score(pc, "gold", scores=ps)
        assert base.valid and moved.valid
        assert base.correct == moved.correct
        assert base.rank_of_gold == moved.rank_of_gold == 0
        # and a losing permutation of the same mapping stays losing
        scores2 = [0.9, 0.4, 0.1, -0.3]
        lost = corrected_score(cands, "gold", scores=scores2)
        assert not lost.correct and lost.rank_of_gold == 2

    def test_conflicting_order_and_scores_rejected(self):
        cs = corrected_score(("a", "gold"), "gold",
                             order=[1, 0], scores=[0.1, 0.9])
        assert not cs.valid
        assert FLAG_CONFLICTING_INPUTS in cs.flags

    def test_missing_representation_invalid_not_exception(self):
        cs = corrected_score(("a", "gold"), "gold")
        assert not cs.valid

    def test_duplicate_candidates_invalid_never_silently_scored(self):
        cands = ("gold", "b", "gold", "c")
        for order in ([2, 0, 3, 1], [1, 2, 3, 0]):
            cs = corrected_score(cands, "gold", order=order)
            assert not cs.valid, order
            assert FLAG_DUPLICATE_CANDIDATES in cs.flags
            assert not cs.correct and cs.rank_of_gold == -1

    def test_nonfinite_scores_invalid_never_sunk_to_minus_inf(self):
        nan, inf = float("nan"), float("inf")
        for bad in ([nan, 0.5], [0.5, inf], [-inf, 0.5]):
            cs = corrected_score(("a", "gold"), "gold", scores=bad)
            assert not cs.valid, bad
            assert FLAG_NONFINITE_SCORES in cs.flags
            assert not cs.correct and cs.rank_of_gold == -1

    def test_top_tie_between_different_candidates_ambiguous_regardless_of_position(
            self):
        # equal top score on two different candidates: array position must
        # NEVER decide — both permutations are invalid.
        cs1 = corrected_score(("a", "gold"), "gold", scores=[0.7, 0.7])
        cs2 = corrected_score(("gold", "a"), "gold", scores=[0.7, 0.7])
        for cs in (cs1, cs2):
            assert not cs.valid
            assert FLAG_AMBIGUOUS_TIE in cs.flags
            assert not cs.correct and cs.rank_of_gold == -1

    def test_three_way_tie_also_invalid(self):
        cs = corrected_score(("x", "y", "z"), "y", scores=[1.0, 1.0, 1.0])
        assert not cs.valid and FLAG_AMBIGUOUS_TIE in cs.flags

    def test_scores_length_mismatch_invalid(self):
        cs = corrected_score(("a", "gold"), "gold", scores=[1.0])
        assert not cs.valid and not cs.correct

    def test_scores_must_be_real_numbers_not_bools_or_strings(self):
        for junk in (["true", 0.5], ["0.9", 0.1], [None, 0.5],
                     [True, False]):
            cs = corrected_score(("a", "gold"), "gold", scores=junk)
            assert not cs.valid, junk

    def test_partial_or_unknown_order_invalid_not_raised(self):
        for order in ([0, 0], [0, 1, 2], [0], [5, 1], [0, True],
                      "01", [0.0, 1.0]):
            cs = corrected_score(("a", "b"), "a", order=order)
            assert not cs.valid, order
            assert FLAG_ORDER_NOT_PERMUTATION in cs.flags

    def test_whitespace_parser_normalization(self):
        assert normalize_answer(" 42 ") == normalize_answer("42")
        assert normalize_answer("new\t york") == "new york"
        cs = corrected_score((" 42", "43"), "42", order=[0, 1])
        assert cs.valid and cs.correct and cs.rank_of_gold == 0
        # internal-whitespace parser difference still matches the gold and
        # is explicitly flagged rather than silently absorbed
        cs2 = corrected_score(("new  york", "boston"), "new\tyork",
                              order=[1, 0])
        assert cs2.valid and not cs2.correct
        assert FLAG_NORMALIZED_MATCH in cs2.flags

    def test_gold_absent_from_candidates_is_decisive_incorrect(self):
        cs = corrected_score(("a", "b"), "zzz", order=[0, 1])
        assert cs.valid and not cs.correct and cs.rank_of_gold == -1
        assert FLAG_GOLD_ABSENT in cs.flags


class TestSelectBestCheckpoint:
    def test_ignores_poisoned_stored_best_and_recomputes(self):
        hist = [{"step": 100, "accuracy": 0.2},
                {"step": 200, "accuracy": 0.9},
                {"step": 300, "accuracy": 0.9}]
        sel = select_best_checkpoint(hist)
        assert sel.step == 200          # earliest among tied best
        assert sel.metric == 0.9
        assert sel.provenance == "recomputed_from_history"

    def test_nan_metrics_rejected_not_skipped_silently(self):
        nan = float("nan")
        hist = [{"step": 100, "accuracy": nan},
                {"step": 200, "accuracy": 0.4}]
        sel = select_best_checkpoint(hist)
        assert sel.n_rejected_nonfinite == 1
        assert sel.step == 200
        assert select_best_checkpoint(
            [{"step": 1, "accuracy": nan}, {"step": 2, "accuracy": nan}]
        ) is None

    def test_bool_float_and_negative_steps_are_rejected_not_coerced(self):
        hist = [{"step": True, "accuracy": 1.0},
                {"step": 100.5, "accuracy": 1.0},
                {"step": -3, "accuracy": 1.0},
                {"step": 200, "accuracy": 0.4}]
        sel = select_best_checkpoint(hist)
        assert sel.step == 200
        assert sel.n_rejected_nonfinite == 3

    def test_bool_metric_rejected(self):
        sel = select_best_checkpoint([{"step": 1, "accuracy": True},
                                      {"step": 2, "accuracy": 0.5}])
        assert sel.step == 2 and sel.n_rejected_nonfinite == 1

    def test_empty_history(self):
        assert select_best_checkpoint([]) is None


class TestRescoreEligibility:
    EX = {"ex_id": "e0", "candidates": ("a", "b", "gold"),
          "answer": "gold"}

    def test_derived_only_records_are_non_rescorable(self):
        recs = [{"ex_id": "e0", "correct": 1.0, "rank_of_gold": 0,
                 "n_candidates": 3}]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == MISSING_RAW_PREDICTION

    def test_raw_candidate_scores_allow_true_rescore(self):
        recs = [
            {"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9]},
            {"ex_id": "e0", "candidate_scores": [0.9, 0.2, 0.1]},
        ]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == LEGACY_RAW_RESCORED
        assert out.corrected_accuracy == pytest.approx(0.5)
        assert out.evidence_scope == LEGACY_EVIDENCE_SCOPE
        assert CURRENT_EVIDENCE_ELIGIBLE is False
        assert out.current_evidence_eligible is False
        assert g.CURRENT_V3_RESCORED != LEGACY_RAW_RESCORED

    def test_nan_score_record_makes_whole_file_INVALID_RECORDS(self):
        recs = [
            {"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9]},
            {"ex_id": "e0", "candidate_scores": [float("nan"), 0.2, 0.1]},
        ]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == INVALID_RECORDS
        assert out.flag_counts.get(FLAG_NONFINITE_SCORES) == 1
        assert out.corrected_accuracy is None

    def test_conflicting_representations_within_record_invalid(self):
        recs = [{"ex_id": "e0", "candidate_scores": [0.1, 0.2, 0.9],
                 "ranked_candidates": ["gold", "a", "b"]}]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == INVALID_RECORDS
        assert FLAG_CONFLICTING_INPUTS in (out.flag_counts or {})

    def test_ranked_candidates_partial_unknown_duplicate_invalid(self):
        good_by_norm = ["gold", "a", "b"]
        cases = {
            "partial": ["gold", "a"],
            "unknown": ["gold", "a", "zzz"],
            "duplicate": ["gold", "gold", "a"],
            "not_a_list": "gold",
        }
        for name, rc in cases.items():
            recs = [{"ex_id": "e0", "ranked_candidates": rc}]
            out = rescore_records(recs, {"e0": self.EX})
            assert out.status == INVALID_RECORDS, name
            assert out.corrected_accuracy is None

    def test_valid_ranked_candidates_rescore(self):
        recs = [{"ex_id": "e0", "ranked_candidates": ["b", "gold", "a"]}]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == LEGACY_RAW_RESCORED
        assert out.corrected_accuracy == pytest.approx(0.0)

    def test_non_dict_records_fail_closed(self):
        out = rescore_records(["nope"], {"e0": self.EX})
        assert out.status == INVALID_RECORDS
        out = rescore_records("records", {"e0": self.EX})
        assert out.status == INVALID_RECORDS

    def test_flags_preserved_through_aggregation(self):
        recs = [{"ex_id": "e0", "predicted_answer": " gold "}]
        out = rescore_records(recs, {"e0": self.EX})
        assert out.status == LEGACY_RAW_RESCORED
        assert out.corrected_accuracy == 1.0

    def test_missing_example_blocks_whole_file(self):
        out = rescore_records([{"ex_id": "nope",
                                "candidate_scores": [1, 2, 3]}],
                              {"e0": self.EX})
        assert out.status == MISSING_RAW_PREDICTION


def _gate_v3_record(ex, *, split="test_id", hit=True,
                    checkpoint_content_hash="b" * 64,
                    record_overrides=None):
    from latent_lab.bench.eval_v3 import build_eval_record, canonical_sha256

    permutation = getattr(ex, "candidate_permutation", None) \
        or tuple(range(len(ex.candidates)))
    permutation_seed = getattr(ex, "candidate_permutation_seed", 0)
    metadata = {
        "run_id": "run-v3",
        "recipe_hash": canonical_sha256(_canonical_recipe()),
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": PINNED_REV_2B,
        "adapter_id": "runs/E_k4_s0",
        "checkpoint_id": "best_params.pt",
        "checkpoint_content_hash": checkpoint_content_hash,
        "suite_id": "behavioral-v3",
        "suite_version": 3,
        "suite_hash": g.current_suite_sha256(),
        "example_id": ex.ex_id,
        "split": split,
        "family": ex.family,
        "prompt": ex.prompt,
        "candidates": ex.candidates,
        "candidate_permutation_seed": permutation_seed,
        "candidate_permutation": permutation,
        "gold_answer": ex.answer,
        "k": 4,
        "recurrence_config": {"interval": [12, 18], "trained_k": 4},
        "compute": {
            "prefill_layers": 12,
            "recurrence_interval_applications": 24,
            "k_loops": 4,
            "candidate_tail_layers": len(ex.candidates) * 6,
            "lm_head_calls": len(ex.candidates),
            "tokenizer_calls": len(ex.candidates) + 1,
            "decode_calls": 0,
            "wall_seconds": 0.1,
            "peak_memory_bytes": None,
            "successful_task": True,
            "eval_ablation": {},
        },
    }
    metadata.update(record_overrides or {})
    scores = [
        ([-0.01]
         if (candidate == metadata["gold_answer"]) == hit else [-2.0])
        for candidate in metadata["candidates"]
    ]
    return build_eval_record(per_token_logprobs=scores, **metadata)


def _gate_eval_payload(record, *, adapter="runs/E_k4_s0", records=None):
    from latent_lab.bench.eval_v3 import aggregate_records
    from latent_lab.bench.latent_run import build_v3_eval_payload

    records = list(records or [record])
    metrics = aggregate_records(records)
    split = record["split"]
    k_steps = record["k"]
    result = {
        "schema_version": metrics["schema_version"],
        "tag": f"E-localized|{split}|clean|K={k_steps}",
        "ablate": {}, "k_steps": k_steps, "n": len(records),
        "metrics": metrics, "seconds": 0.1, "records": records,
    }
    cfg = {
        "mode": "E-localized",
        "model": record["model_identity"]["model_id"],
        "revision": record["model_identity"]["revision"],
        "suite_sha256": record["suite_identity"]["sha256"],
        "interval": [12, 18], "max_k": 16, "k": k_steps, "seed": 0,
    }
    return build_v3_eval_payload(
        adapter=adapter, split=split, config=cfg,
        model_id=record["model_identity"]["model_id"],
        revision=record["model_identity"]["revision"],
        suite_hash=record["suite_identity"]["sha256"],
        tokenizer_class="GateTestTokenizer", interval=[12, 18],
        k_steps=k_steps, ablation=None, seed=0,
        checkpoint_content_digest=record["checkpoint_identity"][
            "content_sha256"], result=result, device="cpu")


def test_gate_recognizes_only_valid_independently_rescored_v3(tmp_path):
    ex = _suite_ex()
    record = _gate_v3_record(ex)
    path = tmp_path / "eval.json"
    payload = _gate_eval_payload(record)
    path.write_text(json.dumps(payload))
    verdict = g.evaluate_eval_file(path, {ex.ex_id: ex}, root_label="2b")
    assert verdict["status"] == g.CURRENT_V3_RESCORED
    assert verdict["record_schema_version"] == "latent_eval.v3"
    assert verdict["evidence_class"] == "VALID_CURRENT"
    assert verdict["corrected_accuracy"] == 1.0
    assert verdict["record_run_id"] == "run-v3"
    assert verdict["record_adapter_id"] == "runs/E_k4_s0"
    assert verdict["checkpoint_content_digest"] == "b" * 64

    payload["results"]["clean"]["records"][0]["correctness"] = False
    path.write_text(json.dumps(payload))
    rejected = g.evaluate_eval_file(path, {ex.ex_id: ex}, root_label="2b")
    assert rejected["status"] == INVALID_RECORDS


def test_gate_rejects_real_example_id_with_fabricated_current_suite_fields(
        tmp_path):
    ex = _suite_ex()
    fabricated_candidates = tuple(
        f"fabricated-{index}" for index in range(len(ex.candidates)))
    record = _gate_v3_record(ex, record_overrides={
        "family": "fabricated-family",
        "prompt": "fabricated prompt",
        "candidates": fabricated_candidates,
        "candidate_permutation_seed": -777,
        "candidate_permutation": tuple(
            reversed(range(len(fabricated_candidates)))),
        "gold_answer": fabricated_candidates[0],
    })
    path = tmp_path / "malicious-current-v3.json"
    path.write_text(json.dumps(_gate_eval_payload(record)))

    rejected = g.evaluate_eval_file(
        path, {ex.ex_id: ex}, root_label="2b")
    assert rejected["status"] == INVALID_RECORDS
    assert "canonical behavioral-v3 fields" in rejected["detail"]
    assert rejected.get("evidence_class") != "VALID_CURRENT"


def test_gate_classifies_derived_only_legacy_as_irrecoverable(tmp_path):
    ex = _suite_ex()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(_eval_for("runs/E_k4_s0", "test_id", [{
        "ex_id": ex.ex_id, "correct": 1.0, "rank_of_gold": 0,
    }])))
    verdict = g.evaluate_eval_file(path, {ex.ex_id: ex}, root_label="2b")
    assert verdict["status"] == g.IRRECOVERABLE_LEGACY_SCORER


def test_old_suite_derived_only_eval_keeps_irrecoverable_classification(
        tmp_path):
    ex = _suite_ex()
    path = tmp_path / "legacy-old-suite.json"
    payload = _eval_for("runs/E_k4_s0", "test_id", [{
        "ex_id": ex.ex_id, "family": ex.family, "depth": ex.depth,
        "correct": 1.0, "rank_of_gold": 0,
        "n_candidates": len(ex.candidates),
    }])
    payload["suite_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    verdict = g.evaluate_eval_file(path, {ex.ex_id: ex}, root_label="2b")
    assert verdict["status"] == g.IRRECOVERABLE_LEGACY_SCORER
    assert verdict["evidence_class"] == g.IRRECOVERABLE_LEGACY_SCORER
    assert any("suite_mismatch" in reason
               for reason in verdict["additional_reasons"])


# ---------------------------------------------------------------------------
# synthetic corpus + gate end-to-end
# ---------------------------------------------------------------------------

PINNED_REV_2B = "15852e8c16360a2fea060d615a32b45270f8a8fc"
REV_4B = "8" * 40


def _semantic_config() -> dict:
    """Full training-semantic config (exactly what recipe_from_config
    requires — nothing defaulted)."""
    return {
        "mode": "E-localized", "interval": [12, 18], "k": 4,
        "lora_r": 8, "lora_alpha": 16.0, "lr": 0.0001, "steps": 800,
        "seed": 0, "max_k": 16, "optimizer": "adamw",
        "weight_decay": 0.01, "lr_schedule": "constant", "warmup": 20,
        "clip": 1.0, "detach_z0": False,
    }


def _semantic_config_over(**over) -> dict:
    return {**_semantic_config(), **over}


def _canonical_recipe(cfg=None) -> dict:
    from latent_lab.train.checkpointing import recipe_from_config

    return recipe_from_config(cfg if cfg is not None else _semantic_config(),
                              g.current_suite_sha256())


def _valid_report(**over) -> dict:
    rep = {
        "config": {
            **_semantic_config(),
            "device": "cuda",
            "train_examples": 490,
            "model": "Qwen/Qwen3.5-2B",
            "revision": PINNED_REV_2B,
            "label": "E_k4_s0",
            "suite_sha256": g.current_suite_sha256(),
        },
        "model": "Qwen/Qwen3.5-2B",
        "revision": PINNED_REV_2B,
        "suite_sha256": g.current_suite_sha256(),
        # The report's recipe must equal the canonical derivation of its
        # own config + suite hash (the gate re-derives it; never trusts).
        "recipe": None,
        "best_val_acc": 0.5,
        "best_step": 200,
        "val_history": [
            {"tag": "val", "ablate": {}, "k_steps": 4, "n": 28,
             "accuracy": 0.25, "by_depth": {}, "by_family": {},
             "seconds": 1.0, "step": 100},
            {"tag": "val", "ablate": {}, "k_steps": 4, "n": 28,
             "accuracy": 0.5, "by_depth": {}, "by_family": {},
             "seconds": 1.0, "step": 200},
        ],
        "final_train_loss": 1.23,
        "gpu_mem": {},
        "wall_seconds": 10.0,
        "peak_rss_mib": 1.0,
        "platform": "test",
        "run_id": "run-v3",
        "adapter_id": "runs/E_k4_s0",
        "selection_provenance":
            "latent_eval.v3_recomputed_from_raw_validation_records",
    }
    rep.update(over)
    rep["recipe"] = _canonical_recipe(rep["config"])
    return rep


def test_train_report_selection_accepts_only_latent_eval_v3_history():
    from latent_lab.bench.latent_run import canonical_v3_history_entry

    record = _gate_v3_record(_suite_ex("validation-"), split="validation")
    report = _valid_report(best_val_acc=1.0, best_step=100)
    report["val_history"] = [
        canonical_v3_history_entry(200, [record]),
        canonical_v3_history_entry(100, [record]),
    ]
    report["selected_adapter_state_sha256"] = record[
        "checkpoint_identity"]["content_sha256"]
    check = g.validate_train_report(report)
    assert "val_history:legacy_scorer_noncanonical" not in check["problems"]
    assert check["history_schema"] == "latent_eval.summary.v3"
    assert check["selection_check"]["consistent_with_reported"] is True
    assert check["selection_check"]["provenance"] == "latent_eval.v3"
    assert not any("selected_adapter_state_sha256" in problem
                   for problem in check["problems"])

    report["selected_adapter_state_sha256"] = "f" * 64
    rejected = g.validate_train_report(report)
    assert "selected_adapter_state_sha256:mismatch_with_selected_raw_history" \
        in rejected["problems"]


def test_train_report_rejects_schema_valid_fabricated_validation_history():
    from latent_lab.bench.latent_run import canonical_v3_history_entry

    ex = _suite_ex("validation-")
    fabricated_candidates = tuple(
        f"fabricated-{index}" for index in range(len(ex.candidates)))
    record = _gate_v3_record(
        ex,
        split="validation",
        record_overrides={
            "family": "fabricated-family",
            "prompt": "fabricated prompt",
            "candidates": fabricated_candidates,
            "candidate_permutation_seed": -777,
            "candidate_permutation": tuple(
                reversed(range(len(fabricated_candidates)))),
            "gold_answer": fabricated_candidates[0],
        },
    )
    report = _valid_report(best_val_acc=1.0, best_step=100)
    report["val_history"] = [canonical_v3_history_entry(100, [record])]
    report["selected_adapter_state_sha256"] = record[
        "checkpoint_identity"]["content_sha256"]

    rejected = g.validate_train_report(report)
    assert any(
        problem.startswith("val_history:v3_raw_records_invalid:")
        and "canonical behavioral-v3 fields" in problem
        for problem in rejected["problems"]
    )
    assert rejected["selection_check"] == {
        "recomputed_best_step": None,
        "provenance": INVALID_RECORDS,
        "consistent_with_reported": False,
    }
    assert rejected["identity"]["selected_adapter_state_sha256"] is None


def _suite_ex(prefix="test_id-"):
    from latent_lab.bench.no_spend_gate import suite_examples_by_id

    prefix = {"ti-": "test_id-", "to-": "test_ood_length-"}.get(
        prefix, prefix)
    for ex_id, ex in suite_examples_by_id().items():
        if ex_id.startswith(prefix):
            return ex
    raise AssertionError(f"no {prefix} example")


def _eval_for(adapter, split, records) -> dict:
    return {
        "adapter": adapter, "split": split,
        "model": "Qwen/Qwen3.5-2B", "revision": PINNED_REV_2B,
        "suite_sha256": g.current_suite_sha256(),
        "results": {"clean": {
            "tag": "t", "ablate": {}, "k_steps": 4, "n": len(records),
            "accuracy": 1.0, "by_depth": {}, "by_family": {},
            "seconds": 1.0,
            "records": records,
        }},
    }


def _raw_record(ex, *, hit=True) -> dict:
    n = len(ex.candidates)
    scores = [0.01 * i for i in range(n)]
    scores[list(ex.candidates).index(ex.answer)] = 9.0 if hit else -9.0
    return {"ex_id": ex.ex_id, "candidate_scores": scores}


def _derived_only_eval(adapter="runs/E_k4_s0") -> dict:
    ex = _suite_ex()
    return _eval_for(adapter, "test_id", [
        {"ex_id": ex.ex_id, "family": ex.family,
         "depth": ex.depth, "correct": 1.0,
         "rank_of_gold": 0, "n_candidates": len(ex.candidates)},
    ])


def _full_split_records(split: str) -> list[dict]:
    """Raw-score records for the COMPLETE preregistered split membership."""
    by_id = g.suite_examples_by_id()
    return [_raw_record(by_id[eid])
            for eid in sorted(g.suite_examples_by_split()[split])]


def _v3_eval_for(adapter, split, records) -> dict:
    assert records and records[0]["split"] == split
    return _gate_eval_payload(records[0], adapter=adapter, records=records)


def _full_split_v3_records(split: str, checkpoint_digest: str) -> list[dict]:
    by_id = g.suite_examples_by_id()
    return [
        _gate_v3_record(
            by_id[eid], split=split,
            checkpoint_content_hash=checkpoint_digest)
        for eid in sorted(g.suite_examples_by_split()[split])
    ]


def _clean_retained_evidence(corpus):
    """Retained 2B run with schema-valid report + verified fp32 bundle and
    an empty live 4B tree; quarantine negative evidence stays intact."""
    import shutil

    import torch

    from latent_lab.train.checkpointing import save_adapter_bundle

    r2b, r4b = corpus
    shutil.rmtree(r4b / "runs")
    shutil.rmtree(r2b / "runs" / "legacy_run")
    run = r2b / "runs" / "E_k4_s0"
    rep = _valid_report()
    rep["config"]["scorer"] = "corrected-gold-aware-v1"
    rep["trainable_precision"] = "fp32"
    (run / "train_report.json").write_text(json.dumps(rep))
    os.remove(run / "best_params.pt")
    save_adapter_bundle(run / "best_params.pt",
                        {"lora.A": torch.eye(2, dtype=torch.float32)},
                        model_id="Qwen/Qwen3.5-2B",
                        revision=PINNED_REV_2B,
                        recipe=rep["recipe"],
                        metrics={"best_score": 0.5, "best_step": 200})
    for p in r2b.glob("results/ev_*.json"):
        os.remove(p)
    return r2b, r4b


@pytest.fixture()
def corpus(tmp_path):
    """Synthetic retained-evidence tree mirroring the historical shape."""
    r2b = tmp_path / "remote_results"
    (r2b / "runs" / "E_k4_s0").mkdir(parents=True)
    (r2b / "runs" / "legacy_run").mkdir(parents=True)

    # valid-schema report (still missing trainable_precision -> blocker)
    (r2b / "runs" / "E_k4_s0" / "train_report.json").write_text(
        json.dumps(_valid_report()))
    # plain-dict legacy checkpoint (bf16 lora + fp32 clock analog)
    import torch

    torch.save({"lora.0.A": torch.zeros(2, 4, dtype=torch.bfloat16),
                "clock.weight": torch.zeros(3, 4)},
               r2b / "runs" / "E_k4_s0" / "best_params.pt")

    # schema-violating report (revision not pinned)
    bad = _valid_report()
    bad["revision"] = "main"
    (r2b / "runs" / "legacy_run" / "train_report.json").write_text(
        json.dumps(bad))
    torch.save({"w": torch.zeros(2)}, r2b / "runs" / "legacy_run" /
               "best_params.pt")

    (r2b / "results").mkdir()
    (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text(
        json.dumps(_derived_only_eval()))

    # 4B: quarantined NaN batch fully duplicated into the live tree
    r4b = tmp_path / "remote_results_4b"
    for top in ("runs", "_rejected_nan_batch"):
        d = r4b / top / "E4_k1_s0"
        d.mkdir(parents=True)
        rep = _valid_report(model="Qwen/Qwen3.5-4B",
                            revision=REV_4B)
        rep["final_train_loss"] = float("nan")
        rep["best_val_acc"] = 1.0
        (d / "train_report.json").write_text(json.dumps(rep))
        torch.save({"w": torch.tensor([float("inf")])}, d / "best_params.pt")
    (r4b / "_rejected_nan_batch" / "REJECTED.md").write_text(
        "# REJECTED\nfinal_train_loss=NaN; fake best 1.0\n")
    return r2b, r4b


PROOF_OK = {"all_passed": True, "returncode": 0,
            "nodes": list(g.PROOF_TEST_NODES)}


def test_proof_runner_opts_in_and_rejects_skipped_transcript(
        monkeypatch, tmp_path):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="16 passed, 9 skipped in 0.42s\n",
        )

    monkeypatch.setattr(g.subprocess, "run", fake_run)
    result = g.run_proof_tests(tmp_path)

    assert "--run-transformer-integration" in observed["command"]
    override = observed["command"].index("-o")
    assert observed["command"][override + 1] == "addopts="
    assert result["returncode"] == 0
    assert result["outcome_counts"]["passed"] == 16
    assert result["outcome_counts"]["skipped"] == 9
    assert result["summary_parsed"] is True
    assert result["all_passed"] is False
    assert any("9 skipped" in reason
               for reason in result["failure_reasons"])


@pytest.mark.parametrize(
    "stdout,returncode,expected_reason",
    [
        ("7 passed in 0.1s\n", 0, None),
        ("6 passed in 0.1s\n", 0, "only 6 passed outcomes"),
        (".......\n", 0, "outcome summary missing"),
        ("7 passed in 0.1s\n", 4, "pytest return code 4"),
    ],
)
def test_proof_runner_requires_every_node_and_success_summary(
        monkeypatch, tmp_path, stdout, returncode, expected_reason):
    monkeypatch.setattr(
        g.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=stdout),
    )
    result = g.run_proof_tests(tmp_path)
    assert result["all_passed"] is (expected_reason is None)
    if expected_reason is not None:
        assert any(expected_reason in reason
                   for reason in result["failure_reasons"])


class TestGateEndToEnd:
    def run_gate(self, corpus, **kw):
        r2b, r4b = corpus
        kw.setdefault("skip_proof_tests", True)
        return g.run_gate(r2b, r4b, repo_root=None, **kw)

    def test_discovery_counts_without_hand_list(self, corpus):
        res = self.run_gate(corpus)
        c = res.gate_verdict["counts"]
        assert c["runs_2b"] == 2
        assert c["runs_4b_live"] == 1
        assert c["runs_4b_rejected"] == 1
        assert c["files_scanned"] == 10
        assert c["checkpoints_total"] == 4

    def test_skipped_proof_transcript_cannot_mark_runtime_proven(self, corpus):
        proof = {
            "all_passed": False,
            "returncode": 0,
            "nodes": list(g.PROOF_TEST_NODES),
            "outcome_counts": {"passed": 16, "skipped": 9},
            "summary_parsed": True,
            "failure_reasons": [
                "required proof outcomes include 9 skipped",
            ],
        }
        res = self.run_gate(corpus, proof_result=proof)
        statuses = {item["id"]: item["status"]
                    for item in res.gate_verdict["prerequisites"]}
        assert statuses[g.PREREQ_RUNTIME] == g.STATUS_FAILED
        assert any(
            blocker["code"] == "PROOF_TESTS_FAILED"
            and "9 skipped" in blocker["detail"]
            for blocker in res.gate_verdict["blockers"]
        )

    def test_inventory_hashes_duplicates_hardlinks(self, tmp_path, corpus):
        r2b, r4b = corpus
        src = r2b / "runs" / "E_k4_s0" / "best_params.pt"
        dup_dir = r2b / "runs" / "dupe_run"
        dup_dir.mkdir()
        (dup_dir / "best_params.pt").write_bytes(src.read_bytes())
        os.link(src, r2b / "results" / "linked_copy.pt")

        res = self.run_gate(corpus)
        inv = {a["id"]: a for a in res.inventory["artifacts"]}
        a = inv["2b/runs/E_k4_s0/best_params.pt"]
        b = inv["2b/runs/dupe_run/best_params.pt"]
        assert a["sha256"] == b["sha256"]
        assert b["identical_content_with"] == sorted(
            ["2b/runs/E_k4_s0/best_params.pt", "2b/results/linked_copy.pt"])
        linked = inv["2b/results/linked_copy.pt"]
        assert linked["hardlinked_with"] == ["2b/runs/E_k4_s0/best_params.pt"]

    def test_historical_corpus_is_NOT_READY_with_exact_blockers(self, corpus):
        res = self.run_gate(corpus)
        v = res.gate_verdict
        assert v["verdict"] == "NOT_READY"
        codes = {b["code"] for b in v["blockers"]}
        assert "CKPT_LEGACY_UNBOUND_IDENTITY" in codes
        assert "IRRECOVERABLE_LEGACY_SCORER" in codes
        assert "REPORT_SCHEMA_MISSING_FIELDS" in codes      # unpinned rev
        assert "REPORT_TRAINABLE_PRECISION_MISSING" in codes
        assert "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS" in codes
        assert "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH" in codes
        assert "SELECTION_PROVENANCE_NOT_CORRECTED" in codes
        for b in v["blockers"]:
            assert b["smallest_next_action"].strip()

    def test_old_suite_derived_only_counts_as_irrecoverable_blocker(
            self, corpus):
        path = (corpus[0] / "results" /
                "ev_E_k4_s0_test_id_clean.json")
        payload = json.loads(path.read_text())
        payload["suite_sha256"] = "0" * 64
        path.write_text(json.dumps(payload))

        res = self.run_gate(corpus)
        verdict = next(item for item
                       in res.artifact_verdicts["evaluations"]
                       if item["file"] == path.name)
        assert verdict["status"] == g.IRRECOVERABLE_LEGACY_SCORER
        assert res.gate_verdict["counts"][
            "eval_files_missing_raw_prediction"] >= 1
        assert any(
            blocker["code"] == g.IRRECOVERABLE_LEGACY_SCORER
            for blocker in res.gate_verdict["blockers"])

    def test_corrupt_checkpoint_classified_and_quarantine_analyzed(
            self, corpus):
        res = self.run_gate(corpus)
        runs = {(r["root"], r["run_id"]): r
                for r in res.artifact_verdicts["runs"]}
        q = res.artifact_verdicts["quarantine_4b"]
        assert runs[("4b", "E4_k1_s0")]["checkpoint"][
            "classification"] == "corrupt"
        assert q["quarantine_marker_present"] is True
        assert q["live_tree_duplicates_quarantine"] is True
        assert q["rejected_runs_with_nan_final_loss"] == ["E4_k1_s0"]
        assert q["live_vs_rejected_differing_files"] == []

    def test_rejected_and_quarantine_duplicates_are_not_retained_or_blocking(
            self, corpus):
        import shutil

        _, r4b = corpus
        live = r4b / "runs" / "E4_k1_s0"
        archived = (r4b / "quarantine" /
                    "live_nan_duplicates_20260827" / "E4_k1_s0")
        archived.parent.mkdir(parents=True)
        shutil.move(str(live), str(archived))
        # Preserved negative evals remain inventoried/classified, but must
        # not join a current run or poison retained-evidence prerequisites.
        negative_eval = archived / "ev_quarantined_invalid.json"
        negative_eval.write_text("{}")

        res = self.run_gate(corpus)
        statuses = {item["id"]: item["status"]
                    for item in res.gate_verdict["prerequisites"]}
        assert statuses[g.PREREQ_QUARANTINE] == g.STATUS_PROVEN
        assert not res.artifact_verdicts["duplicate_checkpoint_bindings"]
        assert res.artifact_verdicts[
            "preserved_negative_duplicate_bindings"]
        four_b = [run for run in res.artifact_verdicts["runs"]
                  if run["root"] == "4b"]
        assert four_b and all(run["scope"] == "rejected_negative"
                              for run in four_b)
        eval_entry = next(
            item for item in res.artifact_verdicts["evaluations"]
            if item.get("artifact_rel", "").endswith(
                "ev_quarantined_invalid.json"))
        assert eval_entry["scope"] == "rejected_negative"
        assert eval_entry["status"] == "invalid_metadata"
        assert eval_entry["bound_run"] is None
        assert eval_entry["binding_excluded_reason"]
        counts = res.gate_verdict["counts"]
        assert counts["eval_files_inventoried"] \
            == counts["eval_files_checked"] + 1
        assert counts["eval_files_rejected_negative"] == 1
        assert not any(
            "ev_quarantined_invalid.json" in blocker["detail"]
            for blocker in res.gate_verdict["blockers"])
        assert not any(
            blocker["code"] == "DUPLICATE_CHECKPOINT_BINDING"
            for blocker in res.gate_verdict["blockers"])

        # Restoring the same bytes under runs/ makes the ambiguity/live
        # quarantine leak current again and must block.
        shutil.copytree(archived, live)
        live_res = self.run_gate(corpus)
        live_statuses = {item["id"]: item["status"]
                         for item in live_res.gate_verdict["prerequisites"]}
        assert live_statuses[g.PREREQ_QUARANTINE] == g.STATUS_FAILED
        assert live_res.artifact_verdicts["duplicate_checkpoint_bindings"]
        assert any(
            blocker["code"] == "DUPLICATE_CHECKPOINT_BINDING"
            for blocker in live_res.gate_verdict["blockers"])

    def test_strict_fp32_bundle_from_project_loader_is_LOADABLE(
            self, corpus):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "bound_run"
        run.mkdir()
        state = {"lora.0.A": torch.eye(2, dtype=torch.float32)}
        rep = _valid_report()
        save_adapter_bundle(run / "best_params.pt", state,
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=rep["recipe"],
                            metrics={"best_score": 0.5, "best_step": 200})
        (run / "train_report.json").write_text(json.dumps(rep))

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "bound_run"][0]
        ck = rv["checkpoint"]
        assert ck["classification"] == "loadable"
        assert "strict_project_loader_passed" in ck["reasons"]
        assert "payload_floating_tensors_all_fp32" in ck["reasons"]
        assert "bundle_recipe_equals_independently_derived_report_recipe" \
            in ck["reasons"]

    def test_bf16_bundle_payload_blocks_READY_even_when_report_claims_fp32(
            self, corpus):
        """Negative control for the audit repro 'bf16_bundle_and_report'."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "bf16_run"
        run.mkdir()
        rep = _valid_report()
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2, dtype=torch.bfloat16)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=rep["recipe"],
                            metrics={"best_score": 0.5, "best_step": 200})
        rep["trainable_precision"] = "fp32"   # lying report string
        (run / "train_report.json").write_text(json.dumps(rep))

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "bf16_run"][0]
        assert rv["checkpoint"]["classification"] == "non-fp32-payload"
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS" in codes
        assert res.gate_verdict["prerequisites"] and \
            res.verdict == "NOT_READY"

    def test_identity_conflict_between_bundle_and_report_invalid(
            self, corpus):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "conflict_run"
        run.mkdir()
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.0.A": torch.eye(2)},
                            model_id="other/model", revision="b" * 40,
                            recipe=_valid_report()["recipe"])
        (run / "train_report.json").write_text(json.dumps(_valid_report()))
        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "conflict_run"][0]
        assert rv["checkpoint"]["classification"] == "invalid"

    # ---- canonical-recipe binding (recipe_from_config integration) ---------

    def test_bundle_recipe_cannot_substitute_for_derived_report_recipe(
            self, corpus):
        """The bundle's own recipe is NEVER trusted: a checkpoint saved
        under a different training recipe than the one independently
        derived from its validated sidecar report config must fail closed
        as invalid — even though its raw in-bundle recipe is internally
        valid and self-consistent."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "recipe_swap_run"
        run.mkdir()
        rep = _valid_report()
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        (run / "train_report.json").write_text(json.dumps(rep))
        # Bundle is byte-valid but trained under a DIFFERENT recipe
        # (different lr/steps) than the report's config implies.
        divergent_cfg = _semantic_config_over(lr=0.002, steps=900)
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=_canonical_recipe(divergent_cfg),
                            metrics={"best_score": 0.5, "best_step": 200})

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "recipe_swap_run"][0]
        ck = rv["checkpoint"]
        assert ck["classification"] == "invalid"
        assert any("identity_error" in str(r) or "recipe" in str(r)
                   for r in ck["reasons"])
        assert "strict_project_loader_passed" not in ck["reasons"]
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "CKPT_INVALID_IDENTITY" in codes
        assert res.verdict == "NOT_READY"

    def test_tampered_report_recipe_fails_closed(self, corpus):
        """A report whose declared recipe differs from the canonical
        derivation of its own validated config + suite hash is a schema
        blocker, yields NO trusted recipe, and cannot make its checkpoint
        loadable."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "tampered_recipe_run"
        run.mkdir()
        rep = _valid_report()
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        tampered = dict(rep["recipe"])
        tampered["lr"] = 9.9          # hand-edited recipe field
        rep["recipe"] = tampered
        (run / "train_report.json").write_text(json.dumps(rep))
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=_canonical_recipe(),   # true recipe
                            metrics={"best_score": 0.5, "best_step": 200})

        check = g.validate_train_report(rep)
        assert "recipe:mismatch_with_derived_canonical" in check["problems"]
        assert check["identity"]["recipe"] is None

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "tampered_recipe_run"][0]
        joined = ",".join(rv["report_problems"])
        assert "recipe:mismatch_with_derived_canonical" in joined
        assert rv["checkpoint"]["classification"] == "invalid"
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "REPORT_RECIPE_NOT_CANONICAL" in codes
        detail = [b["detail"] for b in res.gate_verdict["blockers"]
                  if b["code"] == "REPORT_RECIPE_NOT_CANONICAL"][0]
        assert "tampered_recipe_run" in detail
        statuses = {p["id"]: p["status"]
                    for p in res.gate_verdict["prerequisites"]}
        assert statuses[g.PREREQ_REPORTS] == "FAILED"
        assert res.verdict == "NOT_READY"

    def test_underivable_config_recipe_blocks_report_and_checkpoint(
            self, corpus):
        """A report whose config cannot yield a canonical recipe at all
        (missing training-semantic fields) fails closed on BOTH fronts:
        no trusted recipe and an unloadable checkpoint."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "thin_config_run"
        run.mkdir()
        rep = _valid_report()
        del rep["config"]["optimizer"]    # semantic field now missing
        del rep["recipe"]                 # and nothing to verify against
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        (run / "train_report.json").write_text(json.dumps(rep))
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=_canonical_recipe(),
                            metrics={"best_score": 0.5, "best_step": 200})

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "thin_config_run"][0]
        joined = ",".join(rv["report_problems"])
        assert "recipe:not_derivable_from_validated_config_and_suite" \
            in joined
        assert rv["checkpoint"]["classification"] == "invalid"
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "REPORT_RECIPE_NOT_CANONICAL" in codes
        assert res.verdict == "NOT_READY"

    def test_orphan_identity_bound_bundle_fails_closed_without_recipe(
            self, corpus):
        """An orphan IDENTITY-BOUND bundle (valid bytes, fp32 payload) has
        no owning report, so no canonical recipe can be derived; it must
        classify invalid — never loadable on its own in-bundle claims."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        orphan = r2b / "runs" / "orphan_bound"
        orphan.mkdir()
        save_adapter_bundle(orphan / "best_params.pt",
                            {"lora.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=_canonical_recipe(),
                            metrics={"best_score": 0.5, "best_step": 200})
        # deliberately NO train_report.json sibling

        res = self.run_gate(corpus)
        o = [x for x in res.artifact_verdicts["orphan_checkpoints"]
             if x["id"] == "2b/runs/orphan_bound/best_params.pt"][0]
        assert o["classification"] == "invalid"
        assert any("bundle_unverifiable_without_derived_report_recipe"
                   in str(r) for r in o["reasons"])
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "ORPHAN_CHECKPOINT" in codes
        assert res.verdict == "NOT_READY"

    def test_matching_recipes_positive_control_loadable(self, corpus):
        """Positive matching control: when the report recipe EQUALS the
        canonical derivation AND the bundle was saved against exactly that
        recipe, the strict loader accepts it (loadable) — proving the
        negative controls above fail because of real mismatches, not
        because loading is impossible."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "matching_recipe_run"
        run.mkdir()
        rep = _valid_report()
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        (run / "train_report.json").write_text(json.dumps(rep))
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=rep["recipe"],
                            metrics={"best_score": 0.5, "best_step": 200})

        res = self.run_gate(corpus)
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "matching_recipe_run"][0]
        ck = rv["checkpoint"]
        assert ck["classification"] == "loadable"
        assert "bundle_recipe_equals_independently_derived_report_recipe" \
            in ck["reasons"]
        assert not any(str(p).startswith("recipe:")
                       for p in rv["report_problems"])

    # ---- negative controls mirroring the independent audit -----------------

    def test_eval_suite_hash_mismatch_is_NOT_READY(self, corpus):
        p = corpus[0] / "results" / "ev_E_k4_s0_test_id_clean.json"
        ev = json.loads(p.read_text())
        ev["suite_sha256"] = "0" * 64
        p.write_text(json.dumps(ev))
        res = self.run_gate(corpus)
        entry = res.artifact_verdicts["evaluations"][0]
        assert entry["status"] == g.IRRECOVERABLE_LEGACY_SCORER
        assert any("suite_mismatch" in reason
                   for reason in entry["additional_reasons"])
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_FILE_INVALID" in codes
        assert res.verdict == "NOT_READY"

    def test_eval_no_records_is_NOT_READY(self, corpus):
        p = corpus[0] / "results" / "ev_E_k4_s0_test_id_clean.json"
        ev = json.loads(p.read_text())
        ev["results"] = {}
        p.write_text(json.dumps(ev))
        res = self.run_gate(corpus)
        assert res.artifact_verdicts["evaluations"][0][
            "status"] == "invalid_metadata"
        assert res.verdict == "NOT_READY"

    def test_eval_unreadable_json_is_NOT_READY(self, corpus):
        p = corpus[0] / "results" / "ev_E_k4_s0_test_id_clean.json"
        p.write_text("{broken-json")
        res = self.run_gate(corpus)
        ev = [e for e in res.artifact_verdicts["evaluations"]
              if e["file"] == "ev_E_k4_s0_test_id_clean.json"][0]
        assert ev["status"] == "malformed_json"
        assert ev["bound_run"] is None
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_FILE_INVALID" in codes and "ORPHAN_EVAL_FILE" in codes
        assert res.verdict == "NOT_READY"

    def test_eval_all_nan_scores_fails_at_strict_parse_never_rescored(
            self, corpus):
        """The audit wrote literal NaN tokens; strict JSON must reject the
        whole file before any rescoring can silently sink them to -inf."""
        ex = _suite_ex()
        n = len(ex.candidates)
        ev = _derived_only_eval()
        ev["results"]["clean"]["records"] = [
            {"ex_id": ex.ex_id, "candidate_scores":
                [float("nan")] * n}]
        p = corpus[0] / "results" / "ev_E_k4_s0_test_id_clean.json"
        p.write_text(json.dumps(ev))     # emits literal NaN tokens
        res = self.run_gate(corpus)
        entry = res.artifact_verdicts["evaluations"][0]
        assert entry["status"] == "malformed_json"
        assert "corrected_accuracy" not in entry
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_FILE_INVALID" in codes
        assert res.verdict == "NOT_READY"

    def test_eval_with_nan_literal_json_fails_closed(self, corpus):
        p = corpus[0] / "results" / "ev_E_k4_s0_test_id_clean.json"
        p.write_text('{"adapter": "runs/E_k4_s0", "loss": NaN}')
        res = self.run_gate(corpus)
        entry = [e for e in res.artifact_verdicts["evaluations"]
                 if e["file"] == "ev_E_k4_s0_test_id_clean.json"][0]
        assert entry["status"] == "malformed_json"
        assert res.verdict == "NOT_READY"

    def test_second_loadable_checkpoint_without_eval_coverage_blocks(
            self, corpus):
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        run = r2b / "runs" / "runB"
        run.mkdir()
        rep = _valid_report()
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        (run / "train_report.json").write_text(json.dumps(rep))
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=rep["recipe"],
                            metrics={"best_score": 0.5, "best_step": 200})
        res = self.run_gate(corpus)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_SPLIT_COVERAGE_MISSING" in codes
        detail = [b["detail"] for b in res.gate_verdict["blockers"]
                  if b["code"] == "EVAL_SPLIT_COVERAGE_MISSING"][0]
        assert "2b/runs/runB" in detail
        assert res.verdict == "NOT_READY"

    def test_single_split_coverage_is_insufficient(self, corpus):
        """One ID-split eval cannot prove the global prerequisite."""
        import shutil
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, r4b = corpus
        shutil.rmtree(r4b / "runs")           # clean quarantine side
        shutil.rmtree(r2b / "runs" / "legacy_run")
        run = r2b / "runs" / "E_k4_s0"
        rep = _valid_report()
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        rep["trainable_precision"] = "fp32"
        (run / "train_report.json").write_text(json.dumps(rep))
        os.remove(run / "best_params.pt")
        save_adapter_bundle(run / "best_params.pt",
                            {"lora.A": torch.eye(2)},
                            model_id="Qwen/Qwen3.5-2B",
                            revision=PINNED_REV_2B,
                            recipe=rep["recipe"],
                            metrics={"best_score": 0.5, "best_step": 200})
        ex_ti = _suite_ex("ti-")
        (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text(
            json.dumps(_eval_for("runs/E_k4_s0", "test_id",
                                 [_raw_record(ex_ti)])))
        # NO length-OOD eval at all.
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_SPLIT_COVERAGE_MISSING" in codes
        detail = [b["detail"] for b in res.gate_verdict["blockers"]
                  if b["code"] == "EVAL_SPLIT_COVERAGE_MISSING"][0]
        assert "test_ood_length" in detail
        assert res.verdict == "NOT_READY"

    def test_single_record_cannot_cover_full_preregistered_split(self, corpus):
        """Regression (audit 91e842f defect 1): one raw record merely
        LABELLED for a required split must not count as coverage of the
        complete preregistered example set of that split."""
        r2b, r4b = _clean_retained_evidence(corpus)
        ex_ti = _suite_ex("ti-")
        (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text(
            json.dumps(_eval_for("runs/E_k4_s0", "test_id",
                                 [_raw_record(ex_ti)])))
        (r2b / "results" / "ev_E_k4_s0_test_ood_length_clean.json").write_text(
            json.dumps(_eval_for("runs/E_k4_s0", "test_ood_length",
                                 _full_split_records("test_ood_length"))))
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_SPLIT_COVERAGE_MISSING" in codes
        detail = [b["detail"] for b in res.gate_verdict["blockers"]
                  if b["code"] == "EVAL_SPLIT_COVERAGE_MISSING"][0]
        assert "test_id" in detail and "preregistered" in detail
        assert res.verdict == "NOT_READY"

    def test_test_id_records_relabelled_test_ood_violate_membership(
            self, corpus):
        """Regression (audit 91e842f defect 2): copying test_id examples
        into a file declared test_ood_length must not satisfy OOD coverage; every
        record's ex_id must belong to the preregistered membership of the
        declared split."""
        r2b, r4b = _clean_retained_evidence(corpus)
        ti_records = _full_split_records("test_id")
        (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text(
            json.dumps(_eval_for("runs/E_k4_s0", "test_id", ti_records)))
        (r2b / "results" / "ev_E_k4_s0_test_ood_length_clean.json").write_text(
            json.dumps(_eval_for("runs/E_k4_s0", "test_ood_length",
                                 list(ti_records))))
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "EVAL_SPLIT_MEMBERSHIP_VIOLATION" in codes
        detail = [b["detail"] for b in res.gate_verdict["blockers"]
                  if b["code"] == "EVAL_SPLIT_MEMBERSHIP_VIOLATION"][0]
        assert "test_ood_length" in detail
        cov = [b["detail"] for b in res.gate_verdict["blockers"]
               if b["code"] == "EVAL_SPLIT_COVERAGE_MISSING"][0]
        assert "test_ood_length" in cov
        assert res.verdict == "NOT_READY"

    def test_live_4b_retained_checkpoint_requires_full_prerequisites(
            self, corpus):
        """Regression (audit 91e842f defect 3): a retained live 4B
        checkpoint must meet the SAME report-schema and eval prerequisites
        as 2B; a live 4B report lacking trainable_precision with no eval
        evidence must force NOT_READY."""
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, r4b = corpus
        live = r4b / "runs" / "E4_retained"
        live.mkdir(parents=True)
        rep = _valid_report(model="Qwen/Qwen3.5-4B", revision=REV_4B)
        # _valid_report carries no trainable_precision field at all
        assert "trainable_precision" not in rep
        rep["config"]["scorer"] = "corrected-gold-aware-v1"
        (live / "train_report.json").write_text(json.dumps(rep))
        save_adapter_bundle(live / "best_params.pt",
                            {"lora.A": torch.ones(2, dtype=torch.float32)},
                            model_id="Qwen/Qwen3.5-4B", revision=REV_4B,
                            recipe=rep["recipe"],
                            metrics={"best_score": 0.4, "best_step": 200})
        res = self.run_gate(corpus)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        prec = [b["detail"] for b in res.gate_verdict["blockers"]
                if b["code"] == "REPORT_TRAINABLE_PRECISION_MISSING"][0]
        assert "E4_retained" in prec
        assert "EVAL_SPLIT_COVERAGE_MISSING" in codes
        cov = [b["detail"] for b in res.gate_verdict["blockers"]
               if b["code"] == "EVAL_SPLIT_COVERAGE_MISSING"][0]
        assert "4b/runs/E4_retained" in cov
        assert res.verdict == "NOT_READY"

    def test_orphan_corrupt_checkpoint_is_explicit_blocker(self, corpus):
        import torch

        r2b, _ = corpus
        orphan = r2b / "runs" / "orphan"
        orphan.mkdir()
        torch.save({"w": torch.tensor([float("nan")])},
                   orphan / "best_params.pt")   # no train_report.json
        res = self.run_gate(corpus)
        orphans = res.artifact_verdicts["orphan_checkpoints"]
        assert [o["id"] for o in orphans] == ["2b/runs/orphan/best_params.pt"]
        assert orphans[0]["classification"] == "corrupt"
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "ORPHAN_CHECKPOINT" in codes
        assert res.verdict == "NOT_READY"

    def test_live_4b_partial_duplicate_still_blocks(self, corpus):
        """A differing best_params.pt must not mask the byte-identical
        quarantined train_report.json left in the LIVE tree."""
        import shutil

        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        _, r4b = corpus
        rejected = r4b / "_rejected_nan_batch" / "E4_k1_s0"
        live = r4b / "runs" / "E4_k1_s0"      # already exists in corpus
        shutil.copy2(rejected / "train_report.json",
                     live / "train_report.json")   # identical bytes
        save_adapter_bundle(live / "best_params.pt", {"lora.A": torch.eye(2)},
                            model_id="Qwen/Qwen3.5-4B", revision=REV_4B,
                            recipe=_valid_report(
                                model="Qwen/Qwen3.5-4B",
                                revision=REV_4B)["recipe"],
                            metrics={"best_score": 1.0, "best_step": 200})
        res = self.run_gate(corpus)
        q = res.artifact_verdicts["quarantine_4b"]
        assert q["live_vs_rejected_identical_files"] == \
            ["E4_k1_s0/train_report.json"]
        assert q["live_vs_rejected_differing_files"] == \
            ["E4_k1_s0/best_params.pt"]
        assert q["live_tree_duplicates_quarantine"] is True
        assert q["live_invalid_runs"] == ["E4_k1_s0"]  # NaN loss report live
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH" in codes
        assert "LIVE_INVALID_4B_ARTIFACTS" in codes
        assert res.verdict == "NOT_READY"

    def test_marker_only_empty_quarantine_proves_nothing(self, corpus):
        import shutil

        _, r4b = corpus
        shutil.rmtree(r4b / "_rejected_nan_batch")
        (r4b / "_rejected_nan_batch").mkdir()
        (r4b / "_rejected_nan_batch" / "REJECTED.md").write_text("# R\n")
        res = self.run_gate(corpus)
        q = res.artifact_verdicts["quarantine_4b"]
        assert q["marker_only_empty_quarantine"] is True
        assert q["rejected_batch_empty"] is True
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "QUARANTINE_BATCH_EMPTY" in codes
        statuses = {p["id"]: p["status"]
                    for p in res.gate_verdict["prerequisites"]}
        assert statuses[g.PREREQ_QUARANTINE] == "FAILED"
        assert res.verdict == "NOT_READY"

    def test_duplicate_checkpoint_binding_across_runs_blocks(self, corpus):
        import shutil
        import torch

        from latent_lab.train.checkpointing import save_adapter_bundle

        r2b, _ = corpus
        shutil.rmtree(r2b / "runs" / "legacy_run")
        run_a = r2b / "runs" / "dup_a"
        run_b = r2b / "runs" / "dup_b"
        for d in (run_a, run_b):
            d.mkdir()
            (d / "train_report.json").write_text(json.dumps(_valid_report()))
            save_adapter_bundle(d / "best_params.pt",
                                {"lora.A": torch.eye(2)},
                                model_id="Qwen/Qwen3.5-2B",
                                revision=PINNED_REV_2B,
                                recipe=_valid_report()["recipe"],
                                metrics={"best_score": 0.5, "best_step": 200})
        # make them byte-identical
        (run_b / "best_params.pt").write_bytes(
            (run_a / "best_params.pt").read_bytes())
        res = self.run_gate(corpus)
        codes = {b["code"] for b in res.gate_verdict["blockers"]}
        assert "DUPLICATE_CHECKPOINT_BINDING" in codes
        groups = res.artifact_verdicts["duplicate_checkpoint_bindings"]
        # one group for our dup_a/dup_b pair; the corpus also carries the
        # byte-identical live/rejected 4B checkpoint pair
        dup_pair = [grp for grp in groups
                    if any(m.endswith("dup_a/best_params.pt")
                           for m in grp["members"])]
        assert len(dup_pair) == 1 and len(dup_pair[0]["members"]) == 2
        assert res.verdict == "NOT_READY"

    def test_malformed_train_report_fails_closed_not_crash(self, corpus):
        p = corpus[0] / "runs" / "legacy_run" / "train_report.json"
        p.write_text("{not json at all")
        res = self.run_gate(corpus)      # must not raise
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "legacy_run"][0]
        assert rv["report_problems"] == ["train_report:malformed_json"]
        assert res.verdict == "NOT_READY"

    def test_report_schema_negative_steps_and_bad_step_types_block(
            self, corpus):
        """JSON-expressible schema violations hit the granular checks."""
        p = corpus[0] / "runs" / "E_k4_s0" / "train_report.json"
        rep = _valid_report()
        rep["config"]["steps"] = -800
        rep["best_step"] = 200.5            # float step rejected
        p.write_text(json.dumps(rep))
        res = self.run_gate(corpus)         # must not crash
        rv = [r for r in res.artifact_verdicts["runs"]
              if r["run_id"] == "E_k4_s0"][0]
        joined = ",".join(rv["report_problems"])
        assert "config.steps:negative" in joined
        assert "best_step:type" in joined
        assert res.verdict == "NOT_READY"

    def test_report_nonfinite_metrics_rejected_by_validator(self):
        """NaN/Inf metric values are schema violations (unit level)."""
        rep = _valid_report()
        rep["best_val_acc"] = float("nan")
        rep["val_history"][0]["accuracy"] = float("inf")
        rep["final_train_loss"] = float("nan")
        check = g.validate_train_report(rep)
        joined = ",".join(check["problems"])
        assert "best_val_acc:type_or_nonfinite" in joined
        assert "val_history:bad_entries[0]" in joined
        assert "final_train_loss:type_or_nonfinite" in joined

    def test_report_bool_steps_and_metrics_are_rejected_not_coerced(self):
        rep = _valid_report()
        rep["config"]["steps"] = True
        rep["best_step"] = False
        rep["val_history"][0]["step"] = True
        check = g.validate_train_report(rep)
        joined = ",".join(check["problems"])
        assert "config.steps:type" in joined
        assert "best_step:type" in joined
        assert "val_history:bad_entries[0]" in joined

    def test_legacy_raw_records_remain_historical_unbound(self, corpus):
        ex = _suite_ex()
        ev = _derived_only_eval()
        ev["results"]["clean"]["records"] = [
            _raw_record(ex, hit=True),
            _raw_record(ex, hit=False),
        ]
        r2b, _ = corpus
        p = r2b / "results" / "ev_raw_test_id_clean.json"
        p.write_text(json.dumps(ev))

        res = self.run_gate(corpus)
        entry = [e for e in res.artifact_verdicts["evaluations"]
                 if e["file"] == "ev_raw_test_id_clean.json"][0]
        assert entry["status"] == g.HISTORICAL_UNBOUND_LEGACY_SCORER
        assert entry["status"] in g.EVAL_BAD_STATUSES
        assert entry["legacy_rescore_fraction"] == pytest.approx(0.5)
        assert any(
            blocker["code"] == "EVAL_FILE_INVALID"
            and "ev_raw_test_id_clean.json" in blocker["detail"]
            for blocker in res.gate_verdict["blockers"])

    def test_READY_positive_control_exit0_path(self, corpus):
        """Every prerequisite satisfiable offline IS provable — proves the
        gate is not rigged to permanent NOT_READY. The positive control
        uses a VERIFIED fp32 bundle payload and full ID+OOD raw-score
        coverage bound to the checkpoint's identity."""
        import shutil

        import torch

        from latent_lab.train.checkpointing import (
            adapter_state_sha256, save_adapter_bundle, sha256_file,
            write_run_status, write_train_generation,
        )

        r2b, r4b = corpus
        shutil.rmtree(r4b / "runs")           # live 4B tree holds nothing
        run = r2b / "runs" / "E_k4_s0"
        from latent_lab.bench.latent_run import canonical_v3_history_entry

        val_ex = _suite_ex("validation-")
        state = {"lora.A": torch.eye(2, dtype=torch.float32)}
        selected_state_sha256 = adapter_state_sha256(state)
        losing = _gate_v3_record(val_ex, split="validation", hit=False,
                                 checkpoint_content_hash="1" * 64)
        winning = _gate_v3_record(val_ex, split="validation", hit=True,
                                  checkpoint_content_hash=
                                  selected_state_sha256)
        rep = _valid_report(best_val_acc=1.0, best_step=200,
                            val_history=[
                                canonical_v3_history_entry(100, [losing]),
                                canonical_v3_history_entry(200, [winning]),
                            ])
        rep["trainable_precision"] = "fp32"
        rep["selected_adapter_state_sha256"] = selected_state_sha256
        rep["suite_identity"] = "behavioral-v3"
        rep["suite_version"] = 3
        os.remove(run / "best_params.pt")
        bundle = save_adapter_bundle(
            run / "best_params.pt", state,
            model_id="Qwen/Qwen3.5-2B", revision=PINNED_REV_2B,
            recipe=rep["recipe"],
            metrics={"best_score": 1.0, "best_step": 200})
        rep["checkpoint_content_digest"] = bundle["content_digest"]
        rep["checkpoint_sha256"] = sha256_file(run / "best_params.pt")
        manifest = {
            "kind": "latent_lab.train_generation", "status": "complete",
            "argv": ["python", "-m", "latent_lab.bench.latent_run",
                     "train"],
            "command": "python -m latent_lab.bench.latent_run train",
            "dependencies": {"python": "test", "torch": "test"},
            "precision": {"backbone_dtype": "torch.float32"},
            "seed": 0, "label": "E_k4_s0",
            "identity": {"model_id": "Qwen/Qwen3.5-2B",
                         "revision": PINNED_REV_2B},
            "run_id": "run-v3", "recipe": rep["recipe"],
            "suite_identity": "behavioral-v3", "suite_version": 3,
            "suite_sha256": g.current_suite_sha256(),
            "selected_adapter_state_sha256": selected_state_sha256,
            "checkpoint_content_digest": bundle["content_digest"],
            "checkpoint_sha256": sha256_file(run / "best_params.pt"),
            "wall_seconds": 1.0,
        }
        write_run_status(run, "complete")
        write_train_generation(run, manifest=manifest, report=rep)
        # remove legacy_run's schema-violating report tree entirely
        shutil.rmtree(r2b / "runs" / "legacy_run")

        # Full preregistered behavioral-v3 coverage, bound to the exact
        # checkpoint content digest (honest READY control).
        for split in g.REQUIRED_SPLITS:
            records = _full_split_v3_records(split, bundle["content_digest"])
            (r2b / "results" /
             f"ev_E_k4_s0_{split}_clean.json").write_text(
                json.dumps(_v3_eval_for("runs/E_k4_s0", split, records)))

        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True,
                         proof_result=PROOF_OK)
        v = res.gate_verdict
        assert v["verdict"] == "READY", json.dumps(v, indent=1)
        statuses = {p["id"]: p["status"] for p in v["prerequisites"]}
        assert all(s == "PROVEN" for s in statuses.values())
        assert v["blockers"] == []
        assert v["inputs"]["source_fingerprints"]["unchanged_during_gate"]
        fp = v["inputs"]["source_fingerprints"]["results_2b"]["files"]
        assert fp == 4 + len(g.REQUIRED_SPLITS)
        assert g._exit_for("READY") == 0

        # Even complete raw candidate scores from the pre-v3 schema are
        # legacy salvage only. Adding one such file to an otherwise READY
        # evidence set must make the current gate fail; it cannot become a
        # second scoring truth beside latent_eval.v3.
        legacy_path = (r2b / "results" /
                       "ev_legacy_raw_test_id_clean.json")
        legacy_ex = _suite_ex()
        legacy_path.write_text(json.dumps(_eval_for(
            "runs/E_k4_s0", "test_id", [_raw_record(legacy_ex)])))
        legacy_rejected = g.run_gate(
            r2b, r4b, repo_root=None, skip_proof_tests=True,
            proof_result=PROOF_OK)
        legacy_entry = next(
            item for item in legacy_rejected.artifact_verdicts["evaluations"]
            if item["file"] == legacy_path.name)
        assert legacy_entry["status"] \
            == g.HISTORICAL_UNBOUND_LEGACY_SCORER
        assert legacy_entry["status"] in g.EVAL_BAD_STATUSES
        assert legacy_rejected.gate_verdict["verdict"] == "NOT_READY"
        assert any(
            blocker["code"] == "EVAL_FILE_INVALID"
            and legacy_path.name in blocker["detail"]
            for blocker in legacy_rejected.gate_verdict["blockers"])
        legacy_path.unlink()

        # A different, fully valid bundle cannot inherit the genuine raw
        # best-step history even if every bundle/file/eval digest is
        # coherently refreshed around it.
        replacement = {"lora.A": torch.full((2, 2), 3.0,
                                             dtype=torch.float32)}
        swapped = save_adapter_bundle(
            run / "best_params.pt", replacement,
            model_id="Qwen/Qwen3.5-2B", revision=PINNED_REV_2B,
            recipe=rep["recipe"],
            metrics={"best_score": 1.0, "best_step": 200})
        assert adapter_state_sha256(replacement) != selected_state_sha256
        rep["checkpoint_content_digest"] = swapped["content_digest"]
        rep["checkpoint_sha256"] = sha256_file(run / "best_params.pt")
        manifest["checkpoint_content_digest"] = swapped["content_digest"]
        write_train_generation(run, manifest=manifest, report=rep)
        for split in g.REQUIRED_SPLITS:
            records = _full_split_v3_records(
                split, swapped["content_digest"])
            (r2b / "results" /
             f"ev_E_k4_s0_{split}_clean.json").write_text(
                json.dumps(_v3_eval_for("runs/E_k4_s0", split, records)))

        rejected = g.run_gate(
            r2b, r4b, repo_root=None, skip_proof_tests=True,
            proof_result=PROOF_OK)
        assert rejected.gate_verdict["verdict"] != "READY"
        current = next(
            run_verdict for run_verdict
            in rejected.artifact_verdicts["runs"]
            if run_verdict["run_id"] == "E_k4_s0")
        assert current["generation_state_binding_valid"] is False
        assert current["checkpoint"]["classification"] == "invalid"

    def test_canonical_outputs_byte_stable_and_timestamp_free(
            self, corpus, tmp_path):
        r2b, r4b = corpus
        outs = []
        for i in (1, 2):
            res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True)
            d = tmp_path / f"out{i}"
            d.mkdir()
            g.write_canonical(d / "artifact_inventory.json", res.inventory)
            g.write_canonical(d / "artifact_verdicts.json",
                              res.artifact_verdicts)
            g.write_canonical(d / "gate_verdict.json", res.gate_verdict)
            (d / "GATE_REPORT.md").write_text(res.report_md)
            outs.append(d)

        names = ["artifact_inventory.json", "artifact_verdicts.json",
                 "gate_verdict.json", "GATE_REPORT.md"]
        for name in names:
            assert (outs[0] / name).read_bytes() == \
                (outs[1] / name).read_bytes(), name

        for name in names:
            raw = (outs[0] / name).read_text()
            assert not ISO_TS.search(raw), f"{name} leaks a wall-clock value"
            if name.endswith(".json"):
                doc = _strict_loads((outs[0] / name).read_bytes())
                assert isinstance(doc, dict)

    def test_source_fingerprint_detects_input_mutation(self, corpus):
        r2b, r4b = corpus
        fp1 = g.source_fingerprint(r2b)
        (r2b / "results" / "ev_E_k4_s0_test_id_clean.json").write_text("{}")
        fp2 = g.source_fingerprint(r2b)
        assert fp1 != fp2
        assert fp1["merkle_sha256"] != fp2["merkle_sha256"]

    def test_dry_run_skips_payload_loading_but_still_hashes(self, corpus):
        res = self.run_gate(corpus, dry_run=True)
        runs = res.artifact_verdicts["runs"]
        ck = [r["checkpoint"] for r in runs if r.get("checkpoint")]
        assert all(c["classification"] == "unproven" for c in ck)
        assert res.gate_verdict["counts"]["files_scanned"] == 10
        assert res.gate_verdict["counts"]["eval_files_checked"] == 0
        # dry-run can never be READY: payload-dependent prereqs UNPROVEN
        assert res.verdict == "NOT_READY"


# ---------------------------------------------------------------------------
# CLI exit semantics
# ---------------------------------------------------------------------------

class TestExitSemantics:
    def test_exit_mapping(self):
        assert g._exit_for("READY") == 0
        assert g._exit_for("NOT_READY") == 1
        with pytest.raises(ValueError):
            g._exit_for("SOMETHING_ELSE")

    def test_paths_overlap_helper(self, tmp_path):
        a = tmp_path / "in"
        b = tmp_path / "in" / "sub"
        c = tmp_path / "other"
        assert g.paths_overlap(a, a)
        assert g.paths_overlap(a, b) and g.paths_overlap(b, a)
        assert not g.paths_overlap(a, c)

    def test_out_equal_to_input_root_rejected_before_writing(
            self, corpus, capsys):
        r2b, r4b = corpus
        before = sorted(str(p) for p in r2b.rglob("*"))
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(r2b), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 2
        assert "overlaps an input root" in capsys.readouterr().err
        assert sorted(str(p) for p in r2b.rglob("*")) == before

    def test_out_inside_input_root_rejected(self, corpus, tmp_path, capsys):
        r2b, r4b = corpus
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(r2b / "results" / "gate_out"),
                     "--skip-proof-tests", "--no-telemetry"])
        assert rc == 2
        assert not (r2b / "results" / "gate_out").exists()

    def test_out_containing_input_root_rejected(self, corpus, tmp_path,
                                                capsys):
        r2b, r4b = corpus
        # tmp_path itself CONTAINS both input roots -> must be refused
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(tmp_path),
                     "--skip-proof-tests", "--no-telemetry"])
        assert rc == 2
        assert "overlaps" in capsys.readouterr().err
        assert not (tmp_path / "gate_verdict.json").exists()

    def test_missing_input_roots_is_execution_error(self, tmp_path, capsys):
        rc = g.main(["--results-2b", str(tmp_path / "nope"),
                     "--results-4b", str(tmp_path / "nada"),
                     "--out", str(tmp_path / "o"), "--no-telemetry"])
        assert rc == 2
        assert "execution error" in capsys.readouterr().err

    def test_unreadable_scan_file_is_execution_error_exit2(
            self, corpus, tmp_path, monkeypatch, capsys):
        r2b, r4b = corpus
        calls = {"n": 0}
        real_sha = g.sha256_file

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 3:      # first fingerprint pass hits file 3
                raise OSError("permission denied (simulated)")
            return real_sha(path)

        monkeypatch.setattr(g, "sha256_file", flaky)
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(tmp_path / "o1"),
                     "--skip-proof-tests", "--no-telemetry"])
        assert rc == 2
        assert "execution error" in capsys.readouterr().err

    def test_inputs_modified_mid_gate_is_execution_error(
            self, corpus, monkeypatch, capsys, tmp_path):
        r2b, r4b = corpus
        real_fp = g.source_fingerprint
        state = {"n": 0}

        def mutating_fp(root):
            state["n"] += 1
            if state["n"] == 3:
                # second pass, first root: mutate just before re-hashing so
                # the after-fingerprint diverges from the before-fingerprint
                (root / "results" / "ev_E_k4_s0_test_id_clean.json") \
                    .write_text('{"adapter": "runs/E_k4_s0", "split": '
                                '"test_id", "model": "m", "revision": "r", '
                                '"suite_sha256": "s", "results": {}}')
            return real_fp(root)

        monkeypatch.setattr(g, "source_fingerprint", mutating_fp)
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(tmp_path / "o2"),
                     "--skip-proof-tests", "--no-telemetry"])
        assert rc == 2
        assert "changed while the gate was running" in \
            capsys.readouterr().err

    def test_main_end_to_end_not_ready_writes_artifacts(
            self, corpus, tmp_path, capsys):
        r2b, r4b = corpus
        out = tmp_path / "gate_out"
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(out), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 1
        for name in ("artifact_inventory.json", "artifact_verdicts.json",
                     "gate_verdict.json", "GATE_REPORT.md"):
            assert (out / name).is_file(), name
        v = json.loads((out / "gate_verdict.json").read_text())
        assert v["verdict"] == "NOT_READY"
        assert "NOT_READY" in capsys.readouterr().out

    def test_telemetry_timestamp_is_separate_and_optional(
            self, corpus, tmp_path):
        r2b, r4b = corpus
        out = tmp_path / "g2"
        g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                "--out", str(out), "--skip-proof-tests"])
        tel = json.loads((out / "telemetry_timestamp.json").read_text())
        assert set(tel) == {"note", "telemetry_generated_at_utc"}
        verdict_text = (out / "gate_verdict.json").read_text()
        assert "telemetry_generated_at_utc" not in verdict_text

        out3 = tmp_path / "g3"
        rc = g.main(["--results-2b", str(r2b), "--results-4b", str(r4b),
                     "--out", str(out3), "--skip-proof-tests",
                     "--no-telemetry"])
        assert rc == 1
        assert not (out3 / "telemetry_timestamp.json").exists()

    def test_math_domain_no_nan_leaks_into_canonical_json(self, corpus):
        r2b, r4b = corpus
        res = g.run_gate(r2b, r4b, repo_root=None, skip_proof_tests=True)
        for doc in (res.inventory, res.artifact_verdicts,
                    res.gate_verdict):
            # strict parser raises on any bare NaN/Infinity literal
            assert isinstance(_strict_loads(g.canonical_json_bytes(doc)),
                              dict)
        blob = g.canonical_json_bytes(res.artifact_verdicts)
        assert b": NaN" not in blob and b": Infinity" not in blob \
            and b": -Infinity" not in blob
        assert math.isfinite(1.0)  # sanity on import side-effects-free use
