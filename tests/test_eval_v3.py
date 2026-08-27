from __future__ import annotations

import copy
import hashlib
import math

import pytest

from latent_lab.bench.eval_v3 import (
    ERROR_AMBIGUOUS_TOP_TIE,
    PRIMARY_SCORE_DEFINITION,
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_NONTERMINATION,
    TIE_POLICY,
    EvalV3Error,
    aggregate_records,
    build_error_record,
    build_eval_record,
    checkpoint_history_entry,
    paired_comparison,
    rescore_record,
    score_candidates,
    scorer_identity,
    select_best_checkpoint,
    validate_record,
    validate_record_against_current_suite,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _compute(**over):
    out = {
        "prefill_layers": 12,
        "recurrence_interval_applications": 4,
        "k_loops": 2,
        "candidate_tail_layers": 8,
        "lm_head_calls": 4,
        "tokenizer_calls": 1,
        "decode_calls": 0,
        "wall_seconds": 0.25,
        "peak_memory_bytes": None,
        "successful_task": True,
    }
    out.update(over)
    return out


def _metadata(
    *,
    example_id="e0",
    prompt=None,
    family="fsm",
    candidates=("A", "B", "C"),
    gold_answer="B",
    candidate_permutation=(0, 1, 2),
    candidate_permutation_seed=17,
    run_id="run-1",
    checkpoint_id="step-10",
    split="validation",
    k=2,
    compute=None,
    recurrence_config=None,
):
    return {
        "run_id": run_id,
        "recipe_hash": SHA_A,
        "model_id": "tiny-model",
        "model_revision": "revision-1",
        "adapter_id": "adapter-1",
        "checkpoint_id": checkpoint_id,
        "checkpoint_content_hash": SHA_B,
        "suite_id": "behavioral-v3",
        "suite_version": 3,
        "suite_hash": SHA_C,
        "example_id": example_id,
        "split": split,
        "family": family,
        "prompt": prompt or f"prompt:{example_id}",
        "candidates": candidates,
        "candidate_permutation_seed": candidate_permutation_seed,
        "candidate_permutation": candidate_permutation,
        "gold_answer": gold_answer,
        "k": k,
        "recurrence_config": recurrence_config
        or {"interval": [2, 4], "gradient_semantics": "truncated_cache"},
        "compute": compute or _compute(),
    }


def _record(*, scores=((-3.0,), (-0.1,), (-2.0,)), **metadata):
    return build_eval_record(per_token_logprobs=scores, **_metadata(**metadata))


def _current_suite_record(**over):
    from latent_lab.bench.suite_v3 import build_suite

    suite = build_suite()
    example = suite.validation[0]
    metadata = {
        "run_id": "bound-run",
        "recipe_hash": SHA_A,
        "model_id": "tiny-model",
        "model_revision": "revision-1",
        "adapter_id": "bound-adapter",
        "checkpoint_id": "step-10",
        "checkpoint_content_hash": SHA_B,
        "suite_id": "behavioral-v3",
        "suite_version": 3,
        "suite_hash": suite.records_hash(),
        "example_id": example.ex_id,
        "split": example.split,
        "family": example.family,
        "prompt": example.prompt,
        "candidates": example.candidates,
        "candidate_permutation_seed": example.candidate_permutation_seed,
        "candidate_permutation": example.candidate_permutation,
        "gold_answer": example.answer,
        "k": 2,
        "recurrence_config": {
            "interval": [2, 4], "gradient_semantics": "truncated_cache"
        },
        "compute": _compute(),
    }
    metadata.update(over)
    scores = tuple(
        (-0.1,) if candidate == metadata["gold_answer"] else (-2.0,)
        for candidate in metadata["candidates"]
    )
    return example, build_eval_record(
        per_token_logprobs=scores, **metadata)


@pytest.mark.parametrize("gold_index", range(4))
def test_gold_at_every_candidate_position_is_derived_from_answer(gold_index):
    candidates = tuple(f"c{i}" for i in range(4))
    rows = tuple(((-0.01,) if i == gold_index else (-2.0 - i,)) for i in range(4))
    record = _record(
        candidates=candidates,
        gold_answer=candidates[gold_index],
        candidate_permutation=(2, 0, 3, 1),
        scores=rows,
    )
    assert record["gold_index"] == gold_index
    assert record["predicted_answer"] == candidates[gold_index]
    assert record["correctness"] is True
    assert validate_record(record) == record


def test_candidate_permutation_preserves_semantic_verdict():
    canonical = ("red", "green", "blue")
    raw_by_answer = {"red": (-2.0,), "green": (-0.2,), "blue": (-1.0,)}
    first = _record(
        candidates=canonical,
        gold_answer="green",
        candidate_permutation=(0, 1, 2),
        scores=tuple(raw_by_answer[c] for c in canonical),
    )
    permuted = ("blue", "red", "green")
    second = _record(
        example_id="e1",
        candidates=permuted,
        gold_answer="green",
        candidate_permutation=(2, 0, 1),
        candidate_permutation_seed=29,
        scores=tuple(raw_by_answer[c] for c in permuted),
    )
    assert first["gold_index"] != second["gold_index"]
    assert first["predicted_answer"] == second["predicted_answer"] == "green"
    assert first["correctness"] == second["correctness"] is True


def test_current_evidence_record_binds_to_exact_canonical_example():
    example, record = _current_suite_record()
    assert validate_record_against_current_suite(
        record, expected_split="validation") == record
    assert record["example_id"] == example.ex_id


@pytest.mark.parametrize(
    "field,overrides",
    [
        ("family", {"family": "fabricated-family"}),
        ("prompt_hash", {"prompt": "fabricated prompt"}),
        ("candidate_permutation_seed", {"candidate_permutation_seed": -991}),
    ],
)
def test_real_example_id_and_suite_hash_cannot_bind_fabricated_fields(
        field, overrides):
    _, record = _current_suite_record(**overrides)
    with pytest.raises(EvalV3Error, match=field):
        validate_record_against_current_suite(
            record, expected_split="validation")


def test_current_suite_binding_rejects_internally_valid_candidate_gold_and_order(
):
    example, _ = _current_suite_record()
    candidates = list(example.candidates)
    non_gold = next(i for i, value in enumerate(candidates)
                    if value != example.answer)
    candidates[non_gold] = "fabricated-candidate"
    _, fabricated_candidates = _current_suite_record(
        candidates=tuple(candidates))

    wrong_gold = next(value for value in example.candidates
                      if value != example.answer)
    _, fabricated_gold = _current_suite_record(gold_answer=wrong_gold)

    swapped_candidates = tuple(reversed(example.candidates))
    _, corrupted_order = _current_suite_record(
        candidates=swapped_candidates,
        candidate_permutation=tuple(reversed(example.candidate_permutation)),
    )
    _, corrupted_permutation = _current_suite_record(
        candidate_permutation=tuple(reversed(example.candidate_permutation)))

    for record, expected_field in (
        (fabricated_candidates, "candidates"),
        (fabricated_gold, "gold_answer"),
        (corrupted_order, "candidates"),
        (corrupted_permutation, "candidate_permutation"),
    ):
        with pytest.raises(EvalV3Error, match=expected_field):
            validate_record_against_current_suite(
                record, expected_split="validation")


def test_unequal_token_lengths_use_preregistered_mean_not_raw_sum():
    result = score_candidates(
        ("short", "long"),
        "long",
        ((-0.4,), (-0.3, -0.3)),
    )
    assert result.raw_summed_logprobs == (-0.4, -0.6)
    assert result.normalized_scores == (-0.4, -0.3)
    assert result.ranking == (1, 0)
    assert result.correctness is True


def test_exact_top_tie_is_error_and_retains_complete_ranking_and_tie():
    record = _record(scores=((-0.5,), (-0.5,), (-2.0,)))
    assert record["status"] == STATUS_ERROR
    assert record["error_status"] == ERROR_AMBIGUOUS_TOP_TIE
    assert record["predicted_answer"] is None
    assert record["correctness"] is False
    assert len(record["ranking"]) == 3
    assert record["ties"] == [[0, 1]]
    validate_record(record)


def test_near_tie_is_not_an_exact_tie():
    record = _record(scores=((-0.5,), (-0.5000000000000001,), (-2.0,)))
    assert record["status"] == "ok"
    assert record["ties"] == []
    assert record["predicted_answer"] == "A"


@pytest.mark.parametrize(
    "candidates,gold,rows,match",
    [
        (("A", "A"), "A", ((-1.0,), (-2.0,)), "duplicate"),
        (("A", "B"), "missing", ((-1.0,), (-2.0,)), "absent"),
        (("A", "B"), "A", ((math.nan,), (-2.0,)), "not finite"),
        (("A", "B"), "A", ((-1.0,), (math.inf,)), "not finite"),
        (("A", "B"), "A", ((-1.0,), ()), "at least one"),
    ],
)
def test_duplicate_missing_nonfinite_and_empty_raw_are_rejected(candidates, gold, rows, match):
    with pytest.raises(EvalV3Error, match=match):
        score_candidates(candidates, gold, rows)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda record: record.pop("recipe_hash"), "keys mismatch"),
        (lambda record: record.update(candidate_permutation=[0, 0, 2]), "complete permutation"),
        (lambda record: record.update(gold_index=0), "gold_index disagrees"),
        (lambda record: record.update(ranking=[2, 1, 0]), "ranking disagrees"),
        (lambda record: record.update(primary_score_definition="sum"), "not canonical"),
        (lambda record: record.update(tie_policy="first-index"), "not canonical"),
        (lambda record: record["scorer"].update(sha256="0" * 64), "scorer identity"),
    ],
)
def test_missing_and_corrupted_derived_or_order_fields_fail_closed(mutation, match):
    record = _record()
    mutation(record)
    with pytest.raises(EvalV3Error, match=match):
        validate_record(record)


def test_independent_rescore_matches_every_persisted_verdict_field_exactly():
    record = _record(scores=((-1.1, -0.9), (-0.2, -0.4, -0.3), (-1.0,)))
    rescored = rescore_record(record)
    assert {field: record[field] for field in rescored} == rescored
    assert record["primary_score_definition"] == PRIMARY_SCORE_DEFINITION
    assert record["tie_policy"] == TIE_POLICY


def test_record_hash_catches_raw_evidence_mutation_even_if_derived_is_unchanged():
    record = _record()
    record["per_token_logprobs"][2][0] = -2.5
    with pytest.raises(EvalV3Error):
        validate_record(record)


def test_error_record_counts_as_error_and_incorrect_without_invented_scores():
    record = build_error_record(
        status=STATUS_NONTERMINATION,
        error_status="TIMEOUT",
        error_detail="bounded local timeout",
        **_metadata(compute=_compute(successful_task=False)),
    )
    assert record["per_token_logprobs"] == [[], [], []]
    assert record["correctness"] is False
    validate_record(record)
    summary = aggregate_records([record])
    assert summary["micro_accuracy"] == 0.0
    assert summary["nontermination_error_rate"] == 1.0


def test_aggregate_metrics_cover_micro_macro_family_chance_and_errors():
    records = [
        _record(example_id="a", family="f1"),
        _record(example_id="b", family="f1", scores=((-0.1,), (-1.0,), (-2.0,))),
        _record(
            example_id="c",
            family="f2",
            candidates=("A", "B"),
            gold_answer="B",
            candidate_permutation=(0, 1),
            scores=((-2.0,), (-0.1,)),
        ),
    ]
    summary = aggregate_records(records)
    assert summary["micro_accuracy"] == pytest.approx(2 / 3)
    assert summary["per_family_accuracy"] == {"f1": 0.5, "f2": 1.0}
    assert summary["macro_by_family_accuracy"] == 0.75
    assert summary["mean_uniform_chance"] == pytest.approx((1 / 3 + 1 / 3 + 1 / 2) / 3)
    expected = (summary["micro_accuracy"] - summary["mean_uniform_chance"]) / (
        1 - summary["mean_uniform_chance"]
    )
    assert summary["chance_normalized_accuracy"] == pytest.approx(expected)


def test_paired_bootstrap_is_deterministic_and_requires_identical_examples():
    control = [
        _record(example_id="a", scores=((-0.1,), (-1.0,), (-2.0,))),
        _record(example_id="b"),
    ]
    treatment = [
        _record(example_id="a"),
        _record(example_id="b"),
    ]
    first = paired_comparison(treatment, control, bootstrap_seed=91, bootstrap_samples=200)
    second = paired_comparison(treatment, control, bootstrap_seed=91, bootstrap_samples=200)
    assert first == second
    assert first["accuracy_delta"] == 0.5
    with pytest.raises(EvalV3Error, match="same non-empty example ids"):
        paired_comparison(treatment, control[:1])
    with pytest.raises(EvalV3Error, match="duplicate example_id"):
        paired_comparison(treatment + treatment[:1], control)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"prompt": "different prompt"}, "prompt_hash"),
        ({"candidate_permutation_seed": 99}, "candidate_permutation_seed"),
        ({"candidate_permutation": (2, 1, 0)}, "candidate_permutation"),
    ],
)
def test_paired_bootstrap_rejects_mismatched_prompt_or_permutation(
        overrides, field):
    treatment = [_record(example_id="a")]
    control = [_record(example_id="a", **overrides)]
    with pytest.raises(EvalV3Error, match=field):
        paired_comparison(treatment, control, bootstrap_samples=10)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"run_id": "different-run"}, "run_id"),
        ({"checkpoint_id": "step-99"}, "checkpoint_identity"),
        ({"recurrence_config": {"interval": [0, 4]}},
         "recurrence_config"),
    ],
)
def test_paired_bootstrap_rejects_different_mechanism_identity(
        overrides, field):
    treatment = [_record(example_id="a")]
    control = [_record(example_id="a", **overrides)]
    with pytest.raises(EvalV3Error, match=field):
        paired_comparison(treatment, control, bootstrap_samples=10)


def test_selector_accepts_only_canonical_summaries_and_earliest_tie():
    records = [_record()]
    low = checkpoint_history_entry(10, records)
    high = checkpoint_history_entry(20, records)
    legacy = {"step": 1, "accuracy": 1.0}
    result = select_best_checkpoint([legacy, high, low])
    assert result is not None
    assert result.step == 10
    assert result.metric == 1.0
    assert result.n_considered == 2
    assert result.n_rejected == 1


def test_selector_rejects_tampered_or_nonfinite_summary():
    entry = checkpoint_history_entry(3, [_record()])
    poisoned = copy.deepcopy(entry)
    poisoned["metrics"]["micro_accuracy"] = math.nan
    wrong_scorer = copy.deepcopy(entry)
    wrong_scorer["metrics"]["scorer"]["sha256"] = "0" * 64
    assert select_best_checkpoint([poisoned, wrong_scorer]) is None


def test_unsupported_grad_checkpoint_and_missing_compute_counter_fail_closed():
    with pytest.raises(EvalV3Error, match="grad_checkpoint"):
        _record(recurrence_config={"grad_checkpoint": False})
    compute = _compute()
    del compute["candidate_tail_layers"]
    with pytest.raises(EvalV3Error, match="missing required counters"):
        _record(compute=compute)


def test_scorer_identity_hashes_the_actual_module_source():
    identity = scorer_identity()
    assert identity["implementation"] == "latent_lab.bench.eval_v3"
    assert identity["version"] == "3"
    assert len(identity["sha256"]) == 64
    assert set(identity["sha256"]) <= set("0123456789abcdef")


def test_prompt_hash_is_retained_but_prompt_text_is_not():
    record = _record()
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["prompt_hash"] == hashlib.sha256(b"prompt:e0").hexdigest()
    assert "prompt" not in record
