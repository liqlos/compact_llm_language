"""Executable integrity contract for immutable behavioral-v3."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from latent_lab.bench.suite import gen_obj_track as gen_obj_track_v2
from latent_lab.bench.suite_v3 import (
    DEFAULT_COUNTS,
    FAMILIES,
    LENGTH_OOD_DEPTH,
    MASTER_SEED,
    SELECTION_ELIGIBLE_SPLITS,
    SUITE_IDENTITY,
    SUITE_VERSION,
    TOKENIZER_IDENTITY,
    UNTOUCHED_FINAL_SPLIT,
    _candidate_order,
    audit_suite,
    baseline_suite,
    build_suite,
    counterfactual_prompt,
    parse_prompt,
    reference_solve_prompt,
    validation_report,
    with_reversed_events,
    without_last_causal_event,
)

SUITE = build_suite()
AUDIT = audit_suite(SUITE)
ROOT = Path(__file__).resolve().parents[1]


def _v2_initial_only_answer(prompt: str) -> str:
    initial = prompt.split("Initial situation: ", 1)[1].split(".\nEvents", 1)[0]
    locations = dict(re.findall(r"(\w+) is at the (\w+)", initial))
    holders = {item: person for item, person in re.findall(r"the (\w+) belongs to (\w+)", initial)}
    where = re.search(r"Where is (\S+) at the end\?", prompt)
    if where:
        return locations[where.group(1)]
    item = re.search(r"Who has the (.+?) at the end\?", prompt).group(1)
    return holders[item]


def _v3_initial_query_value(example) -> str:
    scenario = example.scenario
    query = scenario["query"]
    if query["kind"] == "location":
        return scenario["initial_locations"][query["name"]]
    return scenario["initial_holders"][query["name"]]


def test_v2_obj_track_regression_reproduces_initial_state_leak():
    examples = [gen_obj_track_v2(random.Random(seed), 5) for seed in range(32)]
    assert all(_v2_initial_only_answer(example.prompt) == example.answer for example in examples)


def test_v3_obj_track_snapshots_true_initial_state_before_events():
    examples = [example for examples in SUITE.splits().values() for example in examples if example.family == "obj_track"]
    assert examples
    for example in examples:
        parsed = parse_prompt(example.prompt)
        assert parsed["initial_locations"] == example.scenario["initial_locations"]
        assert parsed["initial_holders"] == example.scenario["initial_holders"]
        assert _v3_initial_query_value(example) != example.answer
        assert reference_solve_prompt(example.prompt) == example.answer


@pytest.mark.parametrize("family", FAMILIES)
def test_independent_prompt_parser_and_reference_solver_agree(family):
    for examples in SUITE.splits().values():
        for example in examples:
            if example.family == family:
                assert reference_solve_prompt(example.prompt) == example.answer, example.ex_id


def test_candidate_order_metadata_roundtrips_and_gold_positions_balance():
    for split, examples in SUITE.splits().items():
        for example in examples:
            assert example.candidate_tokenizer_identity == TOKENIZER_IDENTITY
            assert len(set(example.candidates)) == len(example.candidates)
            assert tuple(
                example.canonical_candidates[index]
                for index in example.candidate_permutation
            ) == example.candidates
            assert example.candidates[example.gold_index] == example.answer
            assert example.candidate_token_lengths == tuple(
                len(candidate.encode("utf-8")) for candidate in example.candidates
            )
            record = example.to_dict()
            assert isinstance(record["candidate_permutation_seed"], int)
            assert record["candidate_permutation"] == list(example.candidate_permutation)
        for report in AUDIT["splits"][split]["gold_position_balance"].values():
            assert report["max_minus_min"] <= 1


def test_candidate_permutation_is_semantically_invariant():
    for example in SUITE.validation:
        reversed_candidates = tuple(reversed(example.candidates))
        assert reversed_candidates[reversed_candidates.index(example.answer)] == example.answer
        assert reference_solve_prompt(example.prompt) == example.answer


def test_candidate_order_rejects_duplicates_and_missing_gold():
    with pytest.raises(ValueError, match="unique"):
        _candidate_order(("a", "a"), "a", 1, 0)
    with pytest.raises(ValueError, match="exactly once"):
        _candidate_order(("a", "b"), "c", 1, 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_causal_removal_and_counterfactual_mutation_change_gold(family):
    examples = [example for example in SUITE.validation if example.family == family]
    assert examples
    for example in examples:
        assert reference_solve_prompt(without_last_causal_event(example)) != example.answer
        counterfactual, expected = counterfactual_prompt(example)
        assert parse_prompt(counterfactual)["family"] == family
        assert reference_solve_prompt(counterfactual) == expected
        assert expected != example.answer
        original_initial = {key: value for key, value in example.scenario.items() if key.startswith("initial") or key in {"start", "modulus", "kind"}}
        mutated = parse_prompt(counterfactual)
        mutated_initial = {key: value for key, value in mutated.items() if key.startswith("initial") or key in {"start", "modulus", "kind"}}
        assert mutated_initial == original_initial


@pytest.mark.parametrize("family", tuple(family for family in FAMILIES if family != "graph_walk"))
def test_event_reverse_changes_order_sensitive_families(family):
    examples = [example for example in SUITE.validation if example.family == family]
    changed = sum(reference_solve_prompt(with_reversed_events(example)) != example.answer for example in examples)
    assert changed / len(examples) >= 0.9


def test_graph_walk_declares_order_invariant_repeated_hops_but_is_causal():
    examples = [example for example in SUITE.validation if example.family == "graph_walk"]
    assert all(reference_solve_prompt(with_reversed_events(example)) == example.answer for example in examples)
    assert all(reference_solve_prompt(without_last_causal_event(example)) != example.answer for example in examples)


def test_no_direct_answer_field_or_constant_family_answer():
    for split, examples in SUITE.splits().items():
        for example in examples:
            assert f"Answer: {example.answer}" not in example.prompt
        for family in FAMILIES:
            family_answers = {example.answer for example in examples if example.family == family}
            assert len(family_answers) >= 2, (split, family)


def test_length_and_semantic_ood_are_distinct_domains():
    assert all(LENGTH_OOD_DEPTH[0] <= example.depth <= LENGTH_OOD_DEPTH[1] for example in SUITE.test_ood_length)
    assert all(example.template == "semantic_json_v1" for example in SUITE.test_ood_semantic)
    assert all(example.template == "prose_v1" for example in SUITE.test_id)
    id_words = " ".join(example.prompt for example in SUITE.test_id)
    semantic_words = " ".join(example.prompt for example in SUITE.test_ood_semantic)
    assert "amber" in id_words
    assert "acacia" in semantic_words
    assert "BEGIN_BEHAVIORAL_V3_JSON" in semantic_words


def test_final_test_is_excluded_from_selection_and_baseline_reporting():
    manifest = SUITE.manifest()
    assert manifest["checkpoint_selection_split"] == "validation"
    assert tuple(manifest["selection_eligible_splits"]) == SELECTION_ELIGIBLE_SPLITS
    assert UNTOUCHED_FINAL_SPLIT in manifest["selection_ineligible_splits"]
    assert manifest["untouched_final_test"]["checkpoint_selection_allowed"] is False
    baselines = baseline_suite(SUITE)
    assert UNTOUCHED_FINAL_SPLIT not in baselines["evaluated_splits"]
    assert UNTOUCHED_FINAL_SPLIT not in baselines["splits"]


def test_baseline_suite_contains_all_preregistered_non_model_controls():
    report = baseline_suite(SUITE)
    required = {
        "mean_per_example_uniform_chance",
        "train_majority_constant_answer",
        "candidate_position",
        "candidate_position_by_family",
        "initial_only_sha256",
        "initial_only_state_heuristic",
        "events_only_sha256",
        "events_only_lexical_heuristic",
        "shuffled_events_reference",
        "lexical_candidate_frequency",
    }
    for split in report["evaluated_splits"]:
        assert required <= set(report["splits"][split])
        assert 0 < report["splits"][split]["mean_per_example_uniform_chance"] < 0.5
        assert all(0 <= score <= 1 for score in report["splits"][split]["candidate_position"].values())


def test_audit_has_no_direct_shortcut_or_split_leakage_findings():
    assert AUDIT["ok"], AUDIT["problems"]
    for split in AUDIT["splits"].values():
        for family in split["families"].values():
            assert family["independent_solver_agreement"] == 1.0
            assert family["causal_event_removal_change_rate"] >= 0.95
            assert family["counterfactual_change_rate"] == 1.0
            assert family["initial_only_not_identifying"] is True


def test_deterministic_regeneration_and_suite_identity():
    regenerated = build_suite()
    assert regenerated.records_hash() == SUITE.records_hash()
    assert regenerated.manifest() == SUITE.manifest()
    manifest = SUITE.manifest()
    assert manifest["suite_identity"] == SUITE_IDENTITY
    assert manifest["suite_version"] == SUITE_VERSION
    assert manifest["master_seed"] == MASTER_SEED
    assert manifest["sizes"] == {
        split: per_family * len(FAMILIES)
        for split, per_family in DEFAULT_COUNTS.items()
    }
    assert len(manifest["suite_hash"]) == 64


def test_checked_in_manifest_and_validation_match_regeneration():
    manifest = json.loads((ROOT / "artifacts" / "behavioral_v3_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "artifacts" / "behavioral_v3_validation.json").read_text(encoding="utf-8"))
    assert manifest == SUITE.manifest()
    assert validation == validation_report(SUITE)
    assert validation["ok"] is True
    assert validation["final_test_integrity_audit_only"] is True
    assert validation["untouched_final_test_model_or_baseline_metrics_emitted"] is False
