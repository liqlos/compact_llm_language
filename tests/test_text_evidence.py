"""Raw-evidence and offline-rescore contracts for textual baselines."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from latent_lab.bench.suite_v3 import build_suite
from latent_lab.bench.text_baselines import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    EVALUATION_SPLITS,
    TEXT_RECORD_SCHEMA,
    TEXT_REPORT_SCHEMA,
    TextEvidenceError,
    _record_hash,
    _report_hash,
    _r1_report_eligible,
    _sha256_json,
    _sha256_text,
    _strict_json_loads,
    _write_report,
    build_prompt,
    make_text_record,
    make_text_report,
    offline_rescore_record,
    offline_rescore_report,
    parser_identity,
    producer_identity,
    prompt_messages,
)


@pytest.fixture(scope="module")
def suite_and_example():
    suite = build_suite()
    return suite, suite.validation[0]


@pytest.fixture()
def raw_record(suite_and_example):
    suite, example = suite_and_example
    generated = ("concise scratch\n" * 40) + f"Answer: {example.answer}<|im_end|>"
    candidate_ids = [[100 + index] for index in range(len(example.candidates))]
    return make_text_record(
        run_id="text-A-seed-0-validation",
        suite_manifest=suite.manifest(),
        ex=example,
        mode="A",
        seed=0,
        model_identity={
            "model_id": "Qwen/Qwen3.5-2B",
            "revision": "1" * 40,
            "model_class": "FakeModel",
            "dtype": "bfloat16",
            "device": "cpu",
        },
        tokenizer_identity={
            "tokenizer_id": "fake-tokenizer",
            "tokenizer_class": "FakeTokenizer",
            "revision": "1" * 40,
            "resolved_commit": "1" * 40,
            "vocab_size": 256,
            "model_max_length": 4096,
            "pad_token_id": 0,
            "eos_token_ids": [99],
            "chat_template_sha256": "2" * 64,
        },
        messages=prompt_messages(example, "A"),
        input_ids=[11, 12, 13],
        candidate_token_ids=candidate_ids,
        generated_text=generated,
        generated_token_ids=[21, 22, 99],
        max_new_tokens=64,
        error_status=None,
        compute={
            "batch_index": 0,
            "batch_size": 1,
            "batch_generate_wall_seconds": 0.25,
            "allocated_generate_wall_seconds": 0.25,
            "prefill_tokens": 3,
            "generated_tokens": 3,
            "candidate_tokenizer_calls": len(example.candidates),
            "prompt_tokenizer_calls": 1,
            "decode_calls": 1,
            "successful_task": True,
        },
    )


def _report(record, suite_and_example):
    suite, _example = suite_and_example
    return make_text_report(
        run_id=record["run_id"],
        suite_manifest=suite.manifest(),
        mode="A",
        split="validation",
        seed=0,
        model_identity=record["model_identity"],
        tokenizer_identity=record["tokenizer_identity"],
        max_new_tokens=64,
        records=[record],
        wall_seconds=0.3,
        peak_memory_bytes=1024,
        model_generate_calls=1,
    )


def _full_validation_report(suite, execution_context=None):
    model_identity = {
        "model_id": DEFAULT_MODEL_ID,
        "revision": DEFAULT_REVISION,
        "model_class": "PinnedFakeModel",
        "dtype": "torch.bfloat16",
        "device": "cuda",
    }
    tokenizer_identity = {
        "tokenizer_id": DEFAULT_MODEL_ID,
        "tokenizer_class": "PinnedFakeTokenizer",
        "revision": DEFAULT_REVISION,
        "resolved_commit": DEFAULT_REVISION,
        "vocab_size": 256,
        "model_max_length": 4096,
        "pad_token_id": 0,
        "eos_token_ids": [99],
        "chat_template_sha256": "2" * 64,
    }
    records = []
    for index, example in enumerate(suite.validation):
        records.append(make_text_record(
            run_id="r1-direct-seed-0-validation",
            suite_manifest=suite.manifest(),
            ex=example,
            mode="A",
            seed=0,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            messages=prompt_messages(example, "A"),
            input_ids=[index + 1],
            candidate_token_ids=[
                [candidate_index + 1]
                for candidate_index, _candidate in enumerate(example.candidates)
            ],
            generated_text=f"Answer: {example.answer}<|im_end|>",
            generated_token_ids=[99],
            max_new_tokens=64,
            error_status=None,
            compute={
                "batch_index": index,
                "batch_size": 1,
                "batch_generate_wall_seconds": 0.0,
                "allocated_generate_wall_seconds": 0.0,
                "prefill_tokens": 1,
                "generated_tokens": 1,
                "candidate_tokenizer_calls": len(example.candidates),
                "prompt_tokenizer_calls": 1,
                "decode_calls": 1,
                "successful_task": True,
            },
            execution_context=execution_context,
        ))
    return make_text_report(
        run_id="r1-direct-seed-0-validation",
        suite_manifest=suite.manifest(),
        mode="A",
        split="validation",
        seed=0,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        max_new_tokens=64,
        records=records,
        wall_seconds=0.1,
        peak_memory_bytes=1024,
        model_generate_calls=len(records),
        execution_context=execution_context,
    )


def test_v3_text_record_retains_full_raw_generation_and_candidate_binding(
        raw_record, suite_and_example):
    _suite, example = suite_and_example
    assert raw_record["schema_version"] == TEXT_RECORD_SCHEMA
    assert len(raw_record["generated_text"]) > 400
    assert "generated_preview" not in raw_record
    assert raw_record["generated_token_ids"] == [21, 22, 99]
    assert raw_record["input_ids"] == [11, 12, 13]
    assert raw_record["candidates_in_actual_order"] == list(example.candidates)
    assert raw_record["canonical_candidates"] == list(
        example.canonical_candidates)
    assert raw_record["candidate_permutation"] == list(
        example.candidate_permutation)
    assert raw_record["candidate_permutation_seed"] == (
        example.candidate_permutation_seed)
    assert raw_record["derived_gold_index"] == example.gold_index
    assert raw_record["correctness"] is True
    assert len(raw_record["record_hash"]) == 64
    assert raw_record["parser"] == parser_identity()


def test_online_and_offline_verdicts_agree_exactly(raw_record):
    verdict = offline_rescore_record(raw_record)
    assert verdict == {
        key: raw_record[key]
        for key in (
            "terminated", "termination_status", "parser_status",
            "parsed_answer", "correctness",
        )
    }


def test_parser_identity_is_explicit_and_hash_bound():
    identity = parser_identity()
    assert set(identity) == {"implementation", "version", "sha256"}
    assert len(identity["sha256"]) == 64
    int(identity["sha256"], 16)


@pytest.mark.parametrize("missing", [
    "generated_text",
    "generated_token_ids",
    "input_ids",
    "benchmark_prompt",
    "prompt_messages",
])
def test_missing_raw_fields_fail_closed(raw_record, missing):
    corrupt = copy.deepcopy(raw_record)
    corrupt.pop(missing)
    with pytest.raises(TextEvidenceError, match="missing raw evidence"):
        offline_rescore_record(corrupt)


def test_corrupt_raw_generation_fails_even_if_attacker_rehashes_record(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["generated_text"] = "Answer: definitely-wrong<|im_end|>"
    corrupt["generated_text_sha256"] = _sha256_text(corrupt["generated_text"])
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match="stored verdict disagrees"):
        offline_rescore_record(corrupt)


def test_corrupt_candidate_order_fails_even_with_new_record_hash(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["candidates_in_actual_order"][0], corrupt["candidates_in_actual_order"][1] = (
        corrupt["candidates_in_actual_order"][1],
        corrupt["candidates_in_actual_order"][0],
    )
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(
        TextEvidenceError,
        match=("derived gold index mismatch|candidate order disagrees|"
               "canonical suite example")):
        offline_rescore_record(corrupt)


def test_forged_eos_token_id_cannot_turn_nonterminated_text_into_pass(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["generated_text"] = raw_record["generated_text"].split("<|im_end|>")[0]
    corrupt["generated_text_sha256"] = _sha256_text(corrupt["generated_text"])
    # The configured EOS id remains present, but decoded raw text has no
    # terminal marker and is therefore non-terminating.
    assert corrupt["generated_token_ids"][-1] == 99
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match="stored verdict disagrees"):
        offline_rescore_record(corrupt)


def test_generated_tokens_cannot_exceed_declared_budget(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["generated_token_ids"] = list(range(65))
    corrupt["generated_token_ids_sha256"] = _sha256_json(
        corrupt["generated_token_ids"])
    corrupt["generated_token_count"] = 65
    corrupt["compute"]["generated_tokens"] = 65
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match="exceeds generation budget"):
        offline_rescore_record(corrupt)


def test_self_consistent_forged_gold_is_rejected_by_canonical_suite(raw_record):
    corrupt = copy.deepcopy(raw_record)
    forged_index = (corrupt["derived_gold_index"] + 1) % len(
        corrupt["candidates_in_actual_order"])
    forged_answer = corrupt["candidates_in_actual_order"][forged_index]
    corrupt["gold_answer"] = forged_answer
    corrupt["derived_gold_index"] = forged_index
    corrupt["generated_text"] = f"Answer: {forged_answer}<|im_end|>"
    corrupt["generated_text_sha256"] = _sha256_text(corrupt["generated_text"])
    corrupt["correctness"] = True
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match="canonical suite example"):
        offline_rescore_record(corrupt)


def test_mode_relabel_is_rejected_by_actual_prompt_binding(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["mode"] = "C"
    corrupt["mode_description"] = (
        "concise visible scratch with native thinking disabled")
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match="prompt messages disagree"):
        offline_rescore_record(corrupt)


def test_tokenizer_resolved_commit_must_match_revision(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["tokenizer_identity"]["resolved_commit"] = "3" * 40
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match="resolved commit disagrees"):
        offline_rescore_record(corrupt)


def test_r1_structure_rejects_unregistered_seed_even_with_pinned_identity(
        suite_and_example):
    suite, _example = suite_and_example
    context = {
        "plan_hash": "a" * 64,
        "command_index": 1,
        "driver_source_sha256": "b" * 64,
        "producer_source_sha256": producer_identity()["source_sha256"],
        "suite_hash": suite.manifest()["suite_hash"],
    }
    assert not _r1_report_eligible(
        coverage={"status": "FULL_SPLIT"}, mode="A", max_new_tokens=64,
        model_identity={
            "model_id": DEFAULT_MODEL_ID, "revision": DEFAULT_REVISION,
            "device": "cuda", "dtype": "torch.bfloat16",
        },
        tokenizer_identity={
            "tokenizer_id": DEFAULT_MODEL_ID, "revision": DEFAULT_REVISION,
            "resolved_commit": DEFAULT_REVISION,
            "chat_template_sha256": "2" * 64,
        },
        seed=999, execution_context=context,
        suite_hash=suite.manifest()["suite_hash"],
    )


def test_report_is_distinct_schema_and_rescores_from_raw(raw_record,
                                                          suite_and_example):
    report = _report(raw_record, suite_and_example)
    assert report["schema_version"] == TEXT_REPORT_SCHEMA
    assert report["record_schema_version"] == TEXT_RECORD_SCHEMA
    assert report["selection_eligible"] is False
    assert report["validation_role"] == "comparison_only_no_checkpoint_selection"
    assert "accuracy" not in json.dumps(report, sort_keys=True)
    rescored = offline_rescore_report(report)
    assert rescored["status"] == "VALID_SMOKE_ONLY_NOT_HEADLINE_ELIGIBLE"
    assert rescored["r1_preregistered_evidence_eligible"] is False
    assert report["coverage"]["status"] == "SMOKE_PREFIX_ONLY"
    assert rescored["derived_metrics"] == report["derived_metrics"]
    assert report["derived_metrics"]["exact_match_rate"] == 1.0


def test_full_report_needs_plan_and_receipt_attestation(suite_and_example):
    suite, _example = suite_and_example
    context = {
        "plan_hash": "a" * 64,
        "command_index": 17,
        "driver_source_sha256": "b" * 64,
        "producer_source_sha256": producer_identity()["source_sha256"],
        "suite_hash": suite.manifest()["suite_hash"],
    }
    bound = _full_validation_report(suite, execution_context=context)
    rescored_bound = offline_rescore_report(bound)
    assert bound["r1_preregistered_structure_valid"] is True
    assert bound["r1_preregistered_evidence_eligible"] is False
    assert bound["requires_driver_receipt_attestation"] is True
    assert rescored_bound["status"] == (
        "VALID_FULL_RAW_EVIDENCE_PENDING_DRIVER_RECEIPT")


def test_same_plan_wrong_split_substitution_is_rejected(
        suite_and_example, monkeypatch, capsys):
    from latent_lab.bench import text_baselines

    suite, _example = suite_and_example
    context = {
        "plan_hash": "a" * 64,
        "command_index": 17,
        "driver_source_sha256": "b" * 64,
        "producer_source_sha256": producer_identity()["source_sha256"],
        "suite_hash": suite.manifest()["suite_hash"],
    }
    result = {
        "schema_version": "text_generation_rescore.v1",
        "run_id": "r1-direct-seed-0-validation",
        "mode": "A",
        "seed": 0,
        "split": "validation",
        "r1_preregistered_structure_valid": True,
        "r1_preregistered_evidence_eligible": False,
        "execution_context": context,
        "status": "VALID_FULL_RAW_EVIDENCE_PENDING_DRIVER_RECEIPT",
    }
    monkeypatch.setattr(text_baselines, "load_and_rescore", lambda _path: result)
    monkeypatch.setenv("RCC_R1_ACTIVE_PLAN_HASH", context["plan_hash"])
    monkeypatch.setenv("RCC_R1_ACTIVE_COMMAND_INDEX", "99")
    monkeypatch.setenv(
        "RCC_R1_DRIVER_SOURCE_SHA256", context["driver_source_sha256"])
    monkeypatch.setenv(
        "RCC_R1_TEXT_PRODUCER_SOURCE_SHA256",
        context["producer_source_sha256"],
    )
    monkeypatch.setenv("RCC_R1_DRIVER_SUITE_HASH", context["suite_hash"])
    expected = [
        "--rescore", "unused.json", "--require-r1-preregistered",
        "--expect-producer-command-index", "17", "--expect-mode", "A",
        "--expect-seed", "0", "--expect-split", "validation",
        "--expect-run-id", "r1-direct-seed-0-validation",
    ]
    assert text_baselines.main(expected) == 0
    capsys.readouterr()
    wrong_split = list(expected)
    wrong_split[wrong_split.index("validation")] = "final_test"
    with pytest.raises(TextEvidenceError, match="producer metadata mismatch"):
        text_baselines.main(wrong_split)


def test_duplicate_cherry_picked_examples_cannot_form_report(
        raw_record, suite_and_example):
    suite, _example = suite_and_example
    with pytest.raises(TextEvidenceError, match="complete split or.*prefix"):
        make_text_report(
            run_id=raw_record["run_id"], suite_manifest=suite.manifest(),
            mode="A", split="validation", seed=0,
            model_identity=raw_record["model_identity"],
            tokenizer_identity=raw_record["tokenizer_identity"],
            max_new_tokens=64, records=[raw_record, raw_record],
            wall_seconds=0.3, peak_memory_bytes=1,
            model_generate_calls=1,
        )


def test_report_rejects_derived_metric_tampering_after_rehash(raw_record,
                                                               suite_and_example):
    report = _report(raw_record, suite_and_example)
    report["derived_metrics"]["exact_match_rate"] = 0.0
    report["report_hash"] = _report_hash(report)
    with pytest.raises(TextEvidenceError, match="metrics disagree"):
        offline_rescore_report(report)


def test_report_rejects_false_selection_claim_even_after_rehash(
        raw_record, suite_and_example):
    report = _report(raw_record, suite_and_example)
    report["selection_eligible"] = True
    report["report_hash"] = _report_hash(report)
    with pytest.raises(TextEvidenceError, match="never checkpoint-selection"):
        offline_rescore_report(report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prefill_tokens", 2, "prefill token count mismatch"),
        ("generated_tokens", 2, "generated token count mismatch"),
        ("successful_task", False, "successful_task disagrees"),
    ],
)
def test_record_rejects_corrupt_compute_even_after_rehash(
        raw_record, field, value, message):
    corrupt = copy.deepcopy(raw_record)
    corrupt["compute"][field] = value
    corrupt["record_hash"] = _record_hash(corrupt)
    with pytest.raises(TextEvidenceError, match=message):
        offline_rescore_record(corrupt)


def test_nonfinite_compute_cannot_be_hashed_as_evidence(raw_record):
    corrupt = copy.deepcopy(raw_record)
    corrupt["compute"]["allocated_generate_wall_seconds"] = float("nan")
    with pytest.raises(TextEvidenceError, match="canonical JSON"):
        _record_hash(corrupt)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values():
    with pytest.raises(TextEvidenceError, match="duplicate JSON key"):
        _strict_json_loads('{"records":[],"records":[]}')
    with pytest.raises(TextEvidenceError, match="non-finite JSON"):
        _strict_json_loads('{"wall":NaN}')


def test_offline_file_rescore_never_imports_or_executes_model(
        raw_record, suite_and_example, tmp_path, capsys):
    from latent_lab.bench.text_baselines import main

    report = _report(raw_record, suite_and_example)
    path = tmp_path / "raw-text-evidence.json"
    _write_report(path, report)
    assert main(["--rescore", str(path)]) == 0
    assert '"status":"VALID_SMOKE_ONLY_NOT_HEADLINE_ELIGIBLE"' \
        in capsys.readouterr().out
    with pytest.raises(TextEvidenceError, match="lacks full preregistered"):
        main(["--rescore", str(path), "--require-r1-preregistered"])


def test_behavioral_v3_split_and_selection_contract_is_explicit():
    assert EVALUATION_SPLITS == (
        "validation", "test_id", "test_ood_length", "test_ood_semantic",
        "final_test",
    )


def test_visible_scratch_is_real_prompt_text_and_disables_native_thinking():
    calls = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return SimpleNamespace(input_ids=[[1, 2]])

    example = SimpleNamespace(prompt="Question\nAnswer:")
    assert build_prompt(FakeTokenizer(), example, "C") == [[1, 2]]
    messages, kwargs = calls[0]
    assert "concise visible scratch" in messages[0]["content"].lower()
    assert all(message["role"] != "_evidence" for message in messages)
    assert kwargs["enable_thinking"] is False
