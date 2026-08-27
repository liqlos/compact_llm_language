"""Regression coverage for R1 artifact truth and historical invalidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.classify_r1_artifacts import ALLOWED_CLASSES, build_outputs


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _minimal_evidence(root: Path) -> None:
    model = {
        "model": "example/model",
        "revision": "revision-1",
        "config": {"grad_checkpoint": True},
    }
    eval_record = {
        **model,
        "adapter": "runs/E_k4_s0",
        "results": {"clean": {"records": [{"correct": 1.0, "rank_of_gold": 0}]}},
    }
    text_record = {
        **model,
        "baseline": "A",
        "records": [{"correct": 1.0, "generated_preview": "truncated"}],
    }
    _write(
        root / "remote_results/results/ev_E_k4_s0_test_id_clean.json",
        json.dumps(eval_record),
    )
    _write(
        root / "remote_results/runs/E_k4_s0/train_report.json",
        json.dumps({**model, "best_val_acc": 0.5}),
    )
    _write(root / "remote_results/runs/E_k4_s0/best_params.pt", b"old-checkpoint")
    _write(
        root / "remote_results_4b/results/text4b_A_test_id.json",
        json.dumps(text_record),
    )
    nonfinite = (
        '{"model":"example/4b","revision":"rev-4b",'
        '"final_train_loss":NaN,"best_val_acc":1.0}'
    )
    for base in (
        root / "remote_results_4b/_rejected_nan_batch/E4_k1_s0",
        root
        / "remote_results_4b/quarantine/live_nan_duplicates_20260827/E4_k1_s0",
    ):
        _write(base / "train_report.json", nonfinite)
        _write(base / "best_params.pt", b"duplicate-checkpoint")
    _write(
        root / "smoke_latent/train_report.json",
        json.dumps({**model, "best_val_acc": 0.25}),
    )
    _write(root / "smoke_latent/best_params.pt", b"smoke-checkpoint")


def test_classifier_fails_closed_and_distinguishes_historical_classes(tmp_path):
    _minimal_evidence(tmp_path)
    classification, invalidation = build_outputs(tmp_path, strict=False)
    by_id = {row["artifact_id"]: row for row in classification["artifacts"]}

    legacy = by_id["2b/results/ev_E_k4_s0_test_id_clean.json"]
    assert legacy["retention_class"] == "IRRECOVERABLE_LEGACY_SCORER"
    assert legacy["independently_rescorable"] is False
    assert "raw per-token" in legacy["reason"]

    checkpoint = by_id["2b/runs/E_k4_s0/best_params.pt"]
    assert checkpoint["retention_class"] == "UNPROVEN_RUNTIME"
    assert checkpoint["checkpoint_selection_provenance"] == "selection_provenance_invalid"

    rejected = by_id["4b/_rejected_nan_batch/E4_k1_s0/train_report.json"]
    assert rejected["finite_status"] == "NONFINITE"
    assert rejected["retention_class"] == "CORRUPT_NONFINITE"

    quarantined = by_id[
        "4b/quarantine/live_nan_duplicates_20260827/E4_k1_s0/best_params.pt"
    ]
    assert quarantined["retention_class"] == "QUARANTINED"
    assert quarantined["duplicate_disposition"] == "REJECTED_DUPLICATE"
    assert invalidation["quarantine_moves"][0]["verification"] == (
        "EXACT_FILE_SET_SIZE_AND_SHA256_MATCH"
    )


def test_classifier_rejects_a_quarantined_run_left_in_live_root(tmp_path):
    _minimal_evidence(tmp_path)
    _write(
        tmp_path / "remote_results_4b/runs/E4_k1_s0/train_report.json",
        "{}",
    )
    with pytest.raises(ValueError, match="still present in live root"):
        build_outputs(tmp_path, strict=False)


def test_checked_classification_is_exhaustive_and_matches_baseline_hashes():
    classification = json.loads(
        (ROOT / "artifacts/ARTIFACT_CLASSIFICATION.json").read_text(encoding="utf-8")
    )
    invalidation = json.loads(
        (ROOT / "artifacts/HISTORICAL_EVAL_INVALIDATION.json").read_text(
            encoding="utf-8"
        )
    )
    summary = classification["summary"]
    assert summary == {
        "by_artifact_kind": {
            "adapter_checkpoint": 31,
            "latent_eval_derived_only": 50,
            "text_eval_preview_only": 8,
            "train_report": 31,
        },
        "by_finite_status": {"FINITE": 88, "NONFINITE": 32},
        "by_retention_class": {
            "CORRUPT_NONFINITE": 16,
            "HISTORICAL_UNBOUND": 12,
            "IRRECOVERABLE_LEGACY_SCORER": 50,
            "QUARANTINED": 16,
            "UNPROVEN_RUNTIME": 26,
        },
        "independently_rescorable_model_evals": 0,
        "legacy_latent_eval_count": 50,
        "legacy_selection_checkpoint_count": 13,
        "live_4b_target_runs_remaining": 0,
        "quarantine_move_count": 8,
        "text_preview_only_count": 8,
        "textual_4b_preview_only_count": 6,
        "total_artifacts": 120,
        "valid_current_model_results": 0,
    }
    assert classification["paid_spend_authorized"] is False
    assert classification["current_model_evidence_status"] == "VALID_EXPERIMENT_PENDING"
    assert set(classification["allowed_classes"]) == ALLOWED_CLASSES
    assert all(
        row["retention_class"] != "VALID_CURRENT"
        for row in classification["artifacts"]
    )
    assert len(invalidation["legacy_latent_eval"]["artifacts"]) == 50
    assert len(invalidation["checkpoint_selection"]["artifacts"]) == 13
    assert len(invalidation["quarantine_moves"]) == 8
    for move in invalidation["quarantine_moves"]:
        assert move["source_present_after_move"] is False
        assert move["source_tree_sha256_before_move"] == move["destination_tree_sha256"]
        assert move["destination_tree_sha256"] == move["rejected_counterpart_tree_sha256"]

    audit = json.loads((ROOT / "artifacts/audit_before.json").read_text(encoding="utf-8"))
    baseline = {
        row["id"]: row["sha256"]
        for row in audit["artifact_inventory"]["checkpoint_and_eval_files"]
        if row["id"].startswith("2b/results/ev_")
        or (
            row["id"].startswith("2b/runs/")
            and row["id"].endswith("/best_params.pt")
        )
    }
    current = {
        row["artifact_id"]: row["content_sha256"]
        for row in classification["artifacts"]
        if row["artifact_id"] in baseline
    }
    assert len(baseline) == 63
    assert current == baseline


def test_docs_make_invalidation_and_no_spend_status_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    roadmap = (
        ROOT / "docs/research-context-compiler/LATENT_ROADMAP.md"
    ).read_text(encoding="utf-8")
    no_spend = (ROOT / "docs/NO_SPEND_GATE.md").read_text(encoding="utf-8")

    for text in (readme, roadmap, no_spend):
        assert "PAID_SPEND_NOT_AUTHORIZED" in text
        assert "artifacts/milestone_r1_verdict.json" in text
    assert "VALID_EXPERIMENT_PENDING" in readme
    assert "IRRECOVERABLE_LEGACY_SCORER" in readme
    assert "97/112" in readme and "84/112" in readme
    assert "preview-only" in readme
    assert "MODEL_MEASURED @ Qwen3.5-2B" not in readme
    assert "REJECTED_BY_EVIDENCE (capability limit)" not in readme
    assert "INVALIDATED HISTORICAL TEXT" in roadmap
    assert "above-chance learning replicates across seeds" in roadmap
    assert "The historical paragraph below is preserved" in roadmap
