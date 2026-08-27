#!/usr/bin/env python3
"""Classify retained RCC model artifacts without treating derived metrics as evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
QUARANTINE_BATCH = "live_nan_duplicates_20260827"
QUARANTINED_RUNS = (
    "E4_k1_s0",
    "E4_k4_s0",
    "E4_k4_s1",
    "E4_k4_s2",
    "E4_k8_s0",
    "F4_k0_s0",
    "F4_k0_s1",
    "F4_k0_s2",
)
ALLOWED_CLASSES = {
    "VALID_CURRENT",
    "HISTORICAL_UNBOUND",
    "IRRECOVERABLE_LEGACY_SCORER",
    "CORRUPT_NONFINITE",
    "REJECTED_DUPLICATE",
    "UNPROVEN_RUNTIME",
    "QUARANTINED",
}
LEGACY_SCORER = "legacy_candidate_index_zero_scorer.unversioned"
LEGACY_SELECTION = "selection_provenance_invalid"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tree_receipt(path: Path) -> dict[str, Any]:
    files = []
    for artifact in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": artifact.relative_to(path).as_posix(),
                "sha256": _sha256(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        )
    return {
        "tree_sha256": hashlib.sha256(_canonical_json(files).encode("utf-8")).hexdigest(),
        "files": files,
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, *, strict: bool) -> Any:
    kwargs = {"parse_constant": _reject_constant} if strict else {}
    return json.loads(path.read_text(encoding="utf-8"), **kwargs)


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(key) and _all_finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _json_status(path: Path) -> tuple[str, dict[str, Any] | None]:
    try:
        value = _load_json(path, strict=True)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        try:
            permissive = _load_json(path, strict=False)
        except Exception:
            return "INVALID_JSON", None
        return "NONFINITE" if not _all_finite(permissive) else "INVALID_JSON", permissive
    if not isinstance(value, dict):
        return "FINITE", None
    return ("FINITE" if _all_finite(value) else "NONFINITE"), value


def _mtime(path: Path) -> str:
    moment = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artifact_id(evidence_root: Path, path: Path) -> str:
    relative = path.relative_to(evidence_root)
    parts = relative.parts
    if parts[0] == "remote_results":
        return "2b/" + Path(*parts[1:]).as_posix()
    if parts[0] == "remote_results_4b":
        return "4b/" + Path(*parts[1:]).as_posix()
    return "smoke/" + relative.as_posix()


def _run_id(path: Path, data: dict[str, Any] | None) -> str | None:
    if path.name in {"best_params.pt", "train_report.json"}:
        return path.parent.name
    if data and isinstance(data.get("adapter"), str):
        return Path(data["adapter"]).name
    return None


def _paired_report(path: Path) -> tuple[str, dict[str, Any] | None]:
    report = path.parent / "train_report.json"
    if not report.is_file():
        return "MISSING", None
    return _json_status(report)


def _discover(evidence_root: Path) -> list[Path]:
    patterns = (
        "remote_results/results/ev_*.json",
        "remote_results/results/text_*.json",
        "remote_results/runs/*/train_report.json",
        "remote_results/runs/*/best_params.pt",
        "remote_results_4b/results/text4b_*.json",
        "remote_results_4b/_rejected_nan_batch/*/train_report.json",
        "remote_results_4b/_rejected_nan_batch/*/best_params.pt",
        "remote_results_4b/quarantine/*/*/train_report.json",
        "remote_results_4b/quarantine/*/*/best_params.pt",
        "smoke_*/train_report.json",
        "smoke_*/best_params.pt",
    )
    return sorted({path for pattern in patterns for path in evidence_root.glob(pattern)})


def _tensor_finite_status(path: Path, *, required: bool) -> str:
    try:
        import torch
    except ImportError:
        if required:
            raise RuntimeError("strict checkpoint classification requires torch") from None
        return "BINARY_TENSOR_VALUES_NOT_LOADED"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if required:
            raise ValueError(f"safe checkpoint load failed: {path}: {type(exc).__name__}") from exc
        return "INVALID_CHECKPOINT"

    tensors = []

    def collect(value: Any) -> None:
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(payload)
    if not tensors:
        return "NO_TENSORS"
    return "FINITE" if all(bool(torch.isfinite(tensor).all()) for tensor in tensors) else "NONFINITE"


def _base_record(
    evidence_root: Path,
    path: Path,
    *,
    inspect_tensors: bool,
) -> dict[str, Any]:
    json_status: str
    data: dict[str, Any] | None
    if path.suffix == ".json":
        json_status, data = _json_status(path)
        finite_status = json_status
        associated_status = None
    else:
        json_status, data = _paired_report(path)
        finite_status = _tensor_finite_status(path, required=inspect_tensors)
        associated_status = json_status
    model = data.get("model") if data else None
    revision = data.get("revision") if data else None
    return {
        "artifact_id": _artifact_id(evidence_root, path),
        "source_path": ".rcc_work/" + path.relative_to(evidence_root).as_posix(),
        "content_sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "date_utc": _mtime(path),
        "date_provenance": "filesystem_mtime_utc",
        "model": model or "UNKNOWN_UNBOUND",
        "model_revision": revision or "UNKNOWN_UNBOUND",
        "run_id": _run_id(path, data),
        "finite_status": finite_status,
        "associated_run_finite_status": associated_status,
        "independently_rescorable": False,
    }


def _classify(
    evidence_root: Path,
    path: Path,
    *,
    inspect_tensors: bool,
) -> dict[str, Any]:
    record = _base_record(evidence_root, path, inspect_tensors=inspect_tensors)
    artifact_id = record["artifact_id"]
    name = path.name

    if artifact_id.startswith("2b/results/ev_"):
        record.update(
            {
                "artifact_kind": "latent_eval_derived_only",
                "schema": "legacy_unversioned.latent_eval.derived_only",
                "scorer_provenance": LEGACY_SCORER,
                "checkpoint_selection_provenance": LEGACY_SELECTION,
                "rescorable_status": "NOT_RESCORABLE_MISSING_RAW_CANDIDATE_SCORES",
                "retention_class": "IRRECOVERABLE_LEGACY_SCORER",
                "reason": (
                    "Derived-only records omit candidates in scored order, gold answer/index, "
                    "and raw per-token candidate log probabilities; the legacy scorer ranked "
                    "candidate index 0 instead of the serialized gold answer."
                ),
            }
        )
    elif artifact_id.startswith(("2b/results/text_", "4b/results/text4b_")):
        record.update(
            {
                "artifact_kind": "text_eval_preview_only",
                "schema": "legacy_unversioned.text_eval.preview_only",
                "scorer_provenance": "legacy_text_answer_parser.unversioned",
                "checkpoint_selection_provenance": "NOT_APPLICABLE_TEXT_BASELINE",
                "rescorable_status": "NOT_RESCORABLE_GENERATED_PREVIEW_ONLY",
                "retention_class": "HISTORICAL_UNBOUND",
                "reason": (
                    "Only generated_preview and derived correctness are retained; full model "
                    "outputs and a canonical raw scoring record are absent."
                ),
            }
        )
    elif artifact_id.startswith("2b/runs/"):
        record.update(
            {
                "artifact_kind": "adapter_checkpoint" if name == "best_params.pt" else "train_report",
                "schema": "torch_checkpoint.unversioned" if name == "best_params.pt" else "legacy_unversioned.train_report",
                "scorer_provenance": LEGACY_SCORER,
                "checkpoint_selection_provenance": LEGACY_SELECTION,
                "rescorable_status": "NOT_APPLICABLE_ARTIFACT_IS_NOT_RAW_EVAL",
                "retention_class": "UNPROVEN_RUNTIME",
                "reason": (
                    "Historical runtime semantics are not R1-proven and best-checkpoint "
                    "selection used derived validation values from the invalid legacy scorer."
                ),
            }
        )
    elif artifact_id.startswith("4b/_rejected_nan_batch/"):
        record.update(
            {
                "artifact_kind": "adapter_checkpoint" if name == "best_params.pt" else "train_report",
                "schema": "torch_checkpoint.unversioned" if name == "best_params.pt" else "legacy_unversioned.train_report.nonfinite",
                "scorer_provenance": LEGACY_SCORER,
                "checkpoint_selection_provenance": "INVALID_NONFINITE_TRAIN_REPORT",
                "rescorable_status": "NOT_RESCORABLE_CORRUPT_RUN",
                "retention_class": "CORRUPT_NONFINITE",
                "reason": (
                    "The associated 4B train report contains non-finite JSON and the entire "
                    "run bundle is rejected fail-closed."
                ),
            }
        )
    elif artifact_id.startswith("4b/quarantine/"):
        counterpart = "4b/_rejected_nan_batch/" + "/".join(path.parts[-2:])
        record.update(
            {
                "artifact_kind": "adapter_checkpoint" if name == "best_params.pt" else "train_report",
                "schema": "torch_checkpoint.unversioned" if name == "best_params.pt" else "legacy_unversioned.train_report.nonfinite",
                "scorer_provenance": LEGACY_SCORER,
                "checkpoint_selection_provenance": "INVALID_NONFINITE_TRAIN_REPORT",
                "rescorable_status": "NOT_RESCORABLE_REJECTED_DUPLICATE",
                "retention_class": "QUARANTINED",
                "duplicate_disposition": "REJECTED_DUPLICATE",
                "duplicate_of": counterpart,
                "reason": (
                    "Recoverably quarantined exact duplicate of the rejected non-finite 4B "
                    "run; excluded from every live candidate root."
                ),
            }
        )
    elif artifact_id.startswith("smoke/"):
        record.update(
            {
                "artifact_kind": "adapter_checkpoint" if name == "best_params.pt" else "train_report",
                "schema": "torch_checkpoint.unversioned" if name == "best_params.pt" else "legacy_unversioned.train_report",
                "scorer_provenance": LEGACY_SCORER,
                "checkpoint_selection_provenance": "UNBOUND_SMOKE_SELECTION",
                "rescorable_status": "NOT_APPLICABLE_UNBOUND_SMOKE",
                "retention_class": "HISTORICAL_UNBOUND",
                "reason": (
                    "Local smoke artifact is not bound to a canonical R1 recipe, scorer, "
                    "selection record, or independently rescorable evaluation."
                ),
            }
        )
    else:  # pragma: no cover - discovery and classifier evolve together
        raise ValueError(f"unclassified artifact: {path}")

    if record["retention_class"] not in ALLOWED_CLASSES:
        raise AssertionError(f"invalid retention class: {record['retention_class']}")
    return record


def _quarantine_receipts(evidence_root: Path, *, strict: bool) -> list[dict[str, Any]]:
    live_root = evidence_root / "remote_results_4b" / "runs"
    rejected_root = evidence_root / "remote_results_4b" / "_rejected_nan_batch"
    quarantine_root = (
        evidence_root / "remote_results_4b" / "quarantine" / QUARANTINE_BATCH
    )
    receipts = []
    for run_id in QUARANTINED_RUNS:
        source = live_root / run_id
        destination = quarantine_root / run_id
        counterpart = rejected_root / run_id
        present = (source.exists(), destination.exists(), counterpart.exists())
        if not strict and not any(present):
            continue
        if source.exists():
            raise ValueError(f"quarantined run still present in live root: {source}")
        if not destination.is_dir() or not counterpart.is_dir():
            raise ValueError(f"incomplete quarantine receipt for {run_id}")
        destination_receipt = _tree_receipt(destination)
        counterpart_receipt = _tree_receipt(counterpart)
        if destination_receipt != counterpart_receipt:
            raise ValueError(f"quarantine content differs from rejected counterpart: {run_id}")
        receipts.append(
            {
                "run_id": run_id,
                "source": f"4b/runs/{run_id}",
                "destination": f"4b/quarantine/{QUARANTINE_BATCH}/{run_id}",
                "rejected_counterpart": f"4b/_rejected_nan_batch/{run_id}",
                "source_present_after_move": False,
                "source_tree_sha256_before_move": destination_receipt["tree_sha256"],
                "destination_tree_sha256": destination_receipt["tree_sha256"],
                "rejected_counterpart_tree_sha256": counterpart_receipt["tree_sha256"],
                "files": destination_receipt["files"],
                "verification": "EXACT_FILE_SET_SIZE_AND_SHA256_MATCH",
                "retention_class": "QUARANTINED",
                "duplicate_disposition": "REJECTED_DUPLICATE",
                "reason": "Live candidate duplicate of a rejected non-finite 4B run.",
                "recovery_path": f"4b/runs/{run_id}",
            }
        )
    return receipts


def _expect_counts(summary: dict[str, Any]) -> None:
    expected = {
        "total_artifacts": 120,
        "legacy_latent_eval_count": 50,
        "legacy_selection_checkpoint_count": 13,
        "text_preview_only_count": 8,
        "textual_4b_preview_only_count": 6,
        "quarantine_move_count": 8,
        "live_4b_target_runs_remaining": 0,
        "valid_current_model_results": 0,
        "independently_rescorable_model_evals": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"classification count mismatch: {mismatches}")


def build_outputs(
    evidence_root: Path,
    *,
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_root = evidence_root.resolve()
    records = [
        _classify(evidence_root, path, inspect_tensors=strict)
        for path in _discover(evidence_root)
    ]
    receipts = _quarantine_receipts(evidence_root, strict=strict)
    classes = Counter(record["retention_class"] for record in records)
    kinds = Counter(record["artifact_kind"] for record in records)
    finite_statuses = Counter(record["finite_status"] for record in records)
    live_remaining = sum(
        int((evidence_root / "remote_results_4b" / "runs" / run_id).exists())
        for run_id in QUARANTINED_RUNS
    )
    summary = {
        "total_artifacts": len(records),
        "by_retention_class": dict(sorted(classes.items())),
        "by_artifact_kind": dict(sorted(kinds.items())),
        "by_finite_status": dict(sorted(finite_statuses.items())),
        "legacy_latent_eval_count": sum(
            record["retention_class"] == "IRRECOVERABLE_LEGACY_SCORER"
            for record in records
        ),
        "legacy_selection_checkpoint_count": sum(
            record["artifact_kind"] == "adapter_checkpoint"
            and record["artifact_id"].startswith("2b/runs/")
            and record["checkpoint_selection_provenance"] == LEGACY_SELECTION
            for record in records
        ),
        "text_preview_only_count": sum(
            record["artifact_kind"] == "text_eval_preview_only" for record in records
        ),
        "textual_4b_preview_only_count": sum(
            record["artifact_id"].startswith("4b/results/text4b_") for record in records
        ),
        "quarantine_move_count": len(receipts),
        "live_4b_target_runs_remaining": live_remaining,
        "valid_current_model_results": sum(
            record["retention_class"] == "VALID_CURRENT" for record in records
        ),
        "independently_rescorable_model_evals": sum(
            record["independently_rescorable"] for record in records
        ),
    }
    if strict:
        _expect_counts(summary)
    classification = {
        "schema_version": "r1_artifact_classification.v1",
        "classification_date": "2026-08-27",
        "source_root": ".rcc_work",
        "paid_spend_authorized": False,
        "current_model_evidence_status": "VALID_EXPERIMENT_PENDING",
        "allowed_classes": sorted(ALLOWED_CLASSES),
        "summary": summary,
        "artifacts": records,
    }

    legacy_eval = [
        {"artifact_id": row["artifact_id"], "content_sha256": row["content_sha256"]}
        for row in records
        if row["retention_class"] == "IRRECOVERABLE_LEGACY_SCORER"
    ]
    selection_checkpoints = [
        {"artifact_id": row["artifact_id"], "content_sha256": row["content_sha256"]}
        for row in records
        if row["artifact_kind"] == "adapter_checkpoint"
        and row["artifact_id"].startswith("2b/runs/")
    ]
    text4b = [
        {
            "artifact_id": row["artifact_id"],
            "content_sha256": row["content_sha256"],
            "rescorable_status": row["rescorable_status"],
        }
        for row in records
        if row["artifact_id"].startswith("4b/results/text4b_")
    ]
    invalidation = {
        "schema_version": "r1_historical_eval_invalidation.v1",
        "effective_date": "2026-08-27",
        "status": "INVALIDATION_ACTIVE",
        "paid_spend_authorized": False,
        "legacy_latent_eval": {
            "count": len(legacy_eval),
            "classification": "IRRECOVERABLE_LEGACY_SCORER",
            "independently_rescorable": False,
            "forbidden_current_labels": [
                "accuracy",
                "above_chance",
                "K_dose_response",
                "MODEL_MEASURED",
            ],
            "reason": (
                "Raw per-candidate token scores and actual gold mapping were not retained; "
                "the candidate-index-zero scorer defect cannot be repaired offline."
            ),
            "artifacts": legacy_eval,
        },
        "checkpoint_selection": {
            "count": len(selection_checkpoints),
            "provenance": LEGACY_SELECTION,
            "reason": "Best-step selection used validation histories from the invalid legacy scorer.",
            "artifacts": selection_checkpoints,
        },
        "textual_4b_preview_only": {
            "count": len(text4b),
            "classification": "HISTORICAL_UNBOUND",
            "independently_rescorable": False,
            "historical_observation_not_current_metric": {
                "direct_no_thinking_test_id": "97/112 derived correct flags",
                "direct_no_thinking_test_ood": "84/112 derived correct flags",
                "native_thinking_test_id": "1/112 derived correct flags",
                "native_thinking_test_ood": "0/112 derived correct flags",
            },
            "reason": "Full generations are absent; generated_preview may be truncated.",
            "artifacts": text4b,
        },
        "quarantine_moves": receipts,
        "recovery_policy": (
            "History is preserved. To restore a moved directory, first prove its live target "
            "is absent and then move the recorded destination to recovery_path; restoration "
            "does not make the run eligible evidence."
        ),
    }
    return classification, invalidation


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=ROOT / ".rcc_work")
    parser.add_argument(
        "--classification-out",
        type=Path,
        default=ROOT / "artifacts" / "ARTIFACT_CLASSIFICATION.json",
    )
    parser.add_argument(
        "--invalidation-out",
        type=Path,
        default=ROOT / "artifacts" / "HISTORICAL_EVAL_INVALIDATION.json",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    classification, invalidation = build_outputs(
        args.evidence_root,
        strict=not args.allow_partial,
    )
    _write(args.classification_out, classification)
    _write(args.invalidation_out, invalidation)
    print(
        json.dumps(
            {
                "classification": str(args.classification_out),
                "invalidation": str(args.invalidation_out),
                "summary": classification["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
