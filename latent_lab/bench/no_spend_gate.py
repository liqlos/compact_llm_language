"""Bounded no-spend integrity gate (2026-08-24 variant, fail-closed v2).

Inventories every retained 2B artifact and every live/rejected 4B artifact,
validates all offline-checkable evidence, and emits ONE canonical
machine-readable verdict plus a Markdown report:

    READY     only if every mandatory prerequisite is actually proven
    NOT_READY otherwise, with exact blocker codes and the smallest next
              executable action — never a weakened gate.

Fail-closed contract (v2 repairs of the READY fail-open audit):

1. Scoring consumes EXACTLY ONE lossless raw representation per record
   (full unique candidate permutation or fully finite aligned scores).
   Top-score ties, duplicated candidates/examples, partial or conflicting
   rankings, unknown examples and non-finite scores are explicit invalid
   STATUSES — never uncaught exceptions, never silently scored. Flags and
   statuses propagate through aggregation; strict JSON rejects NaN/Infinity
   literals on every retained input.
2. Report/selector schema requires nonempty strings, exact nonnegative
   integer steps (bool/float rejected), finite metrics and well-formed
   histories; an invalid history fails its artifact — records are never
   skipped-and-selected around.
3. Readiness is a PER-ARTIFACT RELATIONAL JOIN, not global counts: every
   retained loadable 2B bundle must be exactly identity-bound (model +
   40-hex revision, digest recorded) with ACTUAL fp32 trainables inside the
   payload, map one-to-one to its sidecar report and run directory, and own
   valid current-suite corrected raw evaluations covering BOTH mandatory
   splits (test_id AND test_ood).
4. Checkpoints, reports and eval files are discovered symmetrically by
   content; orphans, unreadable files and byte-duplicate bindings are
   blockers. Quarantine must be nonempty COMPLETE rejected evidence
   (marker + report/checkpoint pair) with ZERO known-invalid or
   byte-duplicate live 4B artifacts — one differing file no longer masks
   duplication.
5. Output/input overlap is rejected before anything is written; a bounded
   streaming source fingerprint is taken before and after the scan and any
   mutation aborts with an execution error. Canonical strict JSON contains
   no wall-clock value and no NaN. Exit 0 only on proven READY, 1 on
   evidence-backed NOT_READY, 2 on execution failure.

Usage (from the repository root):

    python -m latent_lab.bench.no_spend_gate \\
        --results-2b .rcc_work/remote_results \\
        --results-4b .rcc_work/remote_results_4b \\
        --out .rcc_work/no_spend_gate_20260824

Exit semantics:
    0  READY            every prerequisite PROVEN (not produced by this
                        evidence set so far)
    1  NOT_READY        evidence-backed negative verdict; see
                        gate_verdict.json blockers
    2  EXECUTION ERROR  bad invocation, output/input overlap, mutated or
                        unreadable sources, crash — no verdict may be
                        inferred from exit code 2

Hardware-free dry run (hashes/metadata only, no tensor payload loading):

    python -m latent_lab.bench.no_spend_gate --dry-run

Determinism: canonical outputs (artifact_inventory.json,
artifact_verdicts.json, gate_verdict.json, GATE_REPORT.md) are byte-stable
across reruns against unchanged inputs; they contain NO wall-clock value.
The only timestamp lives in the optional separate
telemetry_timestamp.json.

Trust boundary: this tool READ-ONLY scans its inputs. Historical .pt files
are repository-owned artifacts inside an isolated working copy; they are
loaded ONLY through torch.load(weights_only=True) and the project loader in
latent_lab.train.checkpointing, never through arbitrary unpickling, and
never modified, moved or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .corrected_scoring import (
    INVALID_AMBIGUOUS_TOP_TIE,
    INVALID_CONFLICTING_REPRESENTATIONS,
    INVALID_DUPLICATE_CANDIDATES,
    INVALID_DUPLICATE_EXAMPLE_RECORDS,
    INVALID_MALFORMED_CANDIDATE_SCORES,
    INVALID_MALFORMED_RANKED_CANDIDATES,
    INVALID_NONFINITE_CANDIDATE_SCORES,
    INVALID_UNKNOWN_EXAMPLE,
    MISSING_RAW_PREDICTION,
    select_best_checkpoint,
)

GATE_SCHEMA_VERSION = 2
GATE_ID = "no-spend-integrity-gate"
GATE_VARIANT = "20260824"

VERDICT_READY = "READY"
VERDICT_NOT_READY = "NOT_READY"

PREREQ_INVENTORY = "inventory_hashed_complete"
PREREQ_REPORTS = "train_reports_schema_identity_and_pins"
PREREQ_CKPT_STRICT = "checkpoints_strict_loadable_identity_bound"
PREREQ_RESCORE = "retained_evals_rescored_with_corrected_scorer"
PREREQ_SELECTION = "checkpoint_selection_uses_corrected_metric"
PREREQ_RUNTIME = "runtime_integrity_regressions_pass"
PREREQ_QUARANTINE = "invalid_4b_quarantined_not_live"
PREREQ_JOIN = "retained_2b_relational_join_complete"

STATUS_PROVEN = "PROVEN"
STATUS_FAILED = "FAILED"
STATUS_UNPROVEN = "UNPROVEN"

CKPT_LOADABLE = "loadable"
CKPT_INVALID = "invalid"
CKPT_LEGACY_UNBOUND = "legacy-unbound"
CKPT_CORRUPT = "corrupt"
CKPT_UNPROVEN = "unproven"

EVAL_OK = "RESCORED_CORRECTED"
EVAL_NO_RECORDS = "NO_RECORDS"
EVAL_UNREADABLE = "unreadable"
EVAL_INVALID_ENVELOPE = "INVALID_ENVELOPE"

MANDATORY_EVAL_SPLITS = ("test_id", "test_ood")

_PINNED_REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_SUITE_SHA_RE = re.compile(r"\A[0-9a-f]{64}\Z")

REQUIRED_CONFIG_FIELDS = {
    "mode": str,
    "interval": list,
    "k": int,
    "seed": int,
    "steps": int,
}
REQUIRED_TOP_LEVEL_NUMERIC = ("best_val_acc", "best_step")

# Regression proofs the gate executes (hardware-free, CPU-only) unless
# skipped. These pin the runtime-integrity prerequisites.
PROOF_TEST_NODES = (
    "tests/test_latent_runtime_integrity.py::test_exact_save_load_roundtrip_and_persisted_metrics",
    "tests/test_latent_runtime_integrity.py::test_bundle_identity_mismatch_rejected",
    "tests/test_latent_runtime_integrity.py::test_lora_and_clock_trainables_are_fp32_over_bf16_backbone",
    "tests/test_latent_runtime_integrity.py::test_recurrence_clock_is_fp32_regardless_of_default_dtype",
    "tests/test_latent_runtime_integrity.py::test_nan_metric_and_nan_state_rejected",
    "tests/test_latent_runtime_integrity.py::test_cached_localized_full_equivalence_across_fp32_roundtrip",
    "tests/test_latent_run.py",
)

TRUST_BOUNDARY = (
    "Gate ran read-only over an isolated private COPY of historical "
    ".rcc_work evidence. .pt payloads were loaded only via "
    "torch.load(weights_only=True) and the project loader "
    "latent_lab.train.checkpointing.load_adapter_bundle; nothing was "
    "modified, moved, deleted, executed beyond deserialization, nor sent "
    "off-device. No cloud/GPU/paid resource was touched."
)

BLOCKER_ACTIONS = {
    "INVENTORY_UNREADABLE_FILES":
        "Restore or document the unreadable paths, then rerun the gate.",
    "REPORT_SCHEMA_MISSING_FIELDS":
        "Re-emit train_report.json for each listed run through the current "
        "driver schema (config pins + trainable_precision); do not hand-edit "
        "historical files.",
    "REPORT_TRAINABLE_PRECISION_MISSING":
        "Add explicit trainable dtype/precision fields to the training "
        "report schema and regenerate reports on the next supervised smoke.",
    "REPORT_TRAINABLE_PRECISION_NOT_FP32":
        "Regenerate the checkpoint + report so stored trainables are fp32; "
        "a bf16/fp16 trainable_precision claim is disqualifying.",
    "CKPT_LEGACY_UNBOUND_IDENTITY":
        "Rebuild identity-bound bundles via save_adapter_bundle ONLY after "
        "weight-provenance verification; otherwise retire the weights into "
        "the next capped paired-seed retrain decision.",
    "CKPT_CORRUPT":
        "Keep quarantined as negative evidence; exclude from any resume or "
        "rescore.",
    "CKPT_AMBIGUOUS_DUPLICATE_BINDING":
        "Byte-identical checkpoint payloads bind ambiguously to multiple "
        "runs; delete or rename all but one canonical copy and record the "
        "provenance decision before rerunning.",
    "ORPHAN_EVIDENCE":
        "Every checkpoint/report/eval file must belong to a discovered run "
        "(or the rejected-quarantine tree); attach the orphans to their run "
        "or move them into documented quarantine.",
    "NON_RESCORABLE_MISSING_RAW_PREDICTION":
        "Rerun evaluation with raw per-candidate score capture enabled "
        "(capped CUDA canary scope) or formally invalidate the latent "
        "conclusions; never relabel derived records as a rescore.",
    "EVAL_EVIDENCE_INVALID":
        "Re-emit the listed eval files through the current driver (strict "
        "JSON, exactly one lossless raw representation per record, unique "
        "finite top score); malformed evidence can never be rescored.",
    "EVAL_SUITE_HASH_MISMATCH":
        "Rerun evaluation against the current behavioral-v2 suite so the "
        "declared suite_sha256 matches the locally recomputed digest.",
    "EVAL_IDENTITY_MISMATCH":
        "Re-emit evals whose model/revision identity equals the joined "
        "checkpoint + report identity; cross-model evals are unusable.",
    "EVAL_UNBOUND_TO_RETAINED_RUN":
        "Every eval file must declare an adapter path resolving to a "
        "discovered retained 2B run; attach or remove the orphaned file.",
    "EVAL_COVERAGE_INCOMPLETE":
        "Each retained loadable 2B run needs valid raw evals covering BOTH "
        "mandatory splits (test_id AND test_ood); rerun the missing split.",
    "SELECTION_PROVENANCE_NOT_CORRECTED":
        "Apply select_best_checkpoint over corrected-metric histories "
        "produced by the next validated run; discard historical best_step "
        "claims.",
    "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS":
        "Fix persistence to cast trainables to fp32 on save and verify on "
        "the next CPU smoke before any GPU spend.",
    "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH":
        "After hashed backup + manifest promotion, remove EVERY "
        "byte-identical copy under the LIVE 4B runs/ tree (deletion is "
        "outside this gate's authority) and keep _rejected_nan_batch as "
        "sole evidence.",
    "LIVE_4B_KNOWN_INVALID_ARTIFACTS":
        "Purge or quarantine the listed live 4B artifacts (corrupt/"
        "unbound payloads, NaN/degenerate reports, noncanonical JSON); "
        "the live tree must contain zero known-invalid artifacts.",
    "QUARANTINE_MARKER_MISSING":
        "Restore/confirm REJECTED.md in the quarantine tree, then rerun.",
    "QUARANTINE_INCOMPLETE":
        "Quarantine must hold nonempty complete rejected evidence "
        "(marker + at least one report/checkpoint pair); a marker-only or "
        "empty tree proves nothing.",
    "PROOF_TESTS_FAILED":
        "Fix the failing regression(s) locally before any spend decision.",
    "PREREQS_UNPROVEN":
        "Execute the skipped/unproven prerequisite checks (proof tests, "
        "payload inspection) — READY requires every prerequisite PROVEN.",
}

NEXT_ACTION_GENERIC = "Review gate_verdict.json prerequisites and rerun."


class GateExecutionError(RuntimeError):
    """Execution-layer failure (bad invocation, mutated sources, I/O).

    No verdict may be inferred from this exception; the CLI maps it to
    exit code 2.
    """


# ---------------------------------------------------------------------------
# deterministic JSON helpers
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively map values onto strict-JSON-safe equivalents.

    Non-finite floats become tagged strings so emitted documents always
    parse with a STRICT parser (json.loads with parse_constant rejecting).
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "<non-finite:nan>"
        if math.isinf(obj):
            return "<non-finite:inf>" if obj > 0 else "<non-finite:-inf>"
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    return str(obj)


def canonical_json_bytes(obj) -> bytes:
    return (json.dumps(_json_safe(obj), indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def write_canonical(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(obj))


def _reject_json_constant(name: str):
    raise ValueError(f"non-strict JSON constant {name!r} rejected")


def strict_json_loads(text: str):
    """STRICT JSON: NaN/Infinity literals raise ValueError immediately."""
    return json.loads(text, parse_constant=_reject_json_constant)


def parse_json_leniently(text: str) -> tuple[object, bool]:
    """Return (data, noncanonical). Falls back to a constant-tolerant parse
    ONLY for negative/quarantine evidence analysis; retained artifacts must
    parse strictly."""
    try:
        return strict_json_loads(text), False
    except ValueError:
        def _const(name: str):
            return {"NaN": float("nan"), "Infinity": float("inf"),
                    "-Infinity": float("-inf")}.get(name, float("nan"))
        return json.loads(text, parse_constant=_const), True


# ---------------------------------------------------------------------------
# bounded streaming source fingerprint
# ---------------------------------------------------------------------------

_FINGERPRINT_DOMAIN = b"no-spend-gate-source-fingerprint-v1\n"


def fingerprint_roots(roots: list[tuple[str, Path]]) -> str:
    """Deterministic bounded-memory fingerprint of the input trees.

    Streams sha256 over sorted (root label, relative path, size, file
    digest) lines; independent of absolute locations, wall clock and
    iteration order. Any read failure raises GateExecutionError.
    """
    h = hashlib.sha256()
    h.update(_FINGERPRINT_DOMAIN)
    for label, root in sorted(roots, key=lambda p: p[0]):
        if not root.is_dir():
            h.update(f"root\t{label}\tmissing\n".encode())
            continue
        h.update(f"root\t{label}\n".encode())
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(root).as_posix()
            fh = hashlib.sha256()
            try:
                with open(p, "rb") as f:
                    size = 0
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        fh.update(chunk)
                        size += len(chunk)
            except OSError as e:
                raise GateExecutionError(
                    f"fingerprint failed on {label}/{rel}: {e}") from e
            h.update(f"{rel}\t{size}\t{fh.hexdigest()}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# discovery + inventory
# ---------------------------------------------------------------------------

def discover_run_dirs(root: Path) -> list[Path]:
    """Every directory containing train_report.json, sorted. No hand lists."""
    found = sorted(
        p.parent for p in root.rglob("train_report.json") if p.is_file()
    )
    return found


def _classify_file_kind(name: str) -> str:
    if name == "train_report.json":
        return "train_report"
    if name == "best_params.pt":
        return "adapter_checkpoint"
    if name.startswith("ev_") and name.endswith(".json"):
        return "eval_json"
    if name.endswith(".json"):
        return "metrics_json"
    if name.endswith(".log"):
        return "log"
    if name == "REJECTED.md":
        return "quarantine_note"
    return "other"


@dataclass
class ScannedFile:
    label: str
    rel: str
    abs: Path
    kind: str
    size: int
    sha256: str
    dev_ino: tuple[int, int]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_roots(results_2b: Path, results_4b: Path, *, errors: list[str]) -> list[ScannedFile]:
    scanned: list[ScannedFile] = []
    for label, root in (("2b", results_2b), ("4b", results_4b)):
        if not root.is_dir():
            errors.append(f"{label}: results root missing: {root}")
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            try:
                st = p.stat()
                digest = sha256_file(p)
            except OSError as e:
                errors.append(f"{label}: unreadable {p.relative_to(root)}: {e}")
                continue
            scanned.append(ScannedFile(
                label=label, rel=p.relative_to(root).as_posix(), abs=p,
                kind=_classify_file_kind(p.name), size=st.st_size,
                sha256=digest, dev_ino=(st.st_dev, st.st_ino)))
    return scanned


def build_inventory(scanned: list[ScannedFile]) -> dict:
    by_hash: dict[str, list[str]] = {}
    for f in scanned:
        by_hash.setdefault(f.sha256, []).append(f"{f.label}/{f.rel}")
    by_storage: dict[tuple[int, int], list[str]] = {}
    for f in scanned:
        by_storage.setdefault(f.dev_ino, []).append(f"{f.label}/{f.rel}")

    artifacts = []
    for f in sorted(scanned, key=lambda x: (x.label, x.rel)):
        fid = f"{f.label}/{f.rel}"
        same_hash = [p for p in by_hash[f.sha256] if p != fid]
        linked = sorted(p for p in by_storage[f.dev_ino] if p != fid)
        entry = {
            "id": fid,
            "kind": f.kind,
            "root": f.label,
            "path": f.rel,
            "size_bytes": f.size,
            "sha256": f.sha256,
        }
        if same_hash:
            entry["identical_content_with"] = sorted(same_hash)
        if len(by_hash[f.sha256]) > 1:
            entry["content_group_size"] = len(by_hash[f.sha256])
        if linked:
            entry["hardlinked_with"] = linked
        artifacts.append(entry)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": {"id": GATE_ID, "variant": GATE_VARIANT},
        "n_artifacts": len(artifacts),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# train-report validation
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    run_id: str
    root: str
    dir_rel: str
    report_rel: str | None
    checkpoint_rel: str | None
    report: dict | None = None
    report_sha256: str | None = None
    ckpt_sha256: str | None = None
    report_noncanonical: bool = False
    report_unreadable: str | None = None


def _is_pinned_revision(v) -> bool:
    return isinstance(v, str) and bool(_PINNED_REVISION_RE.fullmatch(v))


def _is_exact_nonneg_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def validate_history_entries(hist) -> list[str]:
    """Every history entry must be a well-formed {step, accuracy} record.

    An invalid entry fails the WHOLE artifact (never skip-and-select):
    steps must be exact nonnegative integers (bool/float rejected) and
    accuracies finite numbers.
    """
    problems: list[str] = []
    for i, entry in enumerate(hist):
        if not isinstance(entry, dict):
            problems.append(f"val_history[{i}]:not_object")
            continue
        if "step" not in entry:
            problems.append(f"val_history[{i}].step:missing")
        elif not _is_exact_nonneg_int(entry["step"]):
            problems.append(f"val_history[{i}].step:type")
        if "accuracy" not in entry:
            problems.append(f"val_history[{i}].accuracy:missing")
        elif not _is_finite_number(entry["accuracy"]):
            problems.append(f"val_history[{i}].accuracy:type")
    return problems


def validate_train_report(report: dict) -> dict:
    """Schema validation; every violation is recorded, never defaulted."""
    problems: list[str] = []
    config = report.get("config")
    if not isinstance(config, dict):
        problems.append("config:missing")
        config = {}
    for key, typ in REQUIRED_CONFIG_FIELDS.items():
        if key not in config:
            problems.append(f"config.{key}:missing")
        elif typ is int:
            if not _is_exact_nonneg_int(config[key]):
                problems.append(f"config.{key}:type")
            elif key == "steps" and config[key] < 0:
                problems.append(f"config.{key}:negative")
        elif typ is list:
            iv = config[key]
            if not (isinstance(iv, list) and len(iv) == 2
                    and all(_is_exact_nonneg_int(x) or (
                        isinstance(x, int) and not isinstance(x, bool))
                        for x in iv)):
                problems.append(f"config.{key}:type")
        elif typ is str:
            if not isinstance(config[key], str) or not config[key].strip():
                problems.append(f"config.{key}:empty_or_type")
        elif not isinstance(config[key], typ):
            problems.append(f"config.{key}:type")
    for key in REQUIRED_TOP_LEVEL_NUMERIC:
        if key not in report:
            problems.append(f"{key}:missing")
        elif key == "best_step":
            if not _is_exact_nonneg_int(report[key]):
                problems.append(f"{key}:type")
        elif not _is_finite_number(report[key]):
            problems.append(f"{key}:type")
    if "model" not in report or not str(report.get("model", "")).strip():
        problems.append("model:missing_or_empty")
    rev = report.get("revision")
    if not _is_pinned_revision(rev):
        problems.append("revision:not_pinned_40hex")
    suite = report.get("suite_sha256")
    if not isinstance(suite, str) or not _SUITE_SHA_RE.fullmatch(suite):
        problems.append("suite_sha256:not_64hex")

    hist = report.get("val_history")
    if not isinstance(hist, list) or not hist:
        problems.append("val_history:missing_or_empty")
        hist_problems: list[str] = []
    else:
        hist_problems = validate_history_entries(hist)
        problems.extend(hist_problems)

    precision = report.get("trainable_precision")
    if "trainable_precision" not in report:
        problems.append("trainable_precision:missing")
    elif not isinstance(precision, str) or not precision.strip():
        problems.append("trainable_precision:empty_or_type")
    elif precision.strip().lower() != "fp32":
        problems.append("trainable_precision:not_fp32")

    selection = None
    if isinstance(hist, list) and hist and not hist_problems:
        sel = select_best_checkpoint(hist)
        reported_acc = report.get("best_val_acc")
        reported_step = report.get("best_step")
        if sel is not None and _is_finite_number(reported_acc) \
                and _is_exact_nonneg_int(reported_step):
            selection = {
                "recomputed_best_step": sel.step,
                "recomputed_best_metric": sel.metric,
                "reported_best_step": reported_step,
                "reported_best_metric": float(reported_acc),
                "consistent_with_reported": bool(
                    sel.step == reported_step
                    and math.isclose(float(reported_acc), sel.metric,
                                     rel_tol=0, abs_tol=1e-9)),
            }
        elif sel is None:
            selection = {"recomputed_best_step": None,
                         "consistent_with_reported": False}
    elif hist_problems:
        selection = {"consistent_with_reported": False,
                     "reason": "invalid_history_entries"}
    return {
        "problems": problems,
        "scorer_tag": config.get("scorer"),
        "identity": {
            "model_id": report.get("model") if isinstance(
                report.get("model"), str) else None,
            "revision": rev if _is_pinned_revision(rev) else None,
            "suite_sha256": suite if isinstance(suite, str) else None,
        },
        "selection_check": selection,
    }


# ---------------------------------------------------------------------------
# checkpoint classification
# ---------------------------------------------------------------------------

def _load_raw_payload(path: Path):
    """Safe deserialize: weights_only=True, project loader afterwards."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def classify_checkpoint(path: Path, *, report_identity: dict | None,
                        dry_run: bool) -> dict:
    """Classify one .pt payload. Read-only; never mutates anything."""
    info: dict = {"classification": CKPT_UNPROVEN, "reasons": []}
    if dry_run:
        info["reasons"].append("dry_run_skips_payload_loading")
        return info
    try:
        raw = _load_raw_payload(path)
    except Exception as e:  # noqa: BLE001 — any decode failure is corruption
        info["classification"] = CKPT_CORRUPT
        info["reasons"].append(f"decode_failed:{type(e).__name__}")
        return info
    if not isinstance(raw, dict):
        info["classification"] = CKPT_CORRUPT
        info["reasons"].append("payload_not_dict")
        return info

    if all(k in raw for k in ("format_version", "kind", "model_id",
                              "revision", "tensors")):
        return _classify_bundle(raw, path, report_identity, info)
    return _classify_plain_state(raw, info)


def _classify_bundle(raw: dict, path: Path, report_identity, info: dict) -> dict:
    from ..train.checkpointing import (
        AdapterBundleIdentityError,
        AdapterBundleSchemaError,
        CheckpointError,
        load_adapter_bundle,
    )

    bid, brev = raw.get("model_id"), raw.get("revision")
    info["bundle_identity"] = {"model_id": bid, "revision": brev}
    if not _is_pinned_revision(brev):
        info["classification"] = CKPT_INVALID
        info["reasons"].append("bundle_revision_not_pinned_40hex")
        return info
    if report_identity:
        rep_mid = report_identity.get("model_id")
        rep_rev = report_identity.get("revision")
        if (rep_mid and str(bid) != str(rep_mid)) or \
                (rep_rev and str(brev) != str(rep_rev)):
            info["classification"] = CKPT_INVALID
            info["reasons"].append("bundle_identity_conflicts_with_report")
            return info
    try:
        tensors = load_adapter_bundle(path, model_id=str(bid),
                                      revision=str(brev))
    except AdapterBundleIdentityError as e:
        info["classification"] = CKPT_INVALID
        info["reasons"].append(f"identity_error:{e}")
        return info
    except AdapterBundleSchemaError as e:
        info["classification"] = CKPT_INVALID
        info["reasons"].append(f"schema_error:{type(e).__name__}")
        return info
    except CheckpointError as e:
        info["classification"] = CKPT_CORRUPT
        info["reasons"].append(f"non_finite_or_state_error:{type(e).__name__}")
        return info
    except Exception as e:  # noqa: BLE001
        info["classification"] = CKPT_CORRUPT
        info["reasons"].append(f"load_failed:{type(e).__name__}")
        return info

    # Actual payload inspection: identity-bound bundles must STILL carry
    # real fp32 trainables — a bf16 payload behind correct metadata is
    # invalid, not loadable (fail-closed repair of the bf16 fail-open).
    import torch

    dtypes = sorted({str(t.dtype) for t in tensors.values()
                     if torch.is_tensor(t) and t.is_floating_point()})
    info["trainable_dtypes"] = dtypes
    non_fp32 = [d for d in dtypes if d != "torch.float32"]
    if non_fp32:
        info["classification"] = CKPT_INVALID
        info["reasons"].append(
            "non_fp32_trainables_stored:" + ",".join(non_fp32))
        info["reasons"].append("strict_project_loader_passed")
        if report_identity:
            info["reasons"].append("identity_matches_sidecar_report")
        return info

    info["classification"] = CKPT_LOADABLE
    info["reasons"].append("strict_project_loader_passed")
    info["reasons"].append("actual_payload_all_fp32")
    if report_identity:
        info["reasons"].append("identity_matches_sidecar_report")
    return info


def _classify_plain_state(raw: dict, info: dict) -> dict:
    import torch

    tensors = [v for v in raw.values()]
    if not tensors or not all(torch.is_tensor(t) for t in tensors):
        info["classification"] = CKPT_CORRUPT
        info["reasons"].append("plain_dict_with_non_tensor_entries")
        return info
    dtypes = sorted({str(t.dtype) for t in tensors})
    info["tensor_dtypes"] = dtypes
    info["n_tensors"] = len(tensors)
    finite = all(not t.is_floating_point()
                 or bool(torch.isfinite(t).all()) for t in tensors)
    info["all_finite"] = finite
    if not finite:
        info["classification"] = CKPT_CORRUPT
        info["reasons"].append("non_finite_tensor_values")
        return info
    info["classification"] = CKPT_LEGACY_UNBOUND
    info["reasons"].append(
        "plain state dict without in-bundle kind/format_version/"
        "model_id/revision; strict identity-bound load impossible")
    if any(d != "torch.float32" for d in dtypes):
        info["reasons"].append("non_fp32_trainables_stored:" +
                               ",".join(dtypes))
    return info


# ---------------------------------------------------------------------------
# eval rescoring eligibility
# ---------------------------------------------------------------------------

_EVAL_INVALID_STATUSES = frozenset({
    INVALID_UNKNOWN_EXAMPLE,
    INVALID_DUPLICATE_EXAMPLE_RECORDS,
    INVALID_CONFLICTING_REPRESENTATIONS,
    INVALID_MALFORMED_CANDIDATE_SCORES,
    INVALID_NONFINITE_CANDIDATE_SCORES,
    INVALID_AMBIGUOUS_TOP_TIE,
    INVALID_MALFORMED_RANKED_CANDIDATES,
    INVALID_DUPLICATE_CANDIDATES,
})


def is_invalid_eval_status(status: str | None) -> bool:
    return status in _EVAL_INVALID_STATUSES or status == EVAL_INVALID_ENVELOPE


def evaluate_eval_file(path: Path, examples_by_id: dict | None) -> dict:
    """Fail-closed evaluation of one eval JSON artifact.

    Strict JSON only (NaN/Infinity literals make the file unreadable);
    the envelope (adapter/split/model/revision/suite hash) must be sane;
    records must satisfy the single-lossless-representation rescoring
    contract. Any violation yields an explicit non-OK status — the file
    can then never satisfy the readiness join.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return {"file": path.name, "status": EVAL_UNREADABLE,
                "detail": f"os_error:{e}"}
    try:
        data = strict_json_loads(text)
    except ValueError as e:
        return {"file": path.name, "status": EVAL_UNREADABLE,
                "detail": f"strict_json:{e}"}
    if not isinstance(data, dict):
        return {"file": path.name, "status": EVAL_UNREADABLE,
                "detail": "top level not an object"}

    envelope_problems: list[str] = []
    adapter = data.get("adapter")
    if not isinstance(adapter, str) or not adapter.strip():
        envelope_problems.append("adapter:missing_or_empty")
    split = data.get("split")
    if not isinstance(split, str) or not split.strip():
        envelope_problems.append("split:missing_or_empty")
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        envelope_problems.append("model:missing_or_empty")
    revision = data.get("revision")
    if not _is_pinned_revision(revision):
        envelope_problems.append("revision:not_pinned_40hex")
    declared = data.get("suite_sha256")
    suite_ok = isinstance(declared, str) \
        and bool(_SUITE_SHA_RE.fullmatch(declared)) \
        and declared == current_suite_sha256()
    if not isinstance(declared, str) or not _SUITE_SHA_RE.fullmatch(declared):
        envelope_problems.append("suite_sha256:not_64hex")
    results = data.get("results")
    records: list[dict] = []
    if not isinstance(results, dict):
        envelope_problems.append("results:missing_or_type")
    else:
        for key in sorted(results):
            res = results[key]
            if not isinstance(res, dict) or \
                    not isinstance(res.get("records"), list):
                envelope_problems.append(f"results.{key}:malformed")
                continue
            for rec in res["records"]:
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    envelope_problems.append(f"results.{key}:record_not_object")

    out = {
        "file": path.name,
        "adapter": adapter if isinstance(adapter, str) else None,
        "split": split if isinstance(split, str) else None,
        "model": model if isinstance(model, str) else None,
        "revision": revision if isinstance(revision, str) else None,
        "suite_sha256_matches_current_suite": suite_ok,
        "n_records": len(records),
        "bound_run": None,
    }
    if envelope_problems:
        out["status"] = EVAL_INVALID_ENVELOPE
        out["detail"] = ";".join(sorted(set(envelope_problems)))
        return out

    if examples_by_id is None:
        examples_by_id = suite_examples_by_id()

    if not records:
        out["status"] = EVAL_NO_RECORDS
        return out
    from .corrected_scoring import rescore_records

    outcome = rescore_records(records, examples_by_id)
    out["status"] = outcome.status
    if outcome.status == EVAL_OK:
        out["corrected_accuracy"] = outcome.corrected_accuracy
        out["flags"] = list(outcome.flags)
    elif outcome.detail:
        out["detail"] = outcome.detail
    return out


_SUITE_CACHE: dict | None = None
_SUITE_SHA_CACHE: str | None = None


def suite_examples_by_id() -> dict:
    """ex_id -> Example, built once per process (stdlib-deterministic)."""
    global _SUITE_CACHE
    if _SUITE_CACHE is None:
        from ..bench.suite import build_suite

        cache: dict = {}
        suite = build_suite()
        for split in ("train", "validation", "test_id", "test_ood"):
            for ex in getattr(suite, split):
                cache[ex.ex_id] = ex
        _SUITE_CACHE = cache
    return _SUITE_CACHE


def current_suite_sha256() -> str:
    global _SUITE_SHA_CACHE
    if _SUITE_SHA_CACHE is None:
        from ..bench.suite import build_suite

        _SUITE_SHA_CACHE = build_suite().manifest()["sha256"]
    return _SUITE_SHA_CACHE


# ---------------------------------------------------------------------------
# 4B quarantine analysis
# ---------------------------------------------------------------------------

def _read_report_semantics(abs_path: Path, *, strict: bool) -> dict:
    """Extract (loss, best_acc, noncanonical, unreadable) from a report."""
    try:
        text = abs_path.read_text(encoding="utf-8")
    except OSError:
        return {"loss": None, "acc": None, "noncanonical": False,
                "unreadable": True}
    try:
        if strict:
            data = strict_json_loads(text)
            return {"loss": data.get("final_train_loss"),
                    "acc": data.get("best_val_acc"),
                    "noncanonical": False, "unreadable": False}
        data, noncanonical = parse_json_leniently(text)
        if not isinstance(data, dict):
            raise ValueError("top level not an object")
        return {"loss": data.get("final_train_loss"),
                "acc": data.get("best_val_acc"),
                "noncanonical": noncanonical, "unreadable": False}
    except Exception:  # noqa: BLE001 — unreadable/malformed evidence
        return {"loss": None, "acc": None, "noncanonical": False,
                "unreadable": True}


def _is_degenerate_report(loss, acc) -> bool:
    nan_loss = isinstance(loss, float) and not math.isfinite(loss)
    fake_best = isinstance(acc, (int, float)) \
        and not isinstance(acc, bool) \
        and math.isfinite(float(acc)) and float(acc) >= 1.0
    return bool(nan_loss or fake_best)


def analyze_quarantine(scanned: list[ScannedFile]) -> dict:
    def files_under(prefix: str) -> dict[str, ScannedFile]:
        return {
            f.rel[len(prefix):]: f for f in scanned
            if f.label == "4b" and f.rel.startswith(prefix)
        }

    live = files_under("runs/")
    rejected = files_under("_rejected_nan_batch/")
    marker = rejected.get("REJECTED.md")
    identical, differing, only_live, only_rej = [], [], [], []
    for rel in sorted(set(live) | set(rejected)):
        lf, rf = live.get(rel), rejected.get(rel)
        if lf and rf:
            if lf.sha256 == rf.sha256:
                identical.append(rel)
            else:
                differing.append(rel)
        elif lf:
            only_live.append(rel)
        else:
            only_rej.append(rel)
    # Fail-closed duplication rule: ANY byte-identical file shared between
    # the live tree and the rejected batch makes the live tree suspect —
    # a single differing file no longer masks the duplication.
    live_dupes_quarantine = bool(identical)

    nan_runs, fake_best_runs, rejected_unreadable = [], [], []
    rejected_report_rels = 0
    for rel, f in sorted(rejected.items()):
        if f.kind != "train_report":
            continue
        rejected_report_rels += 1
        sem = _read_report_semantics(f.abs, strict=False)
        if sem["unreadable"]:
            rejected_unreadable.append(rel)
            continue
        run = rel.split("/")[0]
        if isinstance(sem["loss"], float) and not math.isfinite(sem["loss"]):
            nan_runs.append(run)
        acc = sem["acc"]
        if isinstance(acc, (int, float)) and not isinstance(acc, bool) \
                and math.isfinite(float(acc)) and float(acc) >= 1.0:
            fake_best_runs.append(run)

    live_invalid_reports, live_noncanonical_reports = [], []
    for rel, f in sorted(live.items()):
        if f.kind != "train_report":
            continue
        sem = _read_report_semantics(f.abs, strict=True)
        if sem["unreadable"]:
            live_invalid_reports.append(rel)
            continue
        if sem["noncanonical"]:
            live_noncanonical_reports.append(rel)
        if _is_degenerate_report(sem["loss"], sem["acc"]):
            live_invalid_reports.append(rel)

    rejected_ckpt_count = sum(
        1 for f in rejected.values() if f.kind == "adapter_checkpoint")
    return {
        "live_run_dirs": sorted({r.split("/")[0] for r in live
                                 if r.endswith("/train_report.json")}),
        "rejected_run_dirs": sorted({r.split("/")[0] for r in rejected
                                     if r.endswith("/train_report.json")}),
        "quarantine_marker_present": marker is not None,
        "quarantine_nonempty_complete": bool(
            marker is not None and rejected_report_rels > 0
            and rejected_ckpt_count > 0),
        "quarantine_has_any_evidence": bool(rejected),
        "rejected_report_files": rejected_report_rels,
        "rejected_checkpoint_files": rejected_ckpt_count,
        "live_vs_rejected_identical_files": identical,
        "live_vs_rejected_differing_files": differing,
        "only_in_live": only_live,
        "only_in_rejected": only_rej,
        "live_tree_duplicates_quarantine": live_dupes_quarantine,
        "rejected_runs_with_nan_final_loss": sorted(set(nan_runs)),
        "rejected_runs_with_degenerate_best_acc_1": sorted(set(fake_best_runs)),
        "rejected_unreadable_reports": sorted(rejected_unreadable),
        "live_invalid_or_degenerate_reports": sorted(
            set(live_invalid_reports)),
        "live_noncanonical_reports": sorted(set(live_noncanonical_reports)),
    }


# ---------------------------------------------------------------------------
# proof tests
# ---------------------------------------------------------------------------

def run_proof_tests(repo_root: Path, log_path: Path | None = None) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header",
           "-p", "no:cacheprovider", *PROOF_TEST_NODES]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=3600)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
    # Canonical payload deliberately excludes wall-clock/durations so that
    # reruns stay byte-identical; the full log goes to proof_tests.log.
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "nodes": list(PROOF_TEST_NODES),
        "all_passed": proc.returncode == 0,
    }


# ---------------------------------------------------------------------------
# gate assembly
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    verdict: str
    inventory: dict
    artifact_verdicts: dict
    gate_verdict: dict
    report_md: str


@dataclass
class _JoinVerdict:
    report_valid: bool
    checkpoint_valid: bool
    eval_splits_covered: list[str] = field(default_factory=list)
    identity_mismatch_evals: list[str] = field(default_factory=list)
    complete: bool = False


def _bind_eval_to_run(adapter: str | None,
                      run_index: dict[str, str]) -> str | None:
    """Resolve an eval's ``adapter`` declaration onto a retained run.

    Accepts the run id, its directory-relative path, or a ``runs/<id>``
    style suffix — but never guesses: an adapter matching no retained run
    stays unbound and blocks readiness.
    """
    if not isinstance(adapter, str):
        return None
    cand = adapter.strip().removeprefix("./").rstrip("/")
    if cand in run_index:
        return run_index[cand]
    tail = cand.rsplit("/", 1)[-1]
    return run_index.get(tail)


def run_gate(results_2b: Path, results_4b: Path, *, repo_root: Path,
             dry_run: bool = False, skip_proof_tests: bool = False,
             proof_log_path: Path | None = None,
             proof_result: dict | None = None,
             corrected_scorer_tag: str = "corrected-gold-aware-v1") -> GateResult:
    """Execute every gate stage and return the assembled result.

    ``proof_result`` is a TEST-ONLY injection hook for the executed proof
    outcomes; the CLI never supplies it, so production runs either execute
    the regressions or record them skipped/unproven.

    Raises GateExecutionError when the input trees mutate mid-scan
    (before/after streaming fingerprints disagree).
    """
    roots = [("2b", results_2b), ("4b", results_4b)]
    fp_before = fingerprint_roots(roots)
    errors: list[str] = []
    scanned = scan_roots(results_2b, results_4b, errors=errors)
    inventory = build_inventory(scanned)

    # ---- runs (discovered, never hand-listed) ----------------------------
    runs: dict[str, list[RunInfo]] = {"2b": [], "4b": []}
    files_by_id = {f"{f.label}/{f.rel}": f for f in scanned}
    for label, root in (("2b", results_2b), ("4b", results_4b)):
        if not root.is_dir():
            continue
        for d in discover_run_dirs(root):
            rel = d.relative_to(root).as_posix()
            rep_rel, ckpt_rel = None, None
            for name in ("train_report.json", "best_params.pt"):
                cand = f"{label}/{rel}/{name}"
                if cand in files_by_id:
                    if name == "train_report.json":
                        rep_rel = cand
                    else:
                        ckpt_rel = cand
            ri = RunInfo(run_id=d.name, root=label, dir_rel=rel,
                         report_rel=rep_rel, checkpoint_rel=ckpt_rel)
            scope_rejected = rel.startswith("_rejected")
            if rep_rel:
                sf = files_by_id[rep_rel]
                try:
                    if scope_rejected:
                        data, noncanonical = parse_json_leniently(
                            sf.abs.read_text(encoding="utf-8"))
                        ri.report = data if isinstance(data, dict) else None
                        ri.report_noncanonical = noncanonical
                        if not isinstance(data, dict):
                            ri.report_unreadable = "top level not an object"
                    else:
                        ri.report = strict_json_loads(
                            sf.abs.read_text(encoding="utf-8"))
                except (ValueError, OSError) as e:
                    ri.report = None
                    ri.report_unreadable = str(e)
                ri.report_sha256 = sf.sha256
            if ckpt_rel:
                ri.ckpt_sha256 = files_by_id[ckpt_rel].sha256
            runs[label].append(ri)

    # ---- checkpoint duplicate groups (live side only) --------------------
    # Byte-identical payloads binding to different runs make the binding
    # ambiguous; quarantined rejected copies are exempt (sole evidence).
    def _is_rejected(f: ScannedFile) -> bool:
        return f.rel.startswith("_rejected")

    ckpt_paths = [f for f in scanned if f.kind == "adapter_checkpoint"
                  and not _is_rejected(f)]
    dup_members: set[str] = set()
    primary: dict[str, str] = {}
    by_hash: dict[str, list[str]] = {}
    for f in ckpt_paths:
        by_hash.setdefault(f.sha256, []).append(f"{f.label}/{f.rel}")
    for h, ids in sorted(by_hash.items()):
        if len(ids) > 1:
            ordered = sorted(ids)
            primary[h] = ordered[0]
            dup_members.update(ordered)

    # ---- per-run validation + classification -----------------------------
    run_verdicts = []

    def _scope_of(ri: RunInfo) -> str:
        return "rejected_negative" if ri.dir_rel.startswith("_rejected") \
            else "retained_candidate"

    for label in ("2b", "4b"):
        for ri in runs[label]:
            rv: dict = {"run_id": ri.run_id, "root": ri.root,
                        "dir": ri.dir_rel, "scope": _scope_of(ri)}
            identity = None
            if ri.report is not None:
                report_check = validate_train_report(ri.report)
                identity = report_check["identity"]
                rv["report_problems"] = report_check["problems"]
                rv["identity"] = identity
                rv["scorer_tag"] = report_check["scorer_tag"]
                rv["selection_check"] = report_check["selection_check"]
                declared = identity.get("suite_sha256")
                rv["suite_sha256_matches_current_suite"] = bool(
                    declared == current_suite_sha256())
            else:
                rv["report_problems"] = ["train_report:unreadable_strict_json"]
                if ri.report_noncanonical:
                    rv["report_problems"] = [
                        "train_report:noncanonical_strict_json"]
                rv["suite_sha256_matches_current_suite"] = None
            rv["report_unreadable"] = ri.report_unreadable
            rv["report_noncanonical"] = ri.report_noncanonical
            if ri.checkpoint_rel:
                cls = classify_checkpoint(
                    files_by_id[ri.checkpoint_rel].abs,
                    report_identity=identity, dry_run=dry_run)
                # checkpoint_rel is ALREADY label-qualified ("2b/<rel>");
                # do not re-prefix it when matching duplicate groups.
                if ri.checkpoint_rel in dup_members:
                    cls = dict(cls)
                    cls["duplicate_of"] = primary[
                        files_by_id[ri.checkpoint_rel].sha256]
                rv["checkpoint"] = cls
                rv["checkpoint_sha256"] = ri.ckpt_sha256
            run_verdicts.append(rv)

    # ---- eval rescoring ----------------------------------------------------
    eval_verdicts = []
    if not dry_run:
        for f in scanned:
            if f.kind == "eval_json":
                ev = evaluate_eval_file(f.abs, None)
                ev["root"] = f.label
                ev["path"] = f.rel
                ev["in_quarantine_tree"] = _is_rejected(f)
                eval_verdicts.append(ev)
    eval_verdicts.sort(key=lambda e: (e.get("path", ""), e["file"]))

    # ---- relational join: retained 2B report <-> checkpoint <-> evals -----
    retained_2b = [rv for rv in run_verdicts
                   if rv["root"] == "2b"
                   and rv["scope"] == "retained_candidate"]
    run_index: dict[str, str] = {}
    for rv in retained_2b:
        run_index.setdefault(rv["run_id"], rv["dir"])
        run_index.setdefault(rv["dir"], rv["dir"])
    live_evals = [e for e in eval_verdicts if not e.get("in_quarantine_tree")]
    for ev in live_evals:
        ev["bound_run"] = _bind_eval_to_run(ev.get("adapter"), run_index)

    joins: dict[str, _JoinVerdict] = {}
    for rv in retained_2b:
        ident = rv.get("identity") or {}
        ck = rv.get("checkpoint") or {}
        ckpt_valid = (
            ck.get("classification") == CKPT_LOADABLE
            and not any(str(r).startswith("non_fp32_trainables_stored")
                        for r in ck.get("reasons", []))
            and "duplicate_of" not in ck
        )
        jv = _JoinVerdict(
            report_valid=not rv.get("report_problems"),
            checkpoint_valid=ckpt_valid)
        for ev in live_evals:
            if ev.get("bound_run") != rv["dir"] or ev.get("root") != "2b":
                continue
            if ev.get("status") != EVAL_OK:
                continue
            if not ev.get("suite_sha256_matches_current_suite"):
                continue
            if ev.get("model") != ident.get("model_id") or \
                    ev.get("revision") != ident.get("revision"):
                jv.identity_mismatch_evals.append(ev["file"])
                continue
            if ev.get("split") in MANDATORY_EVAL_SPLITS and \
                    ev["split"] not in jv.eval_splits_covered:
                jv.eval_splits_covered.append(ev["split"])
        jv.eval_splits_covered.sort()
        jv.complete = bool(
            jv.report_valid and jv.checkpoint_valid
            and set(MANDATORY_EVAL_SPLITS).issubset(
                set(jv.eval_splits_covered)))
        joins[rv["dir"]] = jv
        rv["join"] = {
            "report_valid": jv.report_valid,
            "checkpoint_valid": jv.checkpoint_valid,
            "eval_splits_covered": jv.eval_splits_covered,
            "mandatory_splits_missing": sorted(
                set(MANDATORY_EVAL_SPLITS) - set(jv.eval_splits_covered)),
            "complete": jv.complete,
        }

    # ---- orphan discovery (symmetric, content-based) ----------------------
    expected_run_files: set[str] = set()
    for label, root in (("2b", results_2b), ("4b", results_4b)):
        if not root.is_dir():
            continue
        for d in discover_run_dirs(root):
            rel = d.relative_to(root).as_posix()
            for name in ("train_report.json", "best_params.pt"):
                expected_run_files.add(f"{label}/{rel}/{name}")
    orphan_reports, orphan_ckpts = [], []
    for f in scanned:
        if _is_rejected(f):
            continue
        fid = f"{f.label}/{f.rel}"
        if f.kind == "train_report" and fid not in expected_run_files:
            orphan_reports.append(fid)
        elif f.kind == "adapter_checkpoint" and fid not in expected_run_files:
            orphan_ckpts.append(fid)
    unbound_evals = [e["path"] for e in live_evals
                     if e.get("bound_run") is None]

    # ---- 4B quarantine -----------------------------------------------------
    quarantine = analyze_quarantine(scanned) if results_4b.is_dir() else {}

    # live 4B known-invalid artifacts: corrupt/invalid/unbound payloads,
    # unreadable/degenerate/noncanonical reports — zero allowed for READY.
    live_4b_invalid: list[str] = []
    for rv in run_verdicts:
        if rv["root"] != "4b" or rv["scope"] != "retained_candidate":
            continue
        ck = rv.get("checkpoint") or {}
        if ck.get("classification") in (CKPT_CORRUPT, CKPT_INVALID,
                                        CKPT_LEGACY_UNBOUND):
            live_4b_invalid.append(f"4b/{rv['dir']}/best_params.pt:"
                                   f"{ck.get('classification')}")
    for rel in quarantine.get("live_invalid_or_degenerate_reports", []):
        live_4b_invalid.append(f"4b/runs/{rel}:invalid_report")
    for rel in quarantine.get("live_noncanonical_reports", []):
        live_4b_invalid.append(f"4b/runs/{rel}:noncanonical_json")
    live_4b_invalid.extend(
        f"4b/{fid}:orphan" for fid in orphan_ckpts + orphan_reports
        if fid.startswith("4b/"))

    # ---- proof tests ---------------------------------------------------------
    proof = proof_result
    if proof is None and not dry_run and not skip_proof_tests:
        proof = run_proof_tests(repo_root, log_path=proof_log_path)

    fp_after = fingerprint_roots(roots)
    if fp_after != fp_before:
        raise GateExecutionError(
            "input trees changed while the gate was reading them "
            "(source fingerprint mismatch); refusing to emit a verdict")

    # ---- prerequisites -------------------------------------------------------
    two_b_runs = [rv for rv in run_verdicts if rv["root"] == "2b"]
    # Only retained 2B checkpoints are candidates for reuse; rejected 4B
    # payloads are negative evidence and stay classified without blocking.
    candidate_ckpt_rvs = [rv for rv in run_verdicts
                          if rv.get("checkpoint")
                          and rv["scope"] == "retained_candidate"
                          and rv["root"] == "2b"]
    rejected_ckpt_rvs = [rv for rv in run_verdicts
                         if rv.get("checkpoint")
                         and rv["scope"] == "rejected_negative"]
    ckpt_infos = [rv["checkpoint"] for rv in candidate_ckpt_rvs]
    n_ckpts = len(ckpt_infos)
    n_legacy = sum(c.get("classification") == CKPT_LEGACY_UNBOUND
                   for c in ckpt_infos)
    n_corrupt = sum(c.get("classification") == CKPT_CORRUPT
                    for c in ckpt_infos)
    n_loadable = sum(c.get("classification") == CKPT_LOADABLE
                     for c in ckpt_infos)
    n_dupes = sum("duplicate_of" in c for c in ckpt_infos)
    n_bf16 = sum(any(str(r).startswith("non_fp32_trainables_stored")
                     for r in c.get("reasons", []))
                 for c in ckpt_infos)
    n_strict_ok = sum(
        c.get("classification") == CKPT_LOADABLE
        and not any(str(r).startswith("non_fp32_trainables_stored")
                    for r in c.get("reasons", []))
        and "duplicate_of" not in c
        for c in ckpt_infos)

    prereqs: list[dict] = []

    def prereq(pid, status, detail):
        prereqs.append({"id": pid, "status": status, "detail": detail})

    prereq(PREREQ_INVENTORY,
           STATUS_PROVEN if not errors else STATUS_FAILED,
           f"{len(scanned)} files hashed"
           + ("" if not errors else f"; {len(errors)} unreadable"))

    schema_failures = [rv["run_id"] for rv in two_b_runs
                       if rv.get("report_problems")]
    precision_missing = [rv["run_id"] for rv in two_b_runs
                         if "trainable_precision:missing"
                         in rv.get("report_problems", [])]
    precision_not_fp32 = [rv["run_id"] for rv in two_b_runs
                          if "trainable_precision:not_fp32"
                          in rv.get("report_problems", [])]
    suite_mismatch = [rv["run_id"] for rv in two_b_runs
                      if rv.get("suite_sha256_matches_current_suite") is False]
    reports_ok = not schema_failures and not suite_mismatch
    prereq(PREREQ_REPORTS,
           STATUS_PROVEN if reports_ok else STATUS_FAILED,
           (f"{len(two_b_runs)}/{len(two_b_runs)} retained 2B reports fully "
            f"schema-valid (strict JSON, exact integer steps, finite "
            f"metrics, fp32 precision claim) with pinned revision + suite "
            f"hash matching the recomputed behavioral-v2 suite"
            if reports_ok else
            f"{len(schema_failures)} 2B reports carry schema violations "
            f"(violations are blockers, never defaults): "
            + "; ".join(
                f"{rv['run_id']}: {','.join(rv['report_problems'])}"
                for rv in sorted(two_b_runs,
                                 key=lambda r: r["run_id"])
                if rv.get("report_problems"))))

    ckpt_strict_ok = n_ckpts > 0 and n_strict_ok == n_ckpts and \
        not dup_members
    prereq(PREREQ_CKPT_STRICT,
           STATUS_PROVEN if ckpt_strict_ok else STATUS_FAILED,
           (f"{n_strict_ok}/{n_ckpts} retained 2B checkpoints strictly "
            f"loadable, identity-bound, uniquely bound and carrying ACTUAL "
            f"fp32 trainables in the payload; {n_legacy} legacy-unbound, "
            f"{n_corrupt} corrupt, {n_dupes} duplicate-membered, "
            f"{n_bf16} non-fp32"))

    non_rescorable = [e for e in eval_verdicts
                      if e.get("status") == MISSING_RAW_PREDICTION]
    invalid_evals = [e for e in eval_verdicts
                     if is_invalid_eval_status(e.get("status"))]
    unreadable_evals = [e for e in eval_verdicts
                        if e.get("status") == EVAL_UNREADABLE]
    no_record_evals = [e for e in eval_verdicts
                       if e.get("status") == EVAL_NO_RECORDS]
    suite_mismatch_evals = [
        e for e in eval_verdicts
        if e.get("status") == EVAL_OK
        and e.get("suite_sha256_matches_current_suite") is False]
    rescored = [e for e in eval_verdicts if e.get("status") == EVAL_OK]
    if dry_run:
        rescore_status, rescore_detail = STATUS_UNPROVEN, \
            "eval payload inspection skipped (dry run)"
    elif (non_rescorable or invalid_evals or unreadable_evals
          or no_record_evals or suite_mismatch_evals):
        rescore_status = STATUS_FAILED
        parts = []
        if unreadable_evals:
            parts.append(f"{len(unreadable_evals)} unreadable")
        if no_record_evals:
            parts.append(f"{len(no_record_evals)} without records")
        if invalid_evals:
            parts.append(f"{len(invalid_evals)} invalid records")
        if non_rescorable:
            parts.append(f"{len(non_rescorable)} missing raw predictions")
        if suite_mismatch_evals:
            parts.append(f"{len(suite_mismatch_evals)} wrong suite hash")
        rescore_detail = \
            f"eval evidence rejected ({', '.join(parts)})"
    elif eval_verdicts:
        rescore_status = STATUS_PROVEN
        rescore_detail = f"{len(rescored)} eval files fully rescored"
    else:
        rescore_status, rescore_detail = STATUS_UNPROVEN, "no eval files"
    prereq(PREREQ_RESCORE, rescore_status, rescore_detail)

    selection_ok = all(
        (rv.get("selection_check") or {}).get("consistent_with_reported")
        for rv in two_b_runs if rv.get("selection_check"))
    corrected_histories = bool(two_b_runs) and all(
        rv.get("scorer_tag") == corrected_scorer_tag for rv in two_b_runs)
    if rescore_status == STATUS_PROVEN and corrected_histories and selection_ok:
        selection_status = STATUS_PROVEN
        selection_detail = ("best_step values reproduce from histories "
                            "produced under the corrected gold-aware scorer "
                            f"({corrected_scorer_tag})")
    elif not selection_ok:
        selection_status = STATUS_FAILED
        selection_detail = ("recomputed selection disagrees with reported "
                            "best_step or the history was invalid")
    else:
        selection_status = STATUS_UNPROVEN
        selection_detail = ("recomputed selections agree arithmetically with "
                            "every reported best_step"
                            if selection_ok else "selection inconsistent") + \
                           ("; but every history originates from the "
                            "invalidated historical metric, so no "
                            "artifact-level corrected-metric selection exists")
    prereq(PREREQ_SELECTION, selection_status, selection_detail)

    prereq(PREREQ_RUNTIME,
           STATUS_PROVEN if proof and proof["all_passed"]
           else STATUS_FAILED if proof and not proof["all_passed"]
           else STATUS_UNPROVEN,
           ("focused regressions passed: adapter strict metadata+roundtrip, "
            "fp32 trainables over bf16 backbone, non-finite rejection, "
            "cached-recurrence equivalence, gold-position scoring "
            "invariance" if proof else
            "proof tests skipped (dry-run/skip flag)"))

    if not results_4b.is_dir():
        quarantine_status, quarantine_detail = STATUS_FAILED, \
            "4B results root missing"
    elif not quarantine.get("quarantine_has_any_evidence") \
            and not quarantine.get("live_run_dirs"):
        quarantine_status, quarantine_detail = STATUS_PROVEN, \
            "nothing to quarantine: no rejected batch and no live 4B runs"
    else:
        quarantine_ok = bool(quarantine.get("quarantine_marker_present")) and \
            bool(quarantine.get("quarantine_nonempty_complete")) and \
            not quarantine.get("live_tree_duplicates_quarantine") and \
            not live_4b_invalid
        quarantine_status = STATUS_PROVEN if quarantine_ok else STATUS_FAILED
        quarantine_detail = (
            "rejected NaN batch completely quarantined (marker + "
            f"{quarantine.get('rejected_report_files')} reports + "
            f"{quarantine.get('rejected_checkpoint_files')} checkpoints); "
            "live tree holds zero known-invalid or byte-duplicate artifacts"
            if quarantine_ok else
            f"marker_present="
            f"{quarantine.get('quarantine_marker_present')}, "
            f"nonempty_complete="
            f"{quarantine.get('quarantine_nonempty_complete')}, "
            f"live_duplicates_quarantine="
            f"{quarantine.get('live_tree_duplicates_quarantine')}, "
            f"live_known_invalid={len(live_4b_invalid)}")
    prereq(PREREQ_QUARANTINE, quarantine_status, quarantine_detail)

    n_join_complete = sum(j.complete for j in joins.values())
    # The join is complete only when every retained run joins fully AND no
    # evidence floats outside the join (orphans, unbound evals) — global
    # counts can never substitute for per-artifact coverage.
    join_ok = bool(retained_2b) and n_join_complete == len(retained_2b) \
        and not orphan_reports and not orphan_ckpts and not unbound_evals
    if dry_run:
        join_status, join_detail = STATUS_UNPROVEN, \
            "relational join skipped (dry run)"
    elif join_ok:
        join_status = STATUS_PROVEN
        join_detail = (f"{n_join_complete}/{len(retained_2b)} retained 2B "
                       "runs join report<->fp32 identity-bound checkpoint-> "
                       "valid current-suite raw evals on BOTH mandatory "
                       "splits (test_id, test_ood)")
    else:
        join_status = STATUS_FAILED
        gaps = []
        for rv in sorted(retained_2b, key=lambda r: r["dir"]):
            jv = joins.get(rv["dir"])
            if jv is None or jv.complete:
                continue
            missing = sorted(set(MANDATORY_EVAL_SPLITS)
                             - set(jv.eval_splits_covered))
            gaps.append(f"{rv['dir']}: missing_splits={missing}"
                        f"{'' if jv.report_valid else '; report_invalid'}"
                        f"{'' if jv.checkpoint_valid else '; checkpoint_invalid'}"
                        + (f'; identity_mismatch='
                           f"{jv.identity_mismatch_evals}"
                           if jv.identity_mismatch_evals else ""))
        join_detail = "; ".join(gaps) if gaps else "join incomplete"
    prereq(PREREQ_JOIN, join_status, join_detail)

    status_of = {p["id"]: p["status"] for p in prereqs}

    # ---- blockers --------------------------------------------------------------
    blockers: list[dict] = []

    def blocker(code, detail):
        blockers.append({"code": code, "detail": detail,
                         "smallest_next_action":
                             BLOCKER_ACTIONS.get(code, NEXT_ACTION_GENERIC)})

    if errors:
        blocker("INVENTORY_UNREADABLE_FILES", "; ".join(sorted(errors)))
    if precision_missing:
        blocker("REPORT_TRAINABLE_PRECISION_MISSING",
                f"{len(precision_missing)}/{len(two_b_runs)} 2B reports lack "
                f"explicit trainable-precision fields")
    if precision_not_fp32:
        blocker("REPORT_TRAINABLE_PRECISION_NOT_FP32",
                f"{len(precision_not_fp32)} 2B reports claim a non-fp32 "
                f"trainable precision: {sorted(precision_not_fp32)}")
    if schema_failures:
        blocker("REPORT_SCHEMA_MISSING_FIELDS",
                f"{len(schema_failures)}/{len(two_b_runs)} 2B reports "
                f"violate the required train-report schema")
    if n_bf16:
        blocker("TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS",
                f"{n_bf16}/{n_ckpts} retained checkpoints store LoRA "
                f"trainables in a non-fp32 dtype IN THE ACTUAL PAYLOAD, "
                f"contradicting the fp32-trainables prerequisite at "
                f"artifact level")
    if n_legacy:
        blocker("CKPT_LEGACY_UNBOUND_IDENTITY",
                f"{n_legacy}/{n_ckpts} checkpoints are plain state dicts "
                f"without in-bundle model_id/revision; strict identity-bound "
                f"loading is impossible")
    if n_corrupt:
        blocker("CKPT_CORRUPT",
                f"{n_corrupt}/{n_ckpts} checkpoints are corrupt/non-finite")
    if dup_members:
        blocker("CKPT_AMBIGUOUS_DUPLICATE_BINDING",
                f"{len(dup_members)} live-side checkpoint files are "
                f"byte-identical to another binding: {sorted(dup_members)}")
    if orphan_reports or orphan_ckpts:
        blocker("ORPHAN_EVIDENCE",
                f"orphan train_reports={sorted(orphan_reports)}; "
                f"orphan checkpoints={sorted(orphan_ckpts)}")
    if unreadable_evals or invalid_evals or no_record_evals:
        bad = [(e.get("path") or e["file"], e.get("status"),
                e.get("detail"))
               for e in unreadable_evals + invalid_evals + no_record_evals]
        blocker("EVAL_EVIDENCE_INVALID",
                f"{len(bad)} eval files are unreadable, record-less or "
                f"carry invalid raw rankings: {sorted(bad)}")
    if suite_mismatch_evals:
        blocker("EVAL_SUITE_HASH_MISMATCH",
                f"{len(suite_mismatch_evals)} eval files declare a suite "
                f"hash differing from the current suite: "
                f"{[e.get('path') or e['file'] for e in suite_mismatch_evals]}")
    identity_mismatch_evals = sorted(
        {f for jv in joins.values() for f in jv.identity_mismatch_evals})
    if identity_mismatch_evals:
        blocker("EVAL_IDENTITY_MISMATCH",
                f"evals whose model/revision differ from the joined run "
                f"identity: {identity_mismatch_evals}")
    if unbound_evals:
        blocker("EVAL_UNBOUND_TO_RETAINED_RUN",
                f"eval files whose adapter binds to no retained 2B run: "
                f"{sorted(unbound_evals)}")
    coverage_gaps = [rv["dir"] for rv in retained_2b
                     if not (joins.get(rv["dir"])
                             and joins[rv["dir"]].complete)]
    if coverage_gaps:
        blocker("EVAL_COVERAGE_INCOMPLETE",
                f"{len(coverage_gaps)}/{len(retained_2b)} retained 2B runs "
                f"lack a valid identity-bound corrected raw eval for BOTH "
                f"mandatory splits: {sorted(coverage_gaps)}")
    if non_rescorable:
        blocker(MISSING_RAW_PREDICTION,
                f"{len(non_rescorable)}/{len(eval_verdicts)} eval files "
                f"cannot be rescored: records lack raw scorer inputs")
    if status_of[PREREQ_SELECTION] != STATUS_PROVEN:
        blocker("SELECTION_PROVENANCE_NOT_CORRECTED",
                "reported best_step values are not provably selected by the "
                "corrected metric over a fully valid history")
    if not quarantine.get("quarantine_marker_present") and \
            quarantine.get("quarantine_has_any_evidence"):
        blocker("QUARANTINE_MARKER_MISSING", "REJECTED.md absent")
    if quarantine.get("quarantine_has_any_evidence") and \
            not quarantine.get("quarantine_nonempty_complete"):
        blocker("QUARANTINE_INCOMPLETE",
                "quarantine tree is marker-only or lacks a complete "
                "rejected report/checkpair pair")
    if quarantine.get("live_tree_duplicates_quarantine"):
        blocker("LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH",
                f"LIVE 4B runs/ tree byte-duplicates rejected evidence "
                f"({len(quarantine['live_vs_rejected_identical_files'])} "
                f"identical files); partial differences do not mask this")
    if live_4b_invalid:
        blocker("LIVE_4B_KNOWN_INVALID_ARTIFACTS",
                f"{len(live_4b_invalid)} known-invalid live 4B artifacts: "
                f"{sorted(live_4b_invalid)}")
    if proof and not proof["all_passed"]:
        blocker("PROOF_TESTS_FAILED", f"pytest rc={proof['returncode']}")
    unproven_ids = sorted(p["id"] for p in prereqs
                          if p["status"] == STATUS_UNPROVEN)
    if unproven_ids:
        blocker("PREREQS_UNPROVEN",
                f"prerequisites not proven (skipped or missing evidence): "
                f"{unproven_ids}; READY requires every prerequisite PROVEN")

    verdict = VERDICT_READY if all(p["status"] == STATUS_PROVEN
                                   for p in prereqs) else VERDICT_NOT_READY

    if dry_run:
        finiteness_line = "checkpoint payload finiteness not checked (dry run)"
    elif n_ckpts and n_corrupt == 0:
        finiteness_line = (f"all {n_ckpts} checkpoint payloads deserialize "
                           f"safely and are fully finite")
    else:
        finiteness_line = f"{n_corrupt} corrupt checkpoint payloads detected"
    positive = [
        f"discovered {len(two_b_runs)} retained 2B run directories, "
        f"{len(quarantine.get('live_run_dirs', []))} live and "
        f"{len(quarantine.get('rejected_run_dirs', []))} rejected 4B run "
        f"directories by content scan, not by a hand list",
        f"hashed {len(scanned)} files "
        f"({sum(f.size for f in scanned)} bytes)",
        f"{sum(rv.get('suite_sha256_matches_current_suite') is True for rv in two_b_runs)}"
        f"/{len(two_b_runs)} 2B reports pin a suite hash matching the "
        f"locally recomputed behavioral-v2 suite digest",
        f"{sum((rv.get('selection_check') or {}).get('consistent_with_reported') is True for rv in two_b_runs)}"
        f"/{len(two_b_runs)} reported best_step values reproduce exactly "
        f"from their own val_history (poisoning was metric-level, not "
        f"bookkeeping-level)",
        f"{n_join_complete}/{len(retained_2b)} retained 2B runs pass the "
        f"full report<->checkpoint<->eval relational join",
        finiteness_line,
    ]

    counts = {
        "files_scanned": len(scanned),
        "bytes_scanned": sum(f.size for f in scanned),
        "runs_2b": len(two_b_runs),
        "runs_2b_join_complete": n_join_complete,
        "runs_4b_live": len(quarantine.get("live_run_dirs", [])),
        "runs_4b_rejected": len(quarantine.get("rejected_run_dirs", [])),
        "checkpoints_total": sum(
            rv.get("checkpoint") is not None for rv in run_verdicts),
        "checkpoints_rejected_negative": len(rejected_ckpt_rvs),
        "checkpoints_loadable": n_loadable,
        "checkpoints_legacy_unbound": n_legacy,
        "checkpoints_corrupt": n_corrupt,
        "checkpoints_duplicate_membered": n_dupes,
        "checkpoints_non_fp32_stored": n_bf16,
        "eval_files_checked": len(eval_verdicts),
        "eval_files_rescored_corrected": len(rescored),
        "eval_files_missing_raw_prediction": len(non_rescorable),
        "eval_files_invalid": len(invalid_evals) + len(unreadable_evals)
        + len(no_record_evals) + len(suite_mismatch_evals),
        "eval_files_unbound": len(unbound_evals),
        "orphan_artifacts": len(orphan_reports) + len(orphan_ckpts),
        "live_4b_known_invalid": len(live_4b_invalid),
        "source_fingerprint_stable": fp_before == fp_after,
    }

    gate_verdict = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": {"id": GATE_ID, "variant": GATE_VARIANT},
        "mode": "dry_run" if dry_run else "full",
        "verdict": verdict,
        "inputs": {
            "results_2b": str(results_2b),
            "results_4b": str(results_4b),
            "proof_tests_executed": proof is not None,
            "source_fingerprint_before": fp_before,
            "source_fingerprint_after": fp_after,
            "source_unchanged": fp_before == fp_after,
        },
        "counts": counts,
        "prerequisites": prereqs,
        "blockers": sorted(blockers, key=lambda b: b["code"]),
        "positive_proofs": positive,
        "trust_boundary": TRUST_BOUNDARY,
        "exit_semantics": {"0": VERDICT_READY, "1": VERDICT_NOT_READY,
                           "2": "execution_error"},
        "outputs": ["artifact_inventory.json", "artifact_verdicts.json",
                    "gate_verdict.json", "GATE_REPORT.md"]
                    + ([] if proof is None else ["proof_tests.log"]),
    }

    artifact_verdicts = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": {"id": GATE_ID, "variant": GATE_VARIANT},
        "runs": sorted(run_verdicts, key=lambda r: (r["root"], r["run_id"])),
        "evaluations": eval_verdicts,
        "quarantine_4b": quarantine,
        "orphans": {
            "train_reports": sorted(orphan_reports),
            "checkpoints": sorted(orphan_ckpts),
            "unbound_evals": sorted(unbound_evals),
        },
        "live_4b_known_invalid": sorted(live_4b_invalid),
        "proof_tests": None if proof is None else {
            "all_passed": proof["all_passed"],
            "returncode": proof["returncode"],
            "nodes": proof["nodes"],
        },
        "scan_errors": sorted(errors),
    }

    report_md = render_markdown(gate_verdict)
    return GateResult(verdict=verdict, inventory=inventory,
                      artifact_verdicts=artifact_verdicts,
                      gate_verdict=gate_verdict, report_md=report_md)


def render_markdown(v: dict) -> str:
    lines = [
        f"# No-Spend Integrity Gate ({v['gate']['variant']})",
        "",
        f"**Verdict: `{v['verdict']}`** (mode: {v['mode']})",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in sorted(v["counts"]):
        lines.append(f"| {k} | {v['counts'][k]} |")
    lines += ["", "## Prerequisites", "",
              "| id | status | detail |", "|---|---|---|"]
    for p in v["prerequisites"]:
        detail = p["detail"].replace("|", "\\|")
        lines.append(f"| {p['id']} | {p['status']} | {detail} |")
    lines += ["", "## Blockers", ""]
    if v["blockers"]:
        for b in v["blockers"]:
            lines += [f"### `{b['code']}`", "", b["detail"], "",
                      f"**Next action:** {b['smallest_next_action']}", ""]
    else:
        lines += ["None.", ""]
    lines += ["## Positive proofs", ""]
    for s in v["positive_proofs"]:
        lines.append(f"- {s}")
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m latent_lab.bench.no_spend_gate \\",
        "    --results-2b .rcc_work/remote_results \\",
        "    --results-4b .rcc_work/remote_results_4b \\",
        "    --out .rcc_work/no_spend_gate_20260824",
        "```",
        "",
        "Exit codes: `0` READY · `1` NOT_READY · `2` execution error.",
        "Hardware-free dry run: add `--dry-run` (metadata/hashes only).",
        "",
        "## Trust boundary",
        "",
        v["trust_boundary"],
        "",
    ]
    return "\n".join(lines)


def _exit_for(verdict: str) -> int:
    if verdict == VERDICT_READY:
        return 0
    if verdict == VERDICT_NOT_READY:
        return 1
    raise ValueError(f"unknown gate verdict {verdict!r}")


def _paths_overlap(a: Path, b: Path) -> bool:
    ra, rb = os.path.realpath(a), os.path.realpath(b)
    return ra == rb or ra.startswith(rb + os.sep) or \
        rb.startswith(ra + os.sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="no_spend_gate",
        description="Bounded no-spend integrity gate (READY / NOT_READY).")
    ap.add_argument("--results-2b", default=".rcc_work/remote_results")
    ap.add_argument("--results-4b", default=".rcc_work/remote_results_4b")
    ap.add_argument("--out", default=f".rcc_work/no_spend_gate_{GATE_VARIANT}")
    ap.add_argument("--dry-run", action="store_true",
                    help="hash/metadata only: no tensor payload loading, "
                         "no proof tests")
    ap.add_argument("--skip-proof-tests", action="store_true")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="do not write telemetry_timestamp.json")
    args = ap.parse_args(argv)

    repo_root = Path.cwd()
    results_2b, results_4b = Path(args.results_2b), Path(args.results_4b)
    if not results_2b.is_dir() or not results_4b.is_dir():
        print(f"execution error: input roots missing "
              f"({results_2b}, {results_4b})", file=sys.stderr)
        return 2
    # Fail-closed invocation guard: reject output/input overlap BEFORE any
    # scan or write so emitting outputs can never mutate the evidence.
    out_dir = Path(args.out)
    for label, root in (("results-2b", results_2b), ("results-4b", results_4b)):
        if _paths_overlap(out_dir, root):
            print(f"execution error: output directory {out_dir} overlaps "
                  f"input {label}={root}; refusing to write", file=sys.stderr)
            return 2

    fp_pre = None
    try:
        result = run_gate(results_2b, results_4b, repo_root=repo_root,
                          dry_run=args.dry_run,
                          skip_proof_tests=args.skip_proof_tests,
                          proof_log_path=out_dir / "proof_tests.log")
        fp_pre = result.gate_verdict["inputs"]["source_fingerprint_after"]
    except GateExecutionError as e:
        print(f"execution error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"execution error: {e}", file=sys.stderr)
        return 2

    try:
        inv_bytes = canonical_json_bytes(result.inventory)
        verd_bytes = canonical_json_bytes(result.artifact_verdicts)
        write_canonical(out_dir / "artifact_inventory.json", result.inventory)
        write_canonical(out_dir / "artifact_verdicts.json",
                        result.artifact_verdicts)
        result.gate_verdict["artifact_digests"] = {
            "artifact_inventory.json": hashlib.sha256(inv_bytes).hexdigest(),
            "artifact_verdicts.json": hashlib.sha256(verd_bytes).hexdigest(),
        }
        write_canonical(out_dir / "gate_verdict.json", result.gate_verdict)
        (out_dir / "GATE_REPORT.md").write_text(result.report_md,
                                                encoding="utf-8")
    except OSError as e:
        print(f"execution error: writing outputs failed: {e}",
              file=sys.stderr)
        return 2

    # Post-write source check: the evidence trees must be byte-identical to
    # what was fingerprinted before the scan, proving read-only behavior.
    try:
        fp_post = fingerprint_roots([("2b", results_2b), ("4b", results_4b)])
    except GateExecutionError as e:
        print(f"execution error: {e}", file=sys.stderr)
        return 2
    if fp_post != fp_pre:
        print("execution error: input trees changed during the gate run "
              "(post-write fingerprint mismatch)", file=sys.stderr)
        return 2

    if not args.no_telemetry:
        import time

        write_canonical(out_dir / "telemetry_timestamp.json", {
            "note": "the ONLY wall-clock output; never referenced by "
                    "canonical verdicts",
            "telemetry_generated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    print(f"{result.verdict}: {len(result.gate_verdict['blockers'])} "
          f"blocker(s) -> {out_dir/'gate_verdict.json'}")
    return _exit_for(result.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
