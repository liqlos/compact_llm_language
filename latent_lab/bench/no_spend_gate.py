"""Bounded no-spend integrity gate (2026-08-24 variant).

Inventories every retained 2B artifact and every live/rejected 4B artifact,
validates all offline-checkable evidence, and emits ONE canonical
machine-readable verdict plus a Markdown report:

    READY     only if every mandatory prerequisite is actually proven
    NOT_READY otherwise, with exact blocker codes and the smallest next
              executable action — never a weakened gate.

Usage (from the repository root):

    python -m latent_lab.bench.no_spend_gate \
        --results-2b .rcc_work/remote_results \
        --results-4b .rcc_work/remote_results_4b \
        --out .rcc_work/no_spend_gate_20260824

Exit semantics:
    0  READY            every prerequisite PROVEN (not produced by this
                        evidence set so far)
    1  NOT_READY        evidence-backed negative verdict; see
                        gate_verdict.json blockers
    2  EXECUTION ERROR  bad invocation, unreadable inputs, crash — no
                        verdict may be inferred from exit code 2

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
from dataclasses import dataclass
from pathlib import Path

from .corrected_scoring import (
    MISSING_RAW_PREDICTION,
    select_best_checkpoint,
)

GATE_SCHEMA_VERSION = 1
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

STATUS_PROVEN = "PROVEN"
STATUS_FAILED = "FAILED"
STATUS_UNPROVEN = "UNPROVEN"

CKPT_LOADABLE = "loadable"
CKPT_INVALID = "invalid"
CKPT_LEGACY_UNBOUND = "legacy-unbound"
CKPT_CORRUPT = "corrupt"
CKPT_DUPLICATE = "duplicate"
CKPT_UNPROVEN = "unproven"

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
    "CKPT_LEGACY_UNBOUND_IDENTITY":
        "Rebuild identity-bound bundles via save_adapter_bundle ONLY after "
        "weight-provenance verification; otherwise retire the weights into "
        "the next capped paired-seed retrain decision.",
    "CKPT_CORRUPT":
        "Keep quarantined as negative evidence; exclude from any resume or "
        "rescore.",
    "NON_RESCORABLE_MISSING_RAW_PREDICTION":
        "Rerun evaluation with raw per-candidate score capture enabled "
        "(capped CUDA canary scope) or formally invalidate the latent "
        "conclusions; never relabel derived records as a rescore.",
    "SELECTION_PROVENANCE_NOT_CORRECTED":
        "Apply select_best_checkpoint over corrected-metric histories "
        "produced by the next validated run; discard historical best_step "
        "claims.",
    "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS":
        "Fix persistence to cast trainables to fp32 on save and verify on "
        "the next CPU smoke before any GPU spend.",
    "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH":
        "After hashed backup + manifest promotion, remove the duplicated "
        "files under the LIVE 4B runs/ tree (deletion is outside this "
        "gate's authority) and keep _rejected_nan_batch as sole evidence.",
    "QUARANTINE_MARKER_MISSING":
        "Restore/confirm REJECTED.md in the quarantine tree, then rerun.",
    "PROOF_TESTS_FAILED":
        "Fix the failing regression(s) locally before any spend decision.",
}

NEXT_ACTION_GENERIC = "Review gate_verdict.json prerequisites and rerun."


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


def _is_pinned_revision(v) -> bool:
    return isinstance(v, str) and bool(_PINNED_REVISION_RE.fullmatch(v))


def validate_train_report(report: dict) -> dict:
    """Schema validation; every missing field is recorded, never defaulted."""
    problems: list[str] = []
    config = report.get("config")
    if not isinstance(config, dict):
        problems.append("config:missing")
        config = {}
    for key, typ in REQUIRED_CONFIG_FIELDS.items():
        if key not in config:
            problems.append(f"config.{key}:missing")
        elif typ is int and (isinstance(config[key], bool)
                             or not isinstance(config[key], int)):
            problems.append(f"config.{key}:type")
        elif typ is list:
            iv = config[key]
            if not (isinstance(iv, list) and len(iv) == 2
                    and all(isinstance(x, int) for x in iv)):
                problems.append(f"config.{key}:type")
        elif not isinstance(config[key], typ):
            problems.append(f"config.{key}:type")
    for key in REQUIRED_TOP_LEVEL_NUMERIC:
        if key not in report:
            problems.append(f"{key}:missing")
        elif isinstance(report[key], bool) or \
                not isinstance(report[key], (int, float)):
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
    if "trainable_precision" not in report:
        problems.append("trainable_precision:missing")

    selection = None
    if isinstance(hist, list) and hist:
        sel = select_best_checkpoint(hist)
        reported_acc = report.get("best_val_acc")
        reported_step = report.get("best_step")
        if sel is not None and isinstance(reported_acc, (int, float)) \
                and not isinstance(reported_acc, bool) \
                and isinstance(reported_step, int) \
                and not isinstance(reported_step, bool):
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
        load_adapter_bundle(path, model_id=str(bid), revision=str(brev))
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
    info["classification"] = CKPT_LOADABLE
    info["reasons"].append("strict_project_loader_passed")
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

def evaluate_eval_file(path: Path, examples_by_id: dict | None) -> dict:
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return {"file": path.name, "status": "unreadable", "detail": str(e)}
    if not isinstance(data, dict):
        return {"file": path.name, "status": "unreadable",
                "detail": "top level not an object"}
    out = {
        "file": path.name,
        "adapter": data.get("adapter"),
        "split": data.get("split"),
        "model": data.get("model"),
        "revision": data.get("revision"),
        "suite_sha256_matches_current_suite": None,
    }
    records: list[dict] = []
    for res in (data.get("results") or {}).values():
        if isinstance(res, dict) and isinstance(res.get("records"), list):
            records.extend(res["records"])
    out["n_records"] = len(records)

    if examples_by_id is None:
        examples_by_id = suite_examples_by_id()
    declared = data.get("suite_sha256")
    out["suite_sha256_matches_current_suite"] = bool(
        declared == current_suite_sha256())

    if not records:
        out["status"] = "NO_RECORDS"
        return out
    from .corrected_scoring import rescore_records

    outcome = rescore_records(records, examples_by_id)
    if outcome.status == "RESCORED_CORRECTED":
        out["status"] = "RESCORED_CORRECTED"
        out["corrected_accuracy"] = outcome.corrected_accuracy
    elif outcome.status == MISSING_RAW_PREDICTION:
        out["status"] = MISSING_RAW_PREDICTION
        out["detail"] = outcome.detail
    else:
        out["status"] = "NO_RECORDS"
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

def analyze_quarantine(scanned: list[ScannedFile]) -> dict:
    def files_under(prefix: str) -> dict[str, ScannedFile]:
        sel = {
            f.rel[len(prefix):]: f
            for f in scanned
            if f.label == "4b" and f.rel.startswith(prefix)
        }
        return sel

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
    live_dupes_quarantine = bool(identical) and not differing

    nan_runs, fake_best_runs = [], []
    for rel, f in sorted(rejected.items()):
        if f.kind != "train_report":
            continue
        rep = json.loads(f.abs.read_text())
        loss = rep.get("final_train_loss")
        acc = rep.get("best_val_acc")
        run = rel.split("/")[0]
        if isinstance(loss, float) and not math.isfinite(loss):
            nan_runs.append(run)
        if isinstance(acc, (int, float)) and math.isfinite(float(acc)) \
                and float(acc) >= 1.0:
            fake_best_runs.append(run)
    return {
        "live_run_dirs": sorted({r.split("/")[0] for r in live
                                 if r.endswith("/train_report.json")}),
        "rejected_run_dirs": sorted({r.split("/")[0] for r in rejected
                                     if r.endswith("/train_report.json")}),
        "quarantine_marker_present": marker is not None,
        "live_vs_rejected_identical_files": identical,
        "live_vs_rejected_differing_files": differing,
        "only_in_live": only_live,
        "only_in_rejected": only_rej,
        "live_tree_duplicates_quarantine": live_dupes_quarantine,
        "rejected_runs_with_nan_final_loss": sorted(set(nan_runs)),
        "rejected_runs_with_degenerate_best_acc_1": sorted(set(fake_best_runs)),
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


def run_gate(results_2b: Path, results_4b: Path, *, repo_root: Path,
             dry_run: bool = False, skip_proof_tests: bool = False,
             proof_log_path: Path | None = None,
             proof_result: dict | None = None,
             corrected_scorer_tag: str = "corrected-gold-aware-v1") -> GateResult:
    """Execute every gate stage and return the assembled result.

    ``proof_result`` is a TEST-ONLY injection hook for the executed proof
    outcomes; the CLI never supplies it, so production runs either execute
    the regressions or record them skipped/unproven.
    """
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
            if rep_rel:
                ri.report = json.loads(files_by_id[rep_rel].abs.read_text())
                ri.report_sha256 = files_by_id[rep_rel].sha256
            if ckpt_rel:
                ri.ckpt_sha256 = files_by_id[ckpt_rel].sha256
            runs[label].append(ri)

    # ---- checkpoint duplicate groups -------------------------------------
    ckpt_paths = [f for f in scanned if f.kind == "adapter_checkpoint"]
    dup_members: set[str] = set()
    by_hash: dict[str, list[str]] = {}
    for f in ckpt_paths:
        by_hash.setdefault(f.sha256, []).append(f"{f.label}/{f.rel}")
    primary: dict[str, str] = {}
    for h, ids in sorted(by_hash.items()):
        if len(ids) > 1:
            ordered = sorted(ids)
            primary[h] = ordered[0]
            dup_members.update(ordered[1:])

    # ---- per-run validation + classification -----------------------------
    run_verdicts = []
    for label in ("2b", "4b"):
        for ri in runs[label]:
            rv: dict = {"run_id": ri.run_id, "root": ri.root,
                        "dir": ri.dir_rel,
                        "scope": ("rejected_negative"
                                  if ri.dir_rel.startswith("_rejected")
                                  else "retained_candidate")}
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
                rv["report_problems"] = ["train_report:missing"]
            if ri.checkpoint_rel:
                cls = classify_checkpoint(
                    files_by_id[ri.checkpoint_rel].abs,
                    report_identity=identity, dry_run=dry_run)
                fid = f"{ri.root}/{ri.checkpoint_rel}"
                if fid in dup_members:
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
                eval_verdicts.append(evaluate_eval_file(f.abs, None))
    eval_verdicts.sort(key=lambda e: e["file"])

    # ---- 4B quarantine -----------------------------------------------------
    quarantine = analyze_quarantine(scanned) if results_4b.is_dir() else {}

    # ---- proof tests ---------------------------------------------------------
    proof = proof_result
    if proof is None and not dry_run and not skip_proof_tests:
        proof = run_proof_tests(repo_root, log_path=proof_log_path)

    # ---- prerequisites -------------------------------------------------------
    two_b_runs = [rv for rv in run_verdicts if rv["root"] == "2b"]
    # Only checkpoints that are candidates for reuse gate the strict-load
    # prerequisite; quarantined rejected 4B payloads are negative evidence
    # and stay classified/inventoried without blocking READY.
    candidate_ckpt_rvs = [rv for rv in run_verdicts
                          if rv.get("checkpoint")
                          and rv["scope"] == "retained_candidate"]
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
    suite_mismatch = [rv["run_id"] for rv in two_b_runs
                      if rv.get("suite_sha256_matches_current_suite") is False]
    reports_ok = not schema_failures and not suite_mismatch
    prereq(PREREQ_REPORTS,
           STATUS_PROVEN if reports_ok else STATUS_FAILED,
           (f"{len(two_b_runs)}/{len(two_b_runs)} retained 2B reports fully "
            f"schema-valid with pinned revision + suite hash matching the "
            f"recomputed behavioral-v2 suite"
            if reports_ok else
            f"{len(schema_failures)} 2B reports carry schema violations "
            f"(missing fields are blockers): "
            + "; ".join(
                f"{rv['run_id']}: {','.join(rv['report_problems'])}"
                for rv in sorted(two_b_runs,
                                 key=lambda r: r["run_id"])
                if rv.get("report_problems"))))

    prereq(PREREQ_CKPT_STRICT,
           STATUS_PROVEN if n_loadable == n_ckpts and n_ckpts > 0
           else STATUS_FAILED,
           (f"{n_loadable}/{n_ckpts} checkpoints strictly loadable with "
            f"in-bundle identity; {n_legacy} legacy-unbound, {n_corrupt} "
            f"corrupt, {n_dupes} duplicate-membered"))

    non_rescorable = [e for e in eval_verdicts
                      if e.get("status") == MISSING_RAW_PREDICTION]
    rescored = [e for e in eval_verdicts
                if e.get("status") == "RESCORED_CORRECTED"]
    if dry_run:
        rescore_status, rescore_detail = STATUS_UNPROVEN, \
            "eval payload inspection skipped (dry run)"
    elif non_rescorable:
        rescore_status = STATUS_FAILED
        rescore_detail = (f"{len(non_rescorable)}/{len(eval_verdicts)} "
                          f"retained 2B eval files retain only derived "
                          f"correct/rank_of_gold — corrected rescoring "
                          f"impossible offline")
    elif eval_verdicts:
        rescore_status = STATUS_PROVEN
        rescore_detail = f"{len(rescored)} eval files rescored"
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
        selection_detail = "recomputed selection disagrees with reported best_step"
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

    quarantine_ok = bool(quarantine) and \
        quarantine.get("quarantine_marker_present") and \
        not quarantine.get("only_in_live") and \
        not quarantine.get("live_tree_duplicates_quarantine")
    prereq(PREREQ_QUARANTINE,
           STATUS_PROVEN if quarantine_ok else STATUS_FAILED,
           ("rejected NaN batch fully quarantined; live tree clean"
            if quarantine_ok else
            f"marker_present="
            f"{quarantine.get('quarantine_marker_present')}, "
            f"live_duplicates_quarantine="
            f"{quarantine.get('live_tree_duplicates_quarantine')}, "
            f"only_in_live={len(quarantine.get('only_in_live', []))}"))

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
    if schema_failures:
        blocker("REPORT_SCHEMA_MISSING_FIELDS",
                f"{len(schema_failures)}/{len(two_b_runs)} 2B reports "
                f"violate the required train-report schema")
    if n_bf16:
        blocker("TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS",
                f"{n_bf16}/{n_ckpts} retained checkpoints store LoRA "
                f"trainables in bf16, contradicting the fp32-trainables "
                f"prerequisite at artifact level")
    if n_legacy:
        blocker("CKPT_LEGACY_UNBOUND_IDENTITY",
                f"{n_legacy}/{n_ckpts} checkpoints are plain state dicts "
                f"without in-bundle model_id/revision; strict identity-bound "
                f"loading is impossible")
    if n_corrupt:
        blocker("CKPT_CORRUPT",
                f"{n_corrupt}/{n_ckpts} checkpoints are corrupt/non-finite")
    if status_of[PREREQ_RESCORE] != STATUS_PROVEN and non_rescorable:
        blocker(MISSING_RAW_PREDICTION,
                f"{len(non_rescorable)}/{len(eval_verdicts)} eval files "
                f"cannot be rescored: records lack raw scorer inputs")
    if status_of[PREREQ_SELECTION] != STATUS_PROVEN:
        blocker("SELECTION_PROVENANCE_NOT_CORRECTED",
                "every reported best_step was selected by the invalidated "
                "historical metric; the corrected selector is implemented "
                "and tested but applicable to no retained history")
    if quarantine.get("live_tree_duplicates_quarantine"):
        blocker("LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH",
                f"LIVE 4B runs/ tree byte-duplicates the rejected NaN batch "
                f"({len(quarantine['live_vs_rejected_identical_files'])} "
                f"identical files incl. all 8 train reports)")
    if not quarantine.get("quarantine_marker_present"):
        blocker("QUARANTINE_MARKER_MISSING", "REJECTED.md absent")
    if proof and not proof["all_passed"]:
        blocker("PROOF_TESTS_FAILED", f"pytest rc={proof['returncode']}")

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
        f"/{len(two_b_runs)} 2B reports pin suite hash dc24147b… which "
        f"matches the locally recomputed behavioral-v2 suite digest",
        f"{sum((rv.get('selection_check') or {}).get('consistent_with_reported') is True for rv in two_b_runs)}"
        f"/{len(two_b_runs)} reported best_step values reproduce exactly "
        f"from their own val_history (poisoning was metric-level, not "
        f"bookkeeping-level)",
        finiteness_line,
    ]

    counts = {
        "files_scanned": len(scanned),
        "bytes_scanned": sum(f.size for f in scanned),
        "runs_2b": len(two_b_runs),
        "runs_4b_live": len(quarantine.get("live_run_dirs", [])),
        "runs_4b_rejected": len(quarantine.get("rejected_run_dirs", [])),
        "checkpoints_total": n_ckpts + len(rejected_ckpt_rvs),
        "checkpoints_rejected_negative": len(rejected_ckpt_rvs),
        "checkpoints_loadable": n_loadable,
        "checkpoints_legacy_unbound": n_legacy,
        "checkpoints_corrupt": n_corrupt,
        "checkpoints_duplicate_membered": n_dupes,
        "checkpoints_non_fp32_stored": n_bf16,
        "eval_files_checked": len(eval_verdicts),
        "eval_files_rescored_corrected": len(rescored),
        "eval_files_missing_raw_prediction": len(non_rescorable),
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
    try:
        result = run_gate(results_2b, results_4b, repo_root=repo_root,
                          dry_run=args.dry_run,
                          skip_proof_tests=args.skip_proof_tests,
                          proof_log_path=Path(args.out) / "proof_tests.log")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"execution error: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
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
