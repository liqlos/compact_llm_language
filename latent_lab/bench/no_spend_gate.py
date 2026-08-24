"""Bounded no-spend integrity gate (2026-08-24 variant).

Inventories every retained 2B artifact and every live/rejected 4B artifact,
validates all offline-checkable evidence, and emits ONE canonical
machine-readable verdict plus a Markdown report:

    READY     only if every mandatory prerequisite is actually PROVEN by
              this evidence set
    NOT_READY otherwise, with exact blocker codes and the smallest next
              executable action — never a weakened gate

FAIL-CLOSED contract (every known fail-open is a blocker, never a skip):

* Evidence parsing is STRICT JSON — NaN/Infinity literals make the file
  invalid instead of silently parsing to float("nan").
* Checkpoints must be identity-bound bundles whose PAYLOAD tensors are
  verified FP32 for every floating trainable; report strings alone prove
  nothing. bf16 payloads classify ``non-fp32-payload`` and block READY.
* The canonical training recipe is DERIVED, never trusted: only a
  schema-valid train report config plus its separately validated
  suite_sha256 yield the canonical recipe via
  ``checkpointing.recipe_from_config``, and the report's own
  ``recipe`` field must EQUAL that derivation. The raw in-bundle
  recipe proves nothing; bundles load strictly against the
  independently derived recipe. A bundle or report whose canonical
  recipe cannot be derived/verified fails closed (never loadable).
* Discovery is symmetric: orphan checkpoints, orphan eval files,
  byte-duplicate checkpoints bound to more than one run directory, and
  unreadable artifacts are explicit blockers.
* Every retained loadable checkpoint (2B AND live 4B) needs valid
  rescored raw-score eval evidence bound to it (adapter path + model +
  revision + current suite hash) covering BOTH required splits (test_id
  AND test_ood) over the COMPLETE preregistered example set of each
  declared split; a lone labelled record, or a test_id record relabelled
  into a test_ood file, proves nothing globally.
* The rejected 4B batch must be nonempty, markered, and complete: any
  known-invalid or byte-duplicate 4B artifact left in any live tree is a
  blocker; one differing file does not mask other identical live copies;
  a marker-only empty quarantine proves nothing.
* Bounded streaming SHA-256 fingerprints of both input roots are taken
  before and after gating; any change aborts with an execution error.
  ``--out`` may never overlap an input root, so the gate can never
  self-inventory its outputs.

Usage (from the repository root):

    python -m latent_lab.bench.no_spend_gate \
        --results-2b .rcc_work/remote_results \
        --results-4b .rcc_work/remote_results_4b \
        --out .rcc_work/no_spend_gate_20260824

Exit semantics:
    0  READY            every prerequisite PROVEN by this evidence set
    1  NOT_READY        evidence-backed negative verdict; see
                        gate_verdict.json blockers
    2  EXECUTION ERROR  bad invocation (--out overlapping inputs), missing
                        or unreadable inputs, inputs modified during the
                        run, unwritable outputs, crash — no verdict may be
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
from dataclasses import dataclass
from pathlib import Path

from .corrected_scoring import (
    INVALID_RECORDS,
    MISSING_RAW_PREDICTION,
    RESCORED_CORRECTED,
    select_best_checkpoint,
)

GATE_SCHEMA_VERSION = 1
GATE_ID = "no-spend-integrity-gate"
GATE_VARIANT = "20260824"

VERDICT_READY = "READY"
VERDICT_NOT_READY = "NOT_READY"

PREREQ_INVENTORY = "inventory_hashed_complete"
PREREQ_REPORTS = "train_reports_schema_identity_and_pins"
PREREQ_CKPT_STRICT = "checkpoints_strict_loadable_identity_bound_fp32_payload"
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
CKPT_NON_FP32_PAYLOAD = "non-fp32-payload"

REQUIRED_SPLITS = ("test_id", "test_ood")

# Eval status for records whose ex_id is outside the preregistered
# membership of the file's declared split (relabelling cannot manufacture
# coverage).
SPLIT_MEMBERSHIP_VIOLATION = "SPLIT_MEMBERSHIP_VIOLATION"

_PINNED_REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_SUITE_SHA_RE = re.compile(r"\A[0-9a-f]{64}\Z")

REQUIRED_CONFIG_FIELDS = {
    "mode": str,
    "interval": list,
    "k": int,
    "seed": int,
    "steps": int,
}

# Eval statuses that can never support the rescore prerequisite.
EVAL_BAD_STATUSES = frozenset({
    "unreadable", "malformed_json", "invalid_metadata", "suite_mismatch",
    "NO_RECORDS", INVALID_RECORDS, MISSING_RAW_PREDICTION,
    SPLIT_MEMBERSHIP_VIOLATION,
})

BLOCKER_ACTIONS = {
    "INVENTORY_UNREADABLE_FILES":
        "Restore or document the unreadable paths, then rerun the gate.",
    "REPORT_SCHEMA_MISSING_FIELDS":
        "Re-emit train_report.json for each listed run through the current "
        "driver schema (config pins + trainable_precision); do not hand-edit "
        "historical files.",
    "REPORT_SUITE_HASH_MISMATCH":
        "Re-evaluate against the current suite revision so the retained "
        "report pins the recomputed behavioral-v2 digest; never edit the "
        "hash by hand.",
    "REPORT_TRAINABLE_PRECISION_MISSING":
        "Add explicit trainable dtype/precision fields to the train "
        "report schema and regenerate reports on the next supervised smoke.",
    "REPORT_RECIPE_NOT_CANONICAL":
        "Re-emit each listed train_report.json through the current driver "
        "so its recipe equals checkpointing.recipe_from_config(config, "
        "suite_sha256) exactly; never hand-edit or copy a recipe, and "
        "never accept an in-bundle recipe as a substitute.",
    "CKPT_LEGACY_UNBOUND_IDENTITY":
        "Rebuild identity-bound bundles via save_adapter_bundle ONLY after "
        "weight-provenance verification; otherwise retire the weights into "
        "the next capped paired-seed retrain decision.",
    "CKPT_INVALID_IDENTITY":
        "Repair or retire checkpoints whose in-bundle identity/schema "
        "violates the binding requirements; they can never be reused.",
    "ORPHAN_CHECKPOINT":
        "Give every checkpoint file a sibling identity-bound "
        "train_report.json (or remove/quarantine the orphan); unexplained "
        "weights cannot enter the provenance chain.",
    "ORPHAN_EVAL_FILE":
        "Re-emit the eval with an adapter path that resolves to a "
        "discovered run directory, or move the file out of the evidence "
        "tree.",
    "DUPLICATE_CHECKPOINT_BINDING":
        "Keep exactly one canonical copy of each checkpoint digest inside "
        "its owning run directory; delete or quarantine every other "
        "byte-identical copy so the binding is unambiguous.",
    "EVAL_SPLIT_COVERAGE_MISSING":
        "Run the corrected raw-score evaluation for every listed "
        "(run, split) pair before treating the checkpoint as proven.",
    "EVAL_SPLIT_MEMBERSHIP_VIOLATION":
        "Re-run evaluation per split over exactly the preregistered "
        "example set of the declared split; never satisfy coverage by "
        "copying or relabelling records across split-labelled files.",
    "EVAL_IDENTITY_MISMATCH":
        "Regenerate the eval under the exact model id/revision/suite hash "
        "of the checkpoint it claims to score.",
    "EVAL_FILE_INVALID":
        "Repair or re-emit each listed eval file (strict JSON, nonempty "
        "binding metadata, nonempty valid raw-score records); invalid "
        "files cannot be skipped or partially trusted.",
    "CKPT_CORRUPT":
        "Keep quarantined as negative evidence; exclude from any resume or "
        "rescore.",
    "TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS":
        "Fix persistence to cast trainables to fp32 on save and verify on "
        "the next CPU smoke before any GPU spend.",
    "NON_RESCORABLE_MISSING_RAW_PREDICTION":
        "Rerun evaluation with raw per-candidate score capture enabled "
        "(capped CUDA canary scope) or formally invalidate the latent "
        "conclusions; never relabel derived records as a rescore.",
    "SELECTION_PROVENANCE_NOT_CORRECTED":
        "Apply select_best_checkpoint over corrected-metric histories "
        "produced by the next validated run; discard historical best_step "
        "claims.",
    "LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH":
        "After hashed backup + manifest promotion, remove the duplicated "
        "files under the LIVE 4B runs/ tree (deletion is outside this "
        "gate's authority) and keep _rejected_nan_batch as sole evidence.",
    "LIVE_INVALID_4B_ARTIFACTS":
        "Remove or formally quarantine every listed live-tree 4B artifact "
        "(unreadable report, NaN loss, degenerate accuracy, corrupt "
        "payload); live trees must hold only valid artifacts.",
    "QUARANTINE_MARKER_MISSING":
        "Restore/confirm REJECTED.md in the quarantine tree, then rerun.",
    "QUARANTINE_BATCH_EMPTY":
        "A marker-only empty quarantine proves nothing; populate the "
        "rejected batch with its actual artifacts or remove the tree.",
    "PROOF_TESTS_FAILED":
        "Fix the failing regression(s) locally before any spend decision.",
}

NEXT_ACTION_GENERIC = "Review gate_verdict.json prerequisites and rerun."


class GateExecutionError(RuntimeError):
    """Unreadable inputs, mutated inputs, bad invocation — exit code 2."""


# ---------------------------------------------------------------------------
# deterministic strict-JSON helpers
# ---------------------------------------------------------------------------

def _reject_json_constant(constant: str):
    raise ValueError(f"non-strict JSON constant {constant!r}")


def strict_json_loads(text: str):
    """Parse JSON rejecting NaN/Infinity literals entirely."""
    return json.loads(text, parse_constant=_reject_json_constant)


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
# bounded streaming source fingerprints
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint(root: Path) -> dict:
    """Constant-memory Merkle fingerprint over every regular file below
    root (sorted rel paths, streamed 1 MiB chunks). Raises
    GateExecutionError when any file cannot be read."""
    entries: list[tuple[str, str]] = []
    n_bytes = 0
    try:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            st = p.stat()
            entries.append((p.relative_to(root).as_posix(), sha256_file(p)))
            n_bytes += st.st_size
    except OSError as e:
        raise GateExecutionError(
            f"source fingerprint failed under {root}: {e}") from e
    merkle = hashlib.sha256()
    for rel, digest in entries:
        merkle.update(rel.encode("utf-8"))
        merkle.update(b"\0")
        merkle.update(digest.encode("ascii"))
        merkle.update(b"\n")
    return {"merkle_sha256": merkle.hexdigest(),
            "files": len(entries), "bytes": n_bytes}


def paths_overlap(a: Path, b: Path) -> bool:
    """True when either path equals or contains the other (after making
    both absolute without requiring existence)."""
    aa = Path(os.path.abspath(a))
    bb = Path(os.path.abspath(b))
    return aa == bb or aa in bb.parents or bb in aa.parents


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
    report_error: str | None = None
    report_sha256: str | None = None
    ckpt_sha256: str | None = None


def _is_pinned_revision(v) -> bool:
    return isinstance(v, str) and bool(_PINNED_REVISION_RE.fullmatch(v))


def validate_train_report(report: dict) -> dict:
    """Schema validation; every defect is recorded, never defaulted.

    Steps must be non-negative integers (bool/float rejected), all
    required metric/history values must be finite real numbers, identity
    fields must be nonempty strings with pinned revision and 64-hex suite
    digest."""
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
                    and all(isinstance(x, int) and not isinstance(x, bool)
                            for x in iv)):
                problems.append(f"config.{key}:type")
        elif not isinstance(config[key], typ):
            problems.append(f"config.{key}:type")
    steps = config.get("steps")
    if isinstance(steps, int) and not isinstance(steps, bool) and steps < 0:
        problems.append("config.steps:negative")

    acc = report.get("best_val_acc")
    if "best_val_acc" not in report:
        problems.append("best_val_acc:missing")
    elif isinstance(acc, bool) or not isinstance(acc, (int, float)) \
            or not math.isfinite(float(acc)):
        problems.append("best_val_acc:type_or_nonfinite")
    step = report.get("best_step")
    if "best_step" not in report:
        problems.append("best_step:missing")
    elif isinstance(step, bool) or not isinstance(step, int) or step < 0:
        problems.append("best_step:type")

    model = report.get("model")
    if not isinstance(model, str) or not model.strip():
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
    else:
        bad_idx = []
        for i, entry in enumerate(hist):
            ok = isinstance(entry, dict)
            if ok:
                s = entry.get("step")
                m = entry.get("accuracy")
                ok = (isinstance(s, int) and not isinstance(s, bool)
                      and s >= 0
                      and isinstance(m, (int, float))
                      and not isinstance(m, bool)
                      and math.isfinite(float(m)))
            if not ok:
                bad_idx.append(i)
        if bad_idx:
            shown = ",".join(str(i) for i in bad_idx[:8])
            more = "" if len(bad_idx) <= 8 else ",…"
            problems.append(f"val_history:bad_entries[{shown}{more}]")

    loss = report.get("final_train_loss")
    if "final_train_loss" not in report:
        problems.append("final_train_loss:missing")
    elif isinstance(loss, bool) or not isinstance(loss, (int, float)) \
            or not math.isfinite(float(loss)):
        problems.append("final_train_loss:type_or_nonfinite")
    prec = report.get("trainable_precision")
    if "trainable_precision" not in report:
        problems.append("trainable_precision:missing")
    elif not isinstance(prec, str) or not prec.strip():
        problems.append("trainable_precision:type")

    # Canonical recipe: derived ONLY from this validated config plus its
    # separately validated 64-hex suite digest; the report's own
    # "recipe" field must equal the derivation. Never trusted verbatim.
    declared_recipe = report.get("recipe")
    if declared_recipe is None:
        problems.append("recipe:missing")
    elif not isinstance(declared_recipe, dict):
        problems.append("recipe:type")
    derived_recipe = None
    if isinstance(suite, str) and _SUITE_SHA_RE.fullmatch(suite):
        from ..train.checkpointing import AdapterBundleSchemaError, \
            recipe_from_config

        try:
            derived_recipe = recipe_from_config(config, suite)
        except AdapterBundleSchemaError:
            problems.append(
                "recipe:not_derivable_from_validated_config_and_suite")
        else:
            if isinstance(declared_recipe, dict) \
                    and declared_recipe != derived_recipe:
                problems.append("recipe:mismatch_with_derived_canonical")

    selection = None
    if isinstance(hist, list) and hist:
        sel = select_best_checkpoint(hist)
        reported_acc = report.get("best_val_acc")
        reported_step = report.get("best_step")
        if sel is not None and isinstance(reported_acc, (int, float)) \
                and not isinstance(reported_acc, bool) \
                and math.isfinite(float(reported_acc)) \
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
            "model_id": model if isinstance(model, str) else None,
            "revision": rev if _is_pinned_revision(rev) else None,
            "suite_sha256": suite if isinstance(suite, str) else None,
            # The canonical recipe ONLY when it was derived from the
            # validated config + validated suite hash AND the report's
            # declared recipe equals it; otherwise None (fail closed).
            "recipe": derived_recipe if (
                derived_recipe is not None
                and isinstance(declared_recipe, dict)
                and declared_recipe == derived_recipe
                and not any(p.startswith("recipe:") for p in problems))
            else None,
        },
        "selection_check": selection,
    }

# Regression proofs the gate executes (hardware-free, CPU-only) unless
# skipped. These pin the runtime-integrity prerequisites.
PROOF_TEST_NODES = (
    # Node ids track the CURRENT integrated runtime suite; stale ids would
    # make pytest exit with rc=4 and the prerequisite forever unprovable.
    "tests/test_latent_runtime_integrity.py::test_bundle_v2_roundtrip_digest_and_persisted_metrics",
    "tests/test_latent_runtime_integrity.py::test_bundle_identity_mismatch_rejected",
    "tests/test_latent_runtime_integrity.py::test_lora_and_clock_trainables_are_fp32_over_bf16_backbone",
    "tests/test_latent_runtime_integrity.py::test_recurrence_clock_is_fp32_regardless_of_default_dtype",
    "tests/test_latent_runtime_integrity.py::test_nan_metric_and_bad_step_schema_rejected",
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

# ---------------------------------------------------------------------------
# checkpoint classification
# ---------------------------------------------------------------------------

def _load_raw_payload(path: Path):
    """Safe deserialize: weights_only=True, project loader afterwards."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def classify_checkpoint(path: Path, *, report_identity: dict | None,
                        report_recipe: dict | None, dry_run: bool) -> dict:
    """Classify one .pt payload. Read-only; never mutates anything.

    ``report_recipe`` is the CANONICAL recipe derived from a validated
    train report (config + separately validated suite hash); bundles are
    strictly loaded against it. Without it — orphan bundles, or reports
    whose canonical recipe cannot be derived/verified — classification
    fails closed: an identity-bound bundle is never loadable on the
    strength of its own in-bundle recipe.
    """
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
        return _classify_bundle(raw, path, report_identity, report_recipe,
                                info)
    return _classify_plain_state(raw, info)


def _classify_bundle(raw: dict, path: Path, report_identity,
                     report_recipe: dict | None, info: dict) -> dict:
    from ..train.checkpointing import (
        AdapterBundleIdentityError,
        AdapterBundleSchemaError,
        CheckpointError,
        load_adapter_bundle,
    )

    bid, brev = raw.get("model_id"), raw.get("revision")
    info["bundle_identity"] = {"model_id": bid, "revision": brev}
    if not isinstance(bid, str) or not bid.strip() \
            or not _is_pinned_revision(brev):
        info["classification"] = CKPT_INVALID
        info["reasons"].append("bundle_identity_not_pinable")
        return info
    if report_identity:
        rep_mid = report_identity.get("model_id")
        rep_rev = report_identity.get("revision")
        if (rep_mid and str(bid) != str(rep_mid)) \
                or (rep_rev and str(brev) != str(rep_rev)):
            info["classification"] = CKPT_INVALID
            info["reasons"].append("bundle_identity_conflicts_with_report")
            return info
    if report_recipe is None:
        # FAIL CLOSED: no independently derived canonical recipe exists
        # (orphan bundle, or sidecar report failed recipe derivation/
        # verification). The raw in-bundle recipe can never substitute.
        info["classification"] = CKPT_INVALID
        info["reasons"].append(
            "bundle_unverifiable_without_derived_report_recipe")
        return info
    try:
        # Strict project loader: validates keys, declared shapes/dtypes,
        # finiteness AND identity (model_id/revision/recipe — the recipe
        # compared is the INDEPENDENTLY DERIVED canonical one, never the
        # raw bundle payload) before returning tensor clones — the
        # returned payload is the ground truth for the fp32 check below.
        tensors = load_adapter_bundle(path, model_id=str(bid),
                                      revision=str(brev),
                                      recipe=report_recipe)
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

    # Validate ACTUAL trainable dtypes from the bundle payload — a report
    # string claiming fp32 proves nothing.
    info["tensor_dtypes"] = sorted({str(t.dtype) for t in tensors.values()})
    info["n_tensors"] = len(tensors)
    bad_dtypes = sorted({str(t.dtype) for t in tensors.values()
                         if t.is_floating_point()
                         and str(t.dtype) != "torch.float32"})
    if bad_dtypes:
        info["classification"] = CKPT_NON_FP32_PAYLOAD
        info["reasons"].append(
            "bundle_payload_not_fp32:" + ",".join(bad_dtypes))
        return info
    info["classification"] = CKPT_LOADABLE
    info["reasons"].append("strict_project_loader_passed")
    info["reasons"].append("payload_floating_tensors_all_fp32")
    if report_identity:
        info["reasons"].append("identity_matches_sidecar_report")
    if report_recipe is not None:
        info["reasons"].append(
            "bundle_recipe_equals_independently_derived_report_recipe")
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
# eval rescoring eligibility + identity binding
# ---------------------------------------------------------------------------

def evaluate_eval_file(path: Path, examples_by_id: dict | None,
                       *, root_label: str) -> dict:
    out: dict = {
        "file": path.name,
        "root": root_label,
        "adapter": None,
        "split": None,
        "model": None,
        "revision": None,
        "suite_sha256_matches_current_suite": None,
        "bound_run": None,
        "record_ex_ids": None,
        "n_records": 0,
    }
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        out["status"] = "unreadable"
        out["detail"] = str(e)
        return out
    try:
        data = strict_json_loads(text)
    except ValueError as e:
        out["status"] = "malformed_json"
        out["detail"] = f"strict JSON parse failed: {e}"
        return out
    if not isinstance(data, dict):
        out["status"] = "malformed_json"
        out["detail"] = "top level not an object"
        return out

    for field in ("adapter", "split", "model", "revision", "suite_sha256"):
        val = data.get(field)
        if not isinstance(val, str) or not val.strip():
            out["status"] = "invalid_metadata"
            out["detail"] = f"{field}: missing or not a nonempty string"
            return out
        out[field] = val

    matches = data["suite_sha256"] == current_suite_sha256()
    out["suite_sha256_matches_current_suite"] = matches
    if not matches:
        out["status"] = "suite_mismatch"
        out["detail"] = (
            f"declared suite {data['suite_sha256'][:12]}… != current "
            f"{current_suite_sha256()[:12]}…")
        return out

    records: list = []
    results_obj = data.get("results")
    if not isinstance(results_obj, dict) or not results_obj:
        out["status"] = "invalid_metadata"
        out["detail"] = "results: missing/empty or not an object"
        return out
    for key, res in results_obj.items():
        if not isinstance(res, dict) or not isinstance(res.get("records"),
                                                       list):
            out["status"] = "invalid_metadata"
            out["detail"] = f"results.{key}: records list missing/malformed"
            return out
        records.extend(res["records"])
    out["n_records"] = len(records)
    out["record_ex_ids"] = sorted({str(r.get("ex_id")) for r in records
                                   if isinstance(r, dict)})

    # Relational identity check: every record in a required-split file must
    # reference an example that canonically BELONGS to the preregistered
    # membership of the DECLARED split; copying or relabelling records
    # across split-labelled files can never manufacture coverage.
    if data["split"] in REQUIRED_SPLITS:
        expected = suite_examples_by_split()[data["split"]]
        foreign = [i for i in out["record_ex_ids"] if i not in expected]
        if foreign:
            shown = ", ".join(foreign[:8]) + \
                ("…" if len(foreign) > 8 else "")
            out["status"] = SPLIT_MEMBERSHIP_VIOLATION
            out["detail"] = (
                f"{len(foreign)} record ex_id(s) outside the preregistered "
                f"{data['split']} membership "
                f"({len(out['record_ex_ids'])} distinct ids present): "
                f"{shown}")
            return out

    if examples_by_id is None:
        examples_by_id = suite_examples_by_id()

    from .corrected_scoring import rescore_records

    outcome = rescore_records(records, examples_by_id)
    out["status"] = outcome.status
    if outcome.status == RESCORED_CORRECTED:
        out["corrected_accuracy"] = outcome.corrected_accuracy
    if outcome.detail:
        out["detail"] = outcome.detail
    if outcome.flag_counts:
        out["flag_counts"] = outcome.flag_counts
    return out


_SUITE_CACHE: dict | None = None
_SUITE_SHA_CACHE: str | None = None
_SUITE_SPLIT_CACHE: dict[str, frozenset[str]] | None = None


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


def suite_examples_by_split() -> dict[str, frozenset[str]]:
    """split -> preregistered ex_id membership, built once per process."""
    global _SUITE_SPLIT_CACHE
    if _SUITE_SPLIT_CACHE is None:
        from ..bench.suite import build_suite

        suite = build_suite()
        _SUITE_SPLIT_CACHE = {
            name: frozenset(ex.ex_id for ex in exs)
            for name, exs in suite.splits().items()
        }
    return _SUITE_SPLIT_CACHE


def current_suite_sha256() -> str:
    global _SUITE_SHA_CACHE
    if _SUITE_SHA_CACHE is None:
        from ..bench.suite import build_suite

        _SUITE_SHA_CACHE = build_suite().manifest()["sha256"]
    return _SUITE_SHA_CACHE


# ---------------------------------------------------------------------------
# 4B quarantine analysis
# ---------------------------------------------------------------------------

def _parse_report_lenient(path: Path):
    """Tolerant parse used ONLY to detect pathology (a NaN literal must be
    visible to the detector); any failure returns None."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def analyze_quarantine(scanned: list[ScannedFile]) -> dict:
    def files_under(prefix: str) -> dict[str, ScannedFile]:
        return {
            f.rel[len(prefix):]: f
            for f in scanned
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
    # ANY byte-identical quarantined file still present live is a blocker;
    # a differing sibling must not mask the remaining identical copies.
    live_dupes_quarantine = bool(identical)

    nan_runs, fake_best_runs, unreadable_runs = [], [], []
    for rel, f in sorted(rejected.items()):
        if f.kind != "train_report":
            continue
        rep = _parse_report_lenient(f.abs)
        run = rel.split("/")[0]
        if not isinstance(rep, dict):
            unreadable_runs.append(run)
            continue
        loss = rep.get("final_train_loss")
        acc = rep.get("best_val_acc")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool) \
                and not math.isfinite(float(loss)):
            nan_runs.append(run)
        if isinstance(acc, (int, float)) and not isinstance(acc, bool) \
                and math.isfinite(float(acc)) and float(acc) >= 1.0:
            fake_best_runs.append(run)

    # Known-invalid artifacts remaining LIVE are blockers too.
    live_invalid_runs = []
    for rel, f in sorted(live.items()):
        if f.kind != "train_report":
            continue
        rep = _parse_report_lenient(f.abs)
        run = rel.split("/")[0]
        if not isinstance(rep, dict):
            live_invalid_runs.append(run)
            continue
        loss = rep.get("final_train_loss")
        acc = rep.get("best_val_acc")
        invalid = False
        if isinstance(loss, (int, float)) and not isinstance(loss, bool) \
                and not math.isfinite(float(loss)):
            invalid = True
        if isinstance(acc, (int, float)) and not isinstance(acc, bool) \
                and math.isfinite(float(acc)) and float(acc) >= 1.0:
            invalid = True
        if invalid:
            live_invalid_runs.append(run)

    rejected_reports = sorted({r.split("/")[0] for r in rejected
                               if r.endswith("/train_report.json")})
    marker_only_empty = marker is not None and not rejected_reports \
        and len(rejected) <= 1
    return {
        "live_run_dirs": sorted({r.split("/")[0] for r in live
                                 if r.endswith("/train_report.json")}),
        "rejected_run_dirs": rejected_reports,
        "quarantine_marker_present": marker is not None,
        "live_vs_rejected_identical_files": identical,
        "live_vs_rejected_differing_files": differing,
        "only_in_live": only_live,
        "only_in_rejected": only_rej,
        "live_tree_duplicates_quarantine": live_dupes_quarantine,
        "rejected_batch_empty": not rejected_reports,
        "marker_only_empty_quarantine": marker_only_empty,
        "rejected_runs_with_nan_final_loss": sorted(set(nan_runs)),
        "rejected_runs_with_degenerate_best_acc_1": sorted(set(fake_best_runs)),
        "rejected_runs_unreadable_report": sorted(set(unreadable_runs)),
        "live_invalid_runs": sorted(set(live_invalid_runs)),
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


def _norm_adapter(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().strip("/")


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
    fingerprints_before = {
        "results_2b": source_fingerprint(results_2b),
        "results_4b": source_fingerprint(results_4b),
    }

    errors: list[str] = []
    scanned = scan_roots(results_2b, results_4b, errors=errors)
    if errors:
        # Unreadable inputs mean no trustworthy verdict is possible at all.
        raise GateExecutionError(
            "input roots unreadable/incomplete: " + "; ".join(sorted(errors)))
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
                sf = files_by_id[rep_rel]
                ri.report_sha256 = sf.sha256
                try:
                    parsed = strict_json_loads(
                        sf.abs.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, ValueError) as e:
                    ri.report = None
                    ri.report_error = \
                        f"train_report unreadable/malformed: {e}"
                else:
                    if isinstance(parsed, dict):
                        ri.report = parsed
                    else:
                        ri.report = None
                        ri.report_error = (
                            "train_report malformed: top level not an object")
            if ckpt_rel:
                ri.ckpt_sha256 = files_by_id[ckpt_rel].sha256
            runs[label].append(ri)

    # ---- symmetric checkpoint bookkeeping ---------------------------------
    # RunInfo.checkpoint_rel already carries the "<root>/<path>" artifact id.
    claimed_ckpt_ids = {ri.checkpoint_rel
                        for lbl in ("2b", "4b") for ri in runs[lbl]
                        if ri.checkpoint_rel}
    orphan_ckpts = [f for f in scanned
                    if f.kind == "adapter_checkpoint"
                    and f"{f.label}/{f.rel}" not in claimed_ckpt_ids]

    # Byte-identical checkpoint payloads owned by more than one run
    # directory make checkpoint->report binding ambiguous.
    owner_of: dict[str, tuple[str, str]] = {}
    for lbl in ("2b", "4b"):
        for ri in runs[lbl]:
            if ri.checkpoint_rel:
                owner_of[ri.checkpoint_rel] = (ri.root, ri.dir_rel)
    ckpt_hash_groups: dict[str, list[str]] = {}
    for f in scanned:
        if f.kind == "adapter_checkpoint":
            ckpt_hash_groups.setdefault(f.sha256, []).append(
                f"{f.label}/{f.rel}")
    dup_binding_groups: list[dict] = []
    for digest, members in sorted(ckpt_hash_groups.items()):
        owners = sorted({owner_of.get(m, ("<orphan>", "<orphan>"))
                         for m in members})
        if len(members) > 1 and len(owners) > 1:
            dup_binding_groups.append({
                "sha256": digest,
                "members": sorted(members),
                "owners": [f"{o[0]}/{o[1]}" for o in owners]})

    # ---- per-run validation + classification -----------------------------
    run_verdicts = []
    for lbl in ("2b", "4b"):
        for ri in runs[lbl]:
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
                rv["report_problems"] = (
                    ["train_report:malformed_json"]
                    if ri.report_error else ["train_report:missing"])
                if ri.report_error:
                    rv["report_error"] = ri.report_error
            if ri.checkpoint_rel:
                cls = classify_checkpoint(
                    files_by_id[ri.checkpoint_rel].abs,
                    report_identity=identity,
                    report_recipe=(identity or {}).get("recipe"),
                    dry_run=dry_run)
                rv["checkpoint"] = cls
                rv["checkpoint_sha256"] = ri.ckpt_sha256
            run_verdicts.append(rv)

    orphan_verdicts: list[dict] = []
    if not dry_run:
        for f in orphan_ckpts:
            # Orphan bundles have no owning report: no canonical recipe can
            # be derived, so classification must fail closed (never
            # loadable on the bundle's own claims).
            cls = classify_checkpoint(f.abs, report_identity=None,
                                      report_recipe=None, dry_run=False)
            orphan_verdicts.append({"id": f"{f.label}/{f.rel}",
                                    "classification": cls["classification"],
                                    "reasons": cls["reasons"]})
    else:
        orphan_verdicts = [{"id": f"{f.label}/{f.rel}",
                            "classification": CKPT_UNPROVEN,
                            "reasons": ["dry_run_skips_payload_loading"]}
                           for f in orphan_ckpts]

    # ---- eval rescoring ----------------------------------------------------
    eval_verdicts = []
    if not dry_run:
        for f in scanned:
            if f.kind == "eval_json":
                eval_verdicts.append(evaluate_eval_file(
                    f.abs, None, root_label=f.label))
    eval_verdicts.sort(key=lambda e: (e["root"], e["file"]))

    # Bind every eval to a discovered run via its adapter path; unbound
    # evals prove nothing and are blockers (symmetric discovery).
    runs_by_adapter = {(rv["root"], _norm_adapter(rv["dir"])): rv
                       for rv in run_verdicts}
    for ev in eval_verdicts:
        target = runs_by_adapter.get(
            (ev["root"], _norm_adapter(ev.get("adapter"))))
        ev["bound_run"] = (f"{target['root']}/{target['dir']}"
                           if target else None)

    # ---- 4B quarantine -----------------------------------------------------
    quarantine = analyze_quarantine(scanned) if results_4b.is_dir() else {}

    # ---- proof tests ---------------------------------------------------------
    proof = proof_result
    if proof is None and not dry_run and not skip_proof_tests:
        proof = run_proof_tests(repo_root, log_path=proof_log_path)

    # ---- inputs unchanged? ----------------------------------------------------
    fingerprints_after = {
        "results_2b": source_fingerprint(results_2b),
        "results_4b": source_fingerprint(results_4b),
    }
    inputs_unchanged = fingerprints_before == fingerprints_after
    if not inputs_unchanged:
        raise GateExecutionError(
            "input trees changed while the gate was running; no verdict is "
            "trustworthy")

    # ---- prerequisites -------------------------------------------------------
    two_b_runs = [rv for rv in run_verdicts if rv["root"] == "2b"]
    # Retained candidates = every non-rejected run carrying a checkpoint
    # payload (the retained 2B tree AND live 4B trees); quarantined negative
    # evidence stays classified/inventoried without gating strict-load.
    candidate_ckpt_rvs = [rv for rv in run_verdicts
                          if rv.get("checkpoint")
                          and rv["scope"] == "retained_candidate"]
    # Report-schema/selection prerequisites apply to EVERY retained run,
    # including live 4B runs without checkpoint payloads.
    retained_runs = [rv for rv in run_verdicts
                     if rv["scope"] == "retained_candidate"]
    rejected_ckpt_rvs = [rv for rv in run_verdicts
                         if rv.get("checkpoint")
                         and rv["scope"] == "rejected_negative"]
    ckpt_infos = [rv["checkpoint"] for rv in candidate_ckpt_rvs]
    n_ckpts = len(ckpt_infos)
    n_legacy = sum(c.get("classification") == CKPT_LEGACY_UNBOUND
                   for c in ckpt_infos)
    n_corrupt = sum(c.get("classification") == CKPT_CORRUPT
                    for c in ckpt_infos)
    n_invalid = sum(c.get("classification") == CKPT_INVALID
                    for c in ckpt_infos)
    n_loadable = sum(c.get("classification") == CKPT_LOADABLE
                     for c in ckpt_infos)
    n_nonfp32 = sum(c.get("classification") == CKPT_NON_FP32_PAYLOAD
                    for c in ckpt_infos)
    n_bf16 = n_nonfp32 + sum(
        any(str(r).startswith("non_fp32_trainables_stored")
            for r in c.get("reasons", []))
        for c in ckpt_infos)

    prereqs: list[dict] = []

    def prereq(pid, status, detail):
        prereqs.append({"id": pid, "status": status, "detail": detail})

    prereq(PREREQ_INVENTORY, STATUS_PROVEN,
           f"{len(scanned)} files hashed; streaming input fingerprints "
           f"taken before and after the gate and matched")

    schema_failures = [rv for rv in retained_runs
                       if rv.get("report_problems")]
    precision_missing = [rv for rv in retained_runs
                         if "trainable_precision:missing"
                         in rv.get("report_problems", [])]
    suite_mismatch = [rv for rv in retained_runs
                      if rv.get("suite_sha256_matches_current_suite") is False]
    reports_ok = not schema_failures and not suite_mismatch
    prereq(PREREQ_REPORTS,
           STATUS_PROVEN if reports_ok else STATUS_FAILED,
            (f"{len(retained_runs)}/{len(retained_runs)} retained (2B + live "
             f"4B) reports fully schema-valid with pinned revision + finite "
             f"metrics + suite hash matching the recomputed behavioral-v2 "
             f"suite + recipe equal to the canonically derived "
             f"recipe_from_config(config, suite_sha256)"
            if reports_ok else
            f"{len(schema_failures)} retained reports carry schema "
            f"violations (missing fields are blockers): "
            + "; ".join(
                f"{rv['root']}/{rv['dir']}: {','.join(rv['report_problems'])}"
                for rv in sorted(retained_runs,
                                 key=lambda r: (r["root"], r["dir"]))
                if rv.get("report_problems"))))

    ckpt_strict_ok = (not dry_run and n_ckpts > 0 and n_loadable == n_ckpts
                      and not orphan_ckpts and not dup_binding_groups)
    prereq(PREREQ_CKPT_STRICT,
           STATUS_PROVEN if ckpt_strict_ok else STATUS_FAILED,
           ("dry run skips payload loading; classification unproven"
            if dry_run else
            f"{n_loadable}/{n_ckpts} checkpoints strictly loadable with "
            f"in-bundle identity matching the INDEPENDENTLY derived report "
            f"recipe and VERIFIED fp32 payload tensors; "
            f"{n_legacy} legacy-unbound, {n_corrupt} corrupt, {n_invalid} "
            f"invalid, {n_nonfp32} non-fp32-payload, "
            f"{len(orphan_ckpts)} orphan, {len(dup_binding_groups)} "
            f"ambiguous duplicate bindings"))

    bad_evals = [e for e in eval_verdicts
                 if e.get("status") in EVAL_BAD_STATUSES]
    membership_violations = [e for e in eval_verdicts
                             if e.get("status")
                             == SPLIT_MEMBERSHIP_VIOLATION]
    rescored = [e for e in eval_verdicts
                if e.get("status") == RESCORED_CORRECTED]
    orphan_evals = [e for e in eval_verdicts if e.get("bound_run") is None]

    # Retained candidates in EITHER root with strictly loadable checkpoints
    # must be proven by eval evidence — live 4B checkpoints carry the same
    # prerequisites as retained 2B checkpoints.
    eligible_rvs = [rv for rv in candidate_ckpt_rvs
                    if rv.get("checkpoint", {}).get("classification")
                    == CKPT_LOADABLE]
    expected_by_split = {s: suite_examples_by_split()[s]
                         for s in REQUIRED_SPLITS}
    identity_mismatches: list[str] = []
    coverage_gaps: list[str] = []
    covered_runs: list[str] = []
    if not dry_run:
        for rv in eligible_rvs:
            rid = f"{rv['root']}/{rv['dir']}"
            ident = rv.get("identity") or {}
            covered_ids: dict[str, set] = {s: set() for s in REQUIRED_SPLITS}
            for ev in eval_verdicts:
                if ev.get("bound_run") != rid:
                    continue
                if ev.get("status") != RESCORED_CORRECTED:
                    continue
                if ev.get("split") not in covered_ids:
                    continue
                if ev.get("model") != ident.get("model_id") \
                        or ev.get("revision") != ident.get("revision") \
                        or ev.get("suite_sha256_matches_current_suite") \
                        is not True:
                    identity_mismatches.append(f"{ev['file']} vs {rid}")
                    continue
                covered_ids[ev["split"]].update(ev.get("record_ex_ids") or ())
            gaps = []
            for s in REQUIRED_SPLITS:
                missing_ids = sorted(expected_by_split[s] - covered_ids[s])
                n_expected = len(expected_by_split[s])
                if missing_ids:
                    shown = ", ".join(missing_ids[:5]) + \
                        ("…" if len(missing_ids) > 5 else "")
                    gaps.append(
                        f"{s} missing {len(missing_ids)}/{n_expected} "
                        f"preregistered examples ({shown})")
            if gaps:
                coverage_gaps.append(f"{rid}: " + "; ".join(gaps))
            else:
                covered_runs.append(rid)

    if dry_run:
        rescore_status, rescore_detail = STATUS_UNPROVEN, \
            "eval payload inspection skipped (dry run)"
    elif (bad_evals or orphan_evals or identity_mismatches or coverage_gaps
            or not eligible_rvs or not eval_verdicts):
        rescore_status = STATUS_FAILED
        parts = [
            f"{len(rescored)}/{len(eval_verdicts)} eval files rescored",
            f"{len(bad_evals)} invalid/unrescorable",
            f"{len(orphan_evals)} unbound",
            f"{len(identity_mismatches)} identity mismatches",
            f"{len(membership_violations)} split-membership violations",
            f"{len(coverage_gaps)} runs with incomplete preregistered "
            f"split coverage",
        ]
        rescore_detail = "retained-eval evidence incomplete: " \
            + "; ".join(parts)
    else:
        rescore_status = STATUS_PROVEN
        rescore_detail = (f"{len(rescored)} eval files rescored; all "
                          f"{len(eligible_rvs)} loadable retained checkpoints "
                          f"covered on {'+'.join(REQUIRED_SPLITS)} with "
                          f"exact identity binding")
    prereq(PREREQ_RESCORE, rescore_status, rescore_detail)

    selection_ok = all(
        (rv.get("selection_check") or {}).get("consistent_with_reported")
        for rv in retained_runs if rv.get("selection_check"))
    corrected_histories = bool(retained_runs) and all(
        rv.get("scorer_tag") == corrected_scorer_tag
        for rv in retained_runs)
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
        not quarantine.get("rejected_batch_empty") and \
        not quarantine.get("marker_only_empty_quarantine") and \
        not quarantine.get("live_tree_duplicates_quarantine") and \
        not quarantine.get("only_in_live") and \
        not quarantine.get("live_invalid_runs")
    if quarantine_ok:
        qdetail = ("rejected NaN batch nonempty + markered + fully "
                   "contained; live trees hold no duplicate or invalid 4B "
                   "artifacts")
    elif not quarantine:
        qdetail = "no 4B root scanned"
    else:
        qdetail = (f"marker_present="
                   f"{quarantine.get('quarantine_marker_present')}, "
                   f"batch_empty={quarantine.get('rejected_batch_empty')}, "
                   f"marker_only_empty="
                   f"{quarantine.get('marker_only_empty_quarantine')}, "
                   f"live_duplicates_quarantine="
                   f"{quarantine.get('live_tree_duplicates_quarantine')}, "
                   f"only_in_live={len(quarantine.get('only_in_live', []))}, "
                   f"live_invalid={quarantine.get('live_invalid_runs', [])}")
    prereq(PREREQ_QUARANTINE,
           STATUS_PROVEN if quarantine_ok else STATUS_FAILED, qdetail)

    status_of = {p["id"]: p["status"] for p in prereqs}

    # ---- blockers --------------------------------------------------------------
    blockers: list[dict] = []

    def blocker(code, detail):
        blockers.append({"code": code, "detail": detail,
                         "smallest_next_action":
                             BLOCKER_ACTIONS.get(code, NEXT_ACTION_GENERIC)})

    def _rv_path(rv) -> str:
        return f"{rv['root']}/{rv['dir']}"

    if precision_missing:
        blocker("REPORT_TRAINABLE_PRECISION_MISSING",
                f"{len(precision_missing)}/{len(retained_runs)} retained "
                f"reports lack explicit trainable-precision fields: "
                + ", ".join(_rv_path(rv)
                            for rv in sorted(precision_missing,
                                             key=_rv_path)))
    if schema_failures:
        blocker("REPORT_SCHEMA_MISSING_FIELDS",
                f"{len(schema_failures)}/{len(retained_runs)} retained "
                f"reports violate the required train-report schema")
    recipe_failures = [rv for rv in retained_runs
                       if any(str(p).startswith("recipe:")
                              for p in rv.get("report_problems", []))]
    if recipe_failures:
        blocker("REPORT_RECIPE_NOT_CANONICAL",
                f"{len(recipe_failures)}/{len(retained_runs)} retained "
                f"reports carry no verifiable canonical recipe (missing, "
                f"not derivable from the validated config + suite hash, or "
                f"differing from that derivation): "
                + "; ".join(
                    f"{_rv_path(rv)}: "
                    f"{','.join(p for p in rv['report_problems']
                                if str(p).startswith('recipe:'))}"
                    for rv in sorted(recipe_failures,
                                     key=_rv_path)))
    if suite_mismatch:
        blocker("REPORT_SUITE_HASH_MISMATCH",
                f"{len(suite_mismatch)}/{len(retained_runs)} retained "
                f"reports pin a suite hash different from the current "
                f"recomputed suite")
    if n_bf16:
        blocker("TRAINABLES_NOT_FP32_IN_RETAINED_CHECKPOINTS",
                f"{n_bf16}/{n_ckpts} retained checkpoints store floating "
                f"trainables in non-fp32 dtypes (validated from bundle "
                f"payloads, not report strings)")
    if n_legacy:
        blocker("CKPT_LEGACY_UNBOUND_IDENTITY",
                f"{n_legacy}/{n_ckpts} checkpoints are plain state dicts "
                f"without in-bundle model_id/revision; strict identity-bound "
                f"loading is impossible")
    if n_corrupt:
        blocker("CKPT_CORRUPT",
                f"{n_corrupt}/{n_ckpts} checkpoints are corrupt/non-finite")
    if n_invalid:
        blocker("CKPT_INVALID_IDENTITY",
                f"{n_invalid}/{n_ckpts} checkpoints violate bundle identity/"
                f"schema requirements")
    if orphan_ckpts:
        blocker("ORPHAN_CHECKPOINT",
                f"{len(orphan_ckpts)} checkpoint file(s) have no owning "
                f"discovered run directory: "
                + ", ".join(f"{f.label}/{f.rel}"
                            for f in sorted(orphan_ckpts,
                                            key=lambda x: (x.label, x.rel))))
    if dup_binding_groups:
        blocker("DUPLICATE_CHECKPOINT_BINDING",
                f"{len(dup_binding_groups)} byte-identical checkpoint group(s) "
                f"span multiple run directories (ambiguous binding): "
                + "; ".join(g["members"][0] + "+" +
                            str(len(g["members"]) - 1) + " more"
                            for g in dup_binding_groups[:5]))
    if bad_evals:
        by_status: dict[str, int] = {}
        for e in bad_evals:
            by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        listing = "; ".join(f"{e['file']}:{e['status']}"
                            for e in bad_evals[:8])
        blocker("EVAL_FILE_INVALID",
                f"{len(bad_evals)}/{len(eval_verdicts)} eval files cannot "
                f"support the rescore prerequisite ({summary}): {listing}")
    non_rescorable = [e for e in eval_verdicts
                      if e.get("status") == MISSING_RAW_PREDICTION]
    if non_rescorable:
        listing = "; ".join(f"{e['file']}" for e in non_rescorable[:8])
        blocker(MISSING_RAW_PREDICTION,
                f"{len(non_rescorable)}/{len(eval_verdicts)} eval files "
                f"retain only derived correct/rank_of_gold (or reference "
                f"unknown examples) — corrected rescoring impossible "
                f"offline: {listing}")
    if orphan_evals:
        blocker("ORPHAN_EVAL_FILE",
                f"{len(orphan_evals)} eval file(s) carry an adapter path that "
                f"resolves to no discovered run directory: "
                + ", ".join(e["file"] for e in orphan_evals[:8]))
    if identity_mismatches:
        blocker("EVAL_IDENTITY_MISMATCH",
                f"{len(identity_mismatches)} eval file(s) bound to a run but "
                f"scored under a different model/revision/suite hash: "
                + "; ".join(identity_mismatches[:8]))
    if membership_violations:
        listing = "; ".join(
            f"{e['file']} declares {e['split']}"
            for e in membership_violations[:8])
        blocker("EVAL_SPLIT_MEMBERSHIP_VIOLATION",
                f"{len(membership_violations)}/{len(eval_verdicts)} eval "
                f"file(s) contain records whose ex_id is outside the "
                f"preregistered membership of their declared split "
                f"(relabelling cannot manufacture coverage): {listing}")
    if coverage_gaps:
        blocker("EVAL_SPLIT_COVERAGE_MISSING",
                f"{len(coverage_gaps)}/{len(eligible_rvs)} loadable retained "
                f"checkpoints lack valid rescored evidence on required splits: "
                + "; ".join(coverage_gaps[:8]))
    if status_of[PREREQ_RESCORE] != STATUS_PROVEN and not bad_evals \
            and not orphan_evals and not identity_mismatches \
            and not coverage_gaps and not dry_run:
        blocker(MISSING_RAW_PREDICTION,
                "no loadable fp32 retained checkpoint has complete "
                "corrected raw eval evidence; nothing is proven by this "
                "evidence set")
    if status_of[PREREQ_SELECTION] != STATUS_PROVEN and not dry_run:
        blocker("SELECTION_PROVENANCE_NOT_CORRECTED",
                "reported best_step provenance does not satisfy the "
                "corrected-metric selection prerequisite for this evidence")
    if quarantine.get("live_tree_duplicates_quarantine"):
        blocker("LIVE_4B_DUPLICATES_REJECTED_NAN_BATCH",
                f"LIVE 4B runs/ tree holds {len(quarantine['live_vs_rejected_identical_files'])} "
                f"byte-duplicate file(s) from the rejected NaN batch "
                f"({', '.join(quarantine['live_vs_rejected_identical_files'][:6])}"
                f"{'…' if len(quarantine['live_vs_rejected_identical_files']) > 6 else ''})")
    if quarantine.get("live_invalid_runs"):
        blocker("LIVE_INVALID_4B_ARTIFACTS",
                f"known-invalid artifacts remain LIVE in the 4B tree: "
                f"{quarantine.get('live_invalid_runs')} "
                f"(NaN final loss / degenerate accuracy >= 1 / unreadable)")
    if not quarantine.get("quarantine_marker_present") and quarantine:
        blocker("QUARANTINE_MARKER_MISSING", "REJECTED.md absent")
    if quarantine.get("rejected_batch_empty") or \
            quarantine.get("marker_only_empty_quarantine"):
        blocker("QUARANTINE_BATCH_EMPTY",
                "quarantine tree carries no rejected train_report artifacts "
                "(marker-only empty quarantine proves nothing)")
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
        f"({sum(f.size for f in scanned)} bytes); bounded streaming "
        f"fingerprints before/after match: "
        f"{fingerprints_before['results_2b']['merkle_sha256'][:12]}… / "
        f"{fingerprints_before['results_4b']['merkle_sha256'][:12]}…",
        f"{sum(rv.get('suite_sha256_matches_current_suite') is True for rv in two_b_runs)}"
        f"/{len(two_b_runs)} 2B reports pin the locally recomputed "
        f"behavioral-v2 suite digest",
        f"{sum((rv.get('selection_check') or {}).get('consistent_with_reported') is True for rv in two_b_runs)}"
        f"/{len(two_b_runs)} reported best_step values reproduce exactly "
        f"from their own val_history",
        finiteness_line,
    ]

    counts = {
        "files_scanned": len(scanned),
        "bytes_scanned": sum(f.size for f in scanned),
        "runs_2b": len(two_b_runs),
        "retained_candidate_runs": len(retained_runs),
        "runs_4b_live": len(quarantine.get("live_run_dirs", [])),
        "runs_4b_rejected": len(quarantine.get("rejected_run_dirs", [])),
        "checkpoints_total": n_ckpts + len(rejected_ckpt_rvs),
        "checkpoints_rejected_negative": len(rejected_ckpt_rvs),
        "checkpoints_loadable": n_loadable,
        "checkpoints_legacy_unbound": n_legacy,
        "checkpoints_corrupt": n_corrupt,
        "checkpoints_invalid": n_invalid,
        "checkpoints_non_fp32_stored": n_bf16,
        "checkpoints_orphan": len(orphan_ckpts),
        "checkpoints_duplicate_binding_groups": len(dup_binding_groups),
        "eval_files_checked": len(eval_verdicts),
        "eval_files_rescored_corrected": len(rescored),
        "eval_files_missing_raw_prediction": len(non_rescorable := [
            e for e in eval_verdicts
            if e.get("status") == MISSING_RAW_PREDICTION]),
        "eval_files_invalid_or_unbound": len(bad_evals) + len(orphan_evals),
        "eval_files_split_membership_violations": len(membership_violations),
        "eligible_checkpoints_fully_covered": len(covered_runs),
        "eligible_checkpoints_with_gaps": len(coverage_gaps),
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
            "source_fingerprints": {
                **fingerprints_before,
                "unchanged_during_gate": True,
            },
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
        "runs": sorted(run_verdicts, key=lambda r: (r["root"], r["dir"])),
        "orphan_checkpoints": sorted(orphan_verdicts,
                                     key=lambda o: o["id"]),
        "evaluations": eval_verdicts,
        "quarantine_4b": quarantine,
        "duplicate_checkpoint_bindings": dup_binding_groups,
        "required_splits": list(REQUIRED_SPLITS),
        "preregistered_split_sizes": {
            s: len(expected_by_split[s]) for s in REQUIRED_SPLITS},
        "coverage": {"fully_covered_runs": covered_runs,
                     "gap_runs": coverage_gaps,
                     "identity_mismatches": identity_mismatches,
                     "membership_violations": [
                         {"file": e["file"], "root": e["root"],
                          "split": e["split"]}
                         for e in membership_violations]},
        "proof_tests": None if proof is None else {
            "all_passed": proof["all_passed"],
            "returncode": proof["returncode"],
            "nodes": proof["nodes"],
        },
        "scan_errors": [],
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
    fp = (v.get("inputs") or {}).get("source_fingerprints") or {}
    if fp:
        lines += [
            "",
            "## Input integrity",
            "",
            f"- streaming SHA-256 fingerprints taken before AND after the "
            f"gate matched (`unchanged_during_gate`), so the verdict "
            f"describes exactly these inputs:",
            f"- results_2b merkle `{fp.get('results_2b', {}).get('merkle_sha256', '?')}`",
            f"- results_4b merkle `{fp.get('results_4b', {}).get('merkle_sha256', '?')}`",
        ]
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

    def exec_error(msg: str) -> int:
        print(f"execution error: {msg}", file=sys.stderr)
        return 2

    repo_root = Path.cwd()
    results_2b, results_4b = Path(args.results_2b), Path(args.results_4b)
    out_dir = Path(args.out)
    if not results_2b.is_dir() or not results_4b.is_dir():
        return exec_error(
            f"input roots missing ({results_2b}, {results_4b})")
    # Fail BEFORE any write: --out must not equal, sit inside, or contain
    # either input root (the gate must never self-inventory its outputs).
    if paths_overlap(out_dir, results_2b) or paths_overlap(out_dir, results_4b):
        return exec_error(
            f"--out {out_dir} overlaps an input root "
            f"({results_2b} or {results_4b}); refusing to write")

    try:
        result = run_gate(results_2b, results_4b, repo_root=repo_root,
                          dry_run=args.dry_run,
                          skip_proof_tests=args.skip_proof_tests,
                          proof_log_path=out_dir / "proof_tests.log")
    except GateExecutionError as e:
        return exec_error(str(e))
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return exec_error(f"{type(e).__name__}: {e}")

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
        if not args.no_telemetry:
            import time

            write_canonical(out_dir / "telemetry_timestamp.json", {
                "note": "the ONLY wall-clock output; never referenced by "
                        "canonical verdicts",
                "telemetry_generated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
    except OSError as e:
        return exec_error(f"cannot write gate outputs under {out_dir}: {e}")

    print(f"{result.verdict}: {len(result.gate_verdict['blockers'])} "
          f"blocker(s) -> {out_dir / 'gate_verdict.json'}")
    return _exit_for(result.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
