"""Runtime-integrity primitives: fail-closed, identity-bound, fail-stop.

Responsibilities:
  * require_pinned_revision — only immutable 40-hex commit revisions pass;
    rejection happens before any network/model contact.
  * BestCheckpointTracker — accepts only finite validation metrics over
    finite states, clones on accept, and reloads the selected best state;
    it never falls back to final training state.
  * Identity-bound adapter bundles v2 — best_params.pt carries
    (model_id, pinned revision, canonical recipe, suite hash) plus a
    sha256 content digest over every tensor's exact bytes; the recipe
    canonically binds EVERY training-semantic field (mode, interval, k,
    max_k, LoRA rank/alpha, LR, steps, seed, optimizer, schedule/warmup,
    clip, detach/state controls, suite hash + a config digest), and loads
    prevalidate version, kind, identity, recipe, digest,
    keys/tensor/shape/dtype/finiteness before anything is returned.
  * guarded_optimizer_step — cheap fail-stop stepping: no optimizer.step()
    unless loss, clip config, owned parameters, gradients and the total
    gradient norm are finite; gradients rechecked after clipping;
    parameters AND standard supported optimizer-state tensors rechecked
    after the update. ALL reductions are device/dtype-BUCKETED with at
    most one host decision per bucket (accumulators are created on their
    bucket's device — never per tensor/parameter). ANY clipping, step,
    postcheck, inspection or helper fault after the guarded operation
    begins surfaces as FatalRunInvalidError or an explicit fatal subclass
    preserving the cause; the caller must terminate the run and is never
    allowed to emit success artifacts from the mutated in-memory state.
    Recovery happens exclusively from the last atomically committed
    identity-bound checkpoint in a fresh process.
  * NO per-step transaction snapshots: this module never deep-copies or
    rolls back full optimizer/parameter/gradient state around a step.
  * Durable artifacts: atomic temp+fsync+replace JSON writing, file
    digests, full run-manifest schema/kind/status semantics so
    interrupted/fatal runs can never be confused with complete evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - import-safety without lab group
    torch = None


class CheckpointError(RuntimeError):
    """Base class for runtime-integrity failures."""


class EmptyCheckpointError(CheckpointError):
    """No accepted checkpoint exists; refusing to fall back to final state."""


class NonFiniteMetricError(CheckpointError):
    """A validation/persisted metric was not finite."""


class NonFiniteStateError(CheckpointError):
    """A checkpoint state contained non-finite values or bad entries."""


class AdapterBundleError(CheckpointError):
    """An adapter bundle failed structural, digest or content validation."""


class AdapterBundleIdentityError(AdapterBundleError):
    """Bundle was produced for a different identity (model/revision/recipe)."""


class AdapterBundleSchemaError(AdapterBundleError):
    """Bundle violated the metadata schema (key/type/shape/digest field)."""


class FatalRunInvalidError(CheckpointError):
    """THE explicit fatal run-invalid signal.

    Raised by guarded_optimizer_step for any pre-step rejection, clipping
    failure, optimizer.step exception, or post-step non-finite state. The
    calling run MUST terminate immediately; it must never retry the mutated
    in-memory optimizer and must never write reports/checkpoints/manifests
    from this process. Recovery is from the last atomically committed
    identity-bound checkpoint, loaded into a fresh runtime/process.
    """


class NonFiniteTrainingStateError(FatalRunInvalidError):
    """Loss/gradients/parameters/clip-norm/state were non-finite at step time."""


class OptimizerStateInspectionError(FatalRunInvalidError):
    """Optimizer state was not a standard inspectable topology (fail-stop)."""


class GuardedStepFaultError(FatalRunInvalidError):
    """Operational fault INSIDE a guarded operation.

    Distinct from NonFiniteTrainingStateError (a detected non-finite
    state): this is raised when a clipping kernel, optimizer step,
    parameter postcheck or optimizer-state inspection itself raises.
    Both are fatal; the run must terminate and never emit artifacts.
    """


BUNDLE_KIND = "latent_lab.adapter_bundle"
BUNDLE_FORMAT_VERSION = 2

# Every training-semantic field is canonically bound into the recipe.
# Missing/extra/invalid fields are rejected — never defaulted — so two
# materially different trainings can never share an identity.
RECIPE_KEYS = (
    "mode",            # D-full / E-localized / F-control
    "interval",        # [lo, hi) localized layers
    "k",               # latent steps used at train/eval time
    "max_k",           # step-clock horizon
    "lora_r",
    "lora_alpha",
    "lr",
    "steps",           # training steps
    "seed",
    "optimizer",       # optimizer name
    "weight_decay",
    "lr_schedule",     # constant | cosine
    "warmup",
    "clip",
    "detach_z0",
    "suite_sha256",    # exact suite manifest digest
    "config_sha256",   # canonical digest over the normalized config above
)

_LR_SCHEDULES = ("constant", "cosine")

_PINNED_REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch required for checkpointing")


# ---------------------------------------------------------------------------
# durable artifacts (stdlib-only; safe to import without torch)
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    """Content digest of an on-disk artifact."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path, data: bytes) -> None:
    """Durably persist bytes via tmp file + fsync + os.replace."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path, payload: dict) -> str:
    """Atomically persist canonical JSON; returns the file sha256.

    STRICT JSON: ``allow_nan=False`` — NaN/Infinity/-Infinity can never
    be serialized into evidence (they would re-parse as non-finite
    floats and poison validators downstream).
    """
    data = json.dumps(payload, sort_keys=True, indent=1,
                      allow_nan=False).encode()
    atomic_write_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def _reject_parse_constant(name: str) -> None:
    raise ValueError(
        f"non-standard JSON constant {name} is not permitted in evidence")


def strict_json_loads(text):
    """json.loads that REJECTS NaN/Infinity/-Infinity at parse time."""
    return json.loads(text, parse_constant=_reject_parse_constant)


def assert_json_numbers_finite(obj, *, where: str = "evidence") -> None:
    """Recursively require every decoded float to be finite.

    Defense in depth for values entering through already-decoded objects
    or non-standard readers: NaN/±Inf anywhere in a dict/list tree is a
    hard failure naming the offending path.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CheckpointError(
                f"{where}: non-finite number {obj!r} in evidence")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_json_numbers_finite(v, where=f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_json_numbers_finite(v, where=f"{where}[{i}]")


RUN_STATUS_FILE = "run_status.json"
_RUN_STATUSES = ("running", "complete", "fatal", "interrupted")


def write_run_status(out_dir, status: str, **fields) -> dict:
    """Publish the explicit run status (atomic; single source of truth).

    Only status="complete" runs may carry evidence artifacts; readers must
    refuse anything else so interrupted/fatal runs can never be mistaken
    for complete evidence.
    """
    if status not in _RUN_STATUSES:
        raise ValueError(f"unknown run status {status!r}")
    payload = {"status": status, **fields}
    atomic_write_json(Path(out_dir) / RUN_STATUS_FILE, payload)
    return payload


def read_run_status(out_dir) -> dict | None:
    """Read the run status, or None when absent/unreadable/unsafe.

    Parsing is strict (NaN/Infinity constants reject) and the decoded
    payload is scanned for non-finite numbers; any such defect means the
    status is unusable and None (no evidence) is returned.
    """
    p = Path(out_dir) / RUN_STATUS_FILE
    if not p.exists():
        return None
    try:
        st = strict_json_loads(p.read_text())
        assert_json_numbers_finite(st, where=f"{p}: run status")
        return st
    except Exception:  # noqa: BLE001 - unreadable status means unusable run
        return None


def require_complete_run(out_dir) -> dict:
    """Fail closed unless the run directory carries a complete-evidence mark."""
    st = read_run_status(out_dir)
    if st is None:
        raise CheckpointError(
            f"{out_dir}: no {RUN_STATUS_FILE}; run has no complete evidence")
    if st.get("status") != "complete":
        raise CheckpointError(
            f"{out_dir}: run status is {st.get('status')!r}, not 'complete'; "
            "refusing to treat as evidence")
    return st


TRAIN_REPORT_FILE = "train_report.json"
CHECKPOINT_FILE = "best_params.pt"
RUN_MANIFEST_FILE = "run_manifest.json"


class EvidenceLifecycleError(FatalRunInvalidError):
    """Fail-stop publication hygiene could not guarantee a clean root.

    Raised when, after a failed success publication, quarantining the
    fixed-name success artifacts or publishing the fatal status itself
    also fails. The ORIGINAL run exception is preserved as ``__cause__``;
    this error is a fail-closed escalation, never a replacement.
    """


# Fixed-name success artifacts at the evidence root. Quarantine order
# matters: the COMPLETE-status mark dies FIRST so that even a partial
# cleanup can never leave a generation a validator would accept as
# complete (manifest/report/checkpoint are worthless without it).
SUCCESS_ARTIFACT_FILES = (RUN_STATUS_FILE, RUN_MANIFEST_FILE,
                          TRAIN_REPORT_FILE, CHECKPOINT_FILE)


def quarantine_success_artifacts(out_dir) -> list:
    """Invalidate every fixed-name success artifact in the active root.

    Each artifact is atomically renamed aside to
    ``<name>.invalid.<nanos>`` (preserved for forensics); if renaming is
    impossible the file is unlinked. The FIRST failure propagates — the
    caller must then fail closed (see cmd_train / EvidenceLifecycleError).
    Returns the quarantine targets actually created.
    """
    out = Path(out_dir)
    moved = []
    for name in SUCCESS_ARTIFACT_FILES:
        p = out / name
        if not p.exists() and not p.is_symlink():
            continue
        target = None
        for attempt in range(1000):
            cand = p.with_name(
                f"{p.name}.invalid.{time.time_ns()}.{attempt}")
            if not cand.exists():
                target = cand
                break
        if target is None:  # pragma: no cover - absurd collision loop
            raise OSError(f"cannot derive quarantine name for {p}")
        try:
            os.replace(p, target)
        except OSError:
            p.unlink()  # last resort; any failure here propagates
        moved.append(str(target))
    return moved


def write_train_generation(out_dir, *, manifest: dict,
                           report: dict) -> dict:
    """Promote one coherent generation: report, then manifest LAST.

    The caller must already have atomically persisted the checkpoint
    bundle. The manifest carries sha256 digests of BOTH files; its atomic
    arrival is the commit marker that makes the generation verifiable.
    Returns the manifest as written.
    """
    out_dir = Path(out_dir)
    report_path = out_dir / TRAIN_REPORT_FILE
    ckpt_path = out_dir / CHECKPOINT_FILE
    # report first, then its digest, then the manifest as the LAST write:
    # the manifest's arrival marks the generation as verifiable
    atomic_write_json(report_path, report)
    manifest = dict(manifest)
    manifest["report_sha256"] = sha256_file(report_path)
    manifest["checkpoint_sha256"] = sha256_file(ckpt_path)
    atomic_write_json(out_dir / RUN_MANIFEST_FILE, manifest)
    return manifest


RUN_MANIFEST_REQUIRED_KEYS = (
    "kind", "status", "report_sha256", "checkpoint_sha256",
    "checkpoint_content_digest", "identity", "recipe", "suite_sha256",
    "seed", "argv", "dependencies",
)


def _read_json_file(path) -> tuple[dict, str]:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise CheckpointError(f"{p}: unreadable artifact: {e}") from e
    digest = hashlib.sha256(raw).hexdigest()
    try:
        # strict parse: NaN/Infinity constants are rejected outright ...
        data = strict_json_loads(raw.decode())
    except Exception as e:  # noqa: BLE001 - unreadable JSON fails closed
        raise CheckpointError(f"{p}: invalid JSON: {e}") from e
    try:
        # ... and any decoded non-finite number fails closed as well
        assert_json_numbers_finite(data, where=f"{p}")
    except CheckpointError as e:
        raise CheckpointError(f"{p}: invalid JSON evidence: {e}") from e
    return data, digest


def verify_manifest_schema(manifest, *, where: str = "manifest") -> dict:
    """Require the full run-manifest schema, kind and complete status."""
    if not isinstance(manifest, dict):
        raise CheckpointError(f"{where}: not a JSON object")
    missing = [k for k in RUN_MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing:
        raise CheckpointError(
            f"{where}: missing required fields {missing}; refusing a "
            "partial manifest as evidence")
    if manifest.get("kind") != "latent_lab.train_generation":
        raise CheckpointError(
            f"{where}: kind {manifest.get('kind')!r} is not "
            "'latent_lab.train_generation'")
    if manifest.get("status") != "complete":
        raise CheckpointError(
            f"{where}: status {manifest.get('status')!r} is not 'complete'")
    for key in ("report_sha256", "checkpoint_sha256",
                "checkpoint_content_digest"):
        v = manifest.get(key)
        if not isinstance(v, str) or not _SHA256_RE.fullmatch(v):
            raise CheckpointError(f"{where}: {key} is not a 64-hex sha256")
    ident = manifest.get("identity")
    if not isinstance(ident, dict) \
            or not isinstance(ident.get("model_id"), str) \
            or not ident["model_id"].strip() \
            or not isinstance(ident.get("revision"), str):
        raise CheckpointError(f"{where}: identity must bind model_id+revision")
    require_pinned_revision(ident["revision"], name="manifest revision")
    suite = manifest.get("suite_sha256")
    if not isinstance(suite, str) or not _SHA256_RE.fullmatch(suite):
        raise CheckpointError(f"{where}: suite_sha256 is not 64-hex")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CheckpointError(f"{where}: seed must be an int")
    argv = manifest.get("argv")
    if not isinstance(argv, list) or not argv \
            or any(not isinstance(a, str) or not a for a in argv):
        raise CheckpointError(f"{where}: argv must be a non-empty str list")
    deps = manifest.get("dependencies")
    if not isinstance(deps, dict) or not deps:
        raise CheckpointError(f"{where}: dependencies must be a non-empty dict")
    validate_recipe(manifest.get("recipe"))
    return manifest


def verify_generation(adapter_dir) -> dict:
    """Fail closed unless report/checkpoint/manifest form one coherent,
    digest-linked generation of a COMPLETE run under the FULL schema."""
    adapter_dir = Path(adapter_dir)
    require_complete_run(adapter_dir)
    manifest_path = adapter_dir / RUN_MANIFEST_FILE
    if not manifest_path.exists():
        raise CheckpointError(
            f"{adapter_dir}: no {RUN_MANIFEST_FILE}; refusing unverified "
            "checkpoint generation")
    manifest, _ = _read_json_file(manifest_path)
    try:
        verify_manifest_schema(manifest, where=f"{adapter_dir}: manifest")
    except AdapterBundleSchemaError as e:
        raise AdapterBundleIdentityError(
            f"{adapter_dir}: manifest recipe is not a valid canonical "
            f"recipe: {e}") from e
    except CheckpointError:
        raise
    except Exception as e:  # noqa: BLE001 - schema faults fail closed
        raise CheckpointError(
            f"{adapter_dir}: manifest failed schema verification: {e}") from e
    report_path = adapter_dir / TRAIN_REPORT_FILE
    ckpt_path = adapter_dir / CHECKPOINT_FILE
    for name, path in (("report", report_path), ("checkpoint", ckpt_path)):
        if not path.exists():
            raise CheckpointError(f"{adapter_dir}: missing {name} artifact")
    actual = {"report_sha256": sha256_file(report_path),
              "checkpoint_sha256": sha256_file(ckpt_path)}
    for key, digest in actual.items():
        if manifest.get(key) != digest:
            raise CheckpointError(
                f"{adapter_dir}: {key} mismatch (manifest "
                f"{manifest.get(key)!r} != actual {digest!r}); generation "
                "is not coherent")
    return manifest


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------

def require_pinned_revision(value, *, name: str = "revision") -> str:
    """Require an immutable, pinned commit-style revision (40 hex chars).

    Mutable refs — branches (``main``), tags, ``latest``, short shas and
    any other unpinned string — are rejected WITHOUT any network access,
    before loading/training/saving. Falsey values other than an explicit
    documented default are rejected as-is rather than silently replaced.
    The accepted value is normalized consistently (trimmed, lowercased).
    """
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if not _PINNED_REVISION_RE.fullmatch(normalized):
        raise AdapterBundleIdentityError(
            f"{name} must be a pinned immutable 40-hex commit revision; "
            f"got mutable/unpinned/falsey {value!r}")
    return normalized


# ---------------------------------------------------------------------------
# finiteness
# ---------------------------------------------------------------------------

def assert_all_finite(obj, *, where: str = "state") -> None:
    """Recursively require every floating/complex tensor in obj to be finite."""
    _require_torch()
    if torch.is_tensor(obj):
        # complex tensors: torch.isfinite checks BOTH components correctly
        if (obj.is_floating_point() or obj.is_complex()) \
                and not bool(torch.isfinite(obj).all()):
            raise NonFiniteStateError(f"non-finite tensor at {where}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_all_finite(v, where=f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_all_finite(v, where=f"{where}[{i}]")


def validated_state_clone(state) -> dict:
    """Validate a {str: Tensor} state and return detached CPU clones."""
    _require_torch()
    if not isinstance(state, dict) or not state:
        raise NonFiniteStateError("checkpoint state must be a non-empty dict")
    out = {}
    for k, v in state.items():
        if not isinstance(k, str) or not k:
            raise NonFiniteStateError(f"bad checkpoint key {k!r}")
        if not torch.is_tensor(v):
            raise NonFiniteStateError(f"checkpoint value for {k!r} is not a tensor")
        out[k] = v.detach().to("cpu").clone()
    assert_all_finite(out, where="checkpoint")
    return out


# ---------------------------------------------------------------------------
# best-checkpoint tracking
# ---------------------------------------------------------------------------

class BestCheckpointTracker:
    """Keep the best finite validation checkpoint; never the final state."""

    def __init__(self) -> None:
        self._state = None
        self._score = None
        self._step = None

    @property
    def best_score(self):
        return self._score

    @property
    def best_step(self):
        return self._step

    def has_best(self) -> bool:
        return self._state is not None

    def update(self, score, state, step=None) -> bool:
        """Accept (score, state) only if both are valid; clone on improve.

        Strict deterministic schema: score must be a finite real number;
        step must be None or a non-negative int (bools rejected); ties keep
        the earliest accepted checkpoint.
        """
        try:
            s = float(score)
        except (TypeError, ValueError) as e:
            raise NonFiniteMetricError(f"metric {score!r} is not a number") from e
        if not math.isfinite(s):
            raise NonFiniteMetricError(f"validation metric {score!r} is not finite")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise NonFiniteMetricError(
                    f"step must be None or a non-negative int; got {step!r}")
        clean = validated_state_clone(state)
        improved = self._score is None or s > self._score
        if improved:
            self._state = clean
            self._score = s
            self._step = step
        return improved

    def best_state(self) -> dict:
        """Fresh clones of the selected best state."""
        if self._state is None:
            raise EmptyCheckpointError(
                "no accepted checkpoint; refusing to fall back to final state")
        return {k: v.clone() for k, v in self._state.items()}

    def apply_best(self, apply_fn):
        """Reload the selected best state through apply_fn({str: Tensor})."""
        return apply_fn(self.best_state())

    def save(self, path, *, model_id, revision, recipe=None, metrics=None) -> dict:
        """Persist the selected best state as an identity-bound bundle."""
        state = self.best_state()
        met = {"best_score": self._score, "best_step": self._step}
        if metrics:
            met.update(metrics)
        return save_adapter_bundle(path, state, model_id=model_id,
                                   revision=revision, recipe=recipe,
                                   metrics=met)


# ---------------------------------------------------------------------------
# identity-bound adapter bundles (v2): digest + recipe bound
# ---------------------------------------------------------------------------

def _require_identity(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterBundleIdentityError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if name == "revision":
        return require_pinned_revision(normalized, name=name)
    return normalized


def _recipe_int(v, name: str, *, minimum: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise AdapterBundleSchemaError(
            f"recipe.{name} must be an int >= {minimum}; got {v!r}")
    return v


def _recipe_float(v, name: str, *, minimum_exclusive=None,
                  minimum=None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) \
            or not math.isfinite(float(v)):
        raise AdapterBundleSchemaError(
            f"recipe.{name} must be a finite number; got {v!r}")
    f = float(v)
    if minimum_exclusive is not None and not f > minimum_exclusive:
        raise AdapterBundleSchemaError(
            f"recipe.{name} must be > {minimum_exclusive}; got {f!r}")
    if minimum is not None and not f >= minimum:
        raise AdapterBundleSchemaError(
            f"recipe.{name} must be >= {minimum}; got {f!r}")
    return f


def canonical_config_digest(normalized_cfg: dict) -> str:
    """sha256 over the canonical JSON of a normalized semantic config."""
    payload = json.dumps(normalized_cfg, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def recipe_from_config(cfg: dict, suite_sha256: str) -> dict:
    """The exact training-semantic identity a config implies (validated).

    Every field is REQUIRED — missing fields are rejected, never
    defaulted. The returned recipe is the canonical normalized form; its
    ``config_sha256`` binds all training-semantic fields at once so that
    any K/LR/steps/seed/optimizer/schedule/clip/detach change produces a
    materially different identity.
    """
    if not isinstance(cfg, dict):
        raise AdapterBundleSchemaError("config must be a dict")
    _SEMANTIC_CFG_KEYS = ("mode", "interval", "k", "max_k", "lora_r",
                          "lora_alpha", "lr", "steps", "seed", "optimizer",
                          "weight_decay", "lr_schedule", "warmup", "clip",
                          "detach_z0")
    missing = [k for k in _SEMANTIC_CFG_KEYS if k not in cfg]
    if missing:
        raise AdapterBundleSchemaError(
            f"config is missing training-semantic fields {missing}; "
            "refusing to default the training identity")
    mode = cfg["mode"]
    if not isinstance(mode, str) or not mode.strip():
        raise AdapterBundleSchemaError("config.mode must be a non-empty string")
    interval = cfg["interval"]
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(isinstance(x, bool) or not isinstance(x, int)
                   for x in interval)):
        raise AdapterBundleSchemaError("config.interval must be [lo, hi] ints")
    lo, hi = interval
    if not (0 <= lo < hi):
        raise AdapterBundleSchemaError(
            f"config.interval must satisfy 0<=lo<hi; got {[lo, hi]}")
    detach_z0 = cfg["detach_z0"]
    if not isinstance(detach_z0, bool):
        raise AdapterBundleSchemaError("config.detach_z0 must be a bool")
    # Migration input only: old callers may still send the inert flag as
    # false.  It is never persisted or hashed.  True claimed a mechanism the
    # runtime did not implement and therefore fails closed.
    if "grad_checkpoint" in cfg:
        legacy_flag = cfg["grad_checkpoint"]
        if not isinstance(legacy_flag, bool):
            raise AdapterBundleSchemaError(
                "config.grad_checkpoint migration value must be a bool")
        if legacy_flag:
            raise AdapterBundleSchemaError(
                "config.grad_checkpoint=true is unsupported; no executed "
                "gradient-checkpointing mechanism exists")
    schedule = cfg["lr_schedule"]
    if schedule not in _LR_SCHEDULES:
        raise AdapterBundleSchemaError(
            f"config.lr_schedule must be one of {_LR_SCHEDULES}; "
            f"got {schedule!r}")
    optimizer = cfg["optimizer"]
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise AdapterBundleSchemaError(
            "config.optimizer must be a non-empty string")
    seed = cfg["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AdapterBundleSchemaError(f"config.seed must be an int")
    normalized = {
        "mode": mode.strip(),
        "interval": [int(lo), int(hi)],
        "k": _recipe_int(cfg["k"], "k", minimum=0),
        "max_k": _recipe_int(cfg["max_k"], "max_k", minimum=1),
        "lora_r": _recipe_int(cfg["lora_r"], "lora_r", minimum=1),
        "lora_alpha": _recipe_float(cfg["lora_alpha"], "lora_alpha",
                                    minimum_exclusive=0.0),
        "lr": _recipe_float(cfg["lr"], "lr", minimum_exclusive=0.0),
        "steps": _recipe_int(cfg["steps"], "steps", minimum=1),
        "seed": seed,
        "optimizer": optimizer.strip(),
        "weight_decay": _recipe_float(cfg["weight_decay"], "weight_decay",
                                      minimum=0.0),
        "lr_schedule": schedule,
        "warmup": _recipe_int(cfg["warmup"], "warmup", minimum=0),
        "clip": _recipe_float(cfg["clip"], "clip", minimum_exclusive=0.0),
        "detach_z0": detach_z0,
    }
    normalized["config_sha256"] = canonical_config_digest(normalized)
    normalized["suite_sha256"] = suite_sha256
    # the built recipe must itself pass the strict schema it will be
    # verified against on every load/resume
    return validate_recipe(normalized)


def validate_recipe(recipe) -> dict:
    """Strictly validate the recurrence/training recipe bound into bundles.

    The key set must match RECIPE_KEYS exactly (missing AND extra keys are
    rejected), every field must be valid, and ``config_sha256`` must be
    the exact canonical digest of the other semantic fields.
    """
    if not isinstance(recipe, dict):
        raise AdapterBundleSchemaError("recipe must be a dict")
    if set(recipe) != set(RECIPE_KEYS):
        raise AdapterBundleSchemaError(
            f"recipe keys must be exactly {sorted(RECIPE_KEYS)}; "
            f"got {sorted(recipe)}")
    interval = recipe["interval"]
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(isinstance(x, bool) or not isinstance(x, int)
                   for x in interval)):
        raise AdapterBundleSchemaError("recipe.interval must be [lo, hi] ints")
    lo, hi = interval
    if not (0 <= lo < hi):
        raise AdapterBundleSchemaError(f"recipe.interval must satisfy 0<=lo<hi; "
                                       f"got {[lo, hi]}")
    mode = recipe["mode"]
    if not isinstance(mode, str) or not mode.strip():
        raise AdapterBundleSchemaError("recipe.mode must be a non-empty string")
    optimizer = recipe["optimizer"]
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise AdapterBundleSchemaError(
            "recipe.optimizer must be a non-empty string")
    schedule = recipe["lr_schedule"]
    if schedule not in _LR_SCHEDULES:
        raise AdapterBundleSchemaError(
            f"recipe.lr_schedule must be one of {_LR_SCHEDULES}; got "
            f"{schedule!r}")
    if not isinstance(recipe["detach_z0"], bool):
        raise AdapterBundleSchemaError("recipe.detach_z0 must be a bool")
    seed = recipe["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AdapterBundleSchemaError("recipe.seed must be an int")
    suite = recipe["suite_sha256"]
    if not isinstance(suite, str) or not _SHA256_RE.fullmatch(suite):
        raise AdapterBundleSchemaError(
            "recipe.suite_sha256 must be a 64-hex suite manifest digest")
    out = {
        "mode": mode.strip(),
        "interval": [int(lo), int(hi)],
        "k": _recipe_int(recipe["k"], "k", minimum=0),
        "max_k": _recipe_int(recipe["max_k"], "max_k", minimum=1),
        "lora_r": _recipe_int(recipe["lora_r"], "lora_r", minimum=1),
        "lora_alpha": _recipe_float(recipe["lora_alpha"], "lora_alpha",
                                    minimum_exclusive=0.0),
        "lr": _recipe_float(recipe["lr"], "lr", minimum_exclusive=0.0),
        "steps": _recipe_int(recipe["steps"], "steps", minimum=1),
        "seed": seed,
        "optimizer": optimizer.strip(),
        "weight_decay": _recipe_float(recipe["weight_decay"],
                                      "weight_decay", minimum=0.0),
        "lr_schedule": schedule,
        "warmup": _recipe_int(recipe["warmup"], "warmup", minimum=0),
        "clip": _recipe_float(recipe["clip"], "clip",
                               minimum_exclusive=0.0),
        "detach_z0": recipe["detach_z0"],
        "suite_sha256": suite.lower(),
    }
    digest = recipe["config_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise AdapterBundleSchemaError(
            "recipe.config_sha256 must be a 64-hex canonical config digest")
    expected = canonical_config_digest({
        k: out[k] for k in RECIPE_KEYS if k not in
        ("suite_sha256", "config_sha256")})
    if digest.lower() != expected:
        raise AdapterBundleIdentityError(
            f"recipe.config_sha256 mismatch: bound {digest.lower()!r}, "
            f"but semantic fields canonically hash to {expected!r}; the "
            "recipe is internally inconsistent")
    out["config_sha256"] = digest.lower()
    return out


def _tensor_entry(t) -> dict:
    """Metadata + exact-bytes digest entry for one tensor."""
    buf = t.detach().to("cpu", copy=True).contiguous()
    raw = buf.view(torch.uint8).numpy().tobytes()
    return {"shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


def adapter_state_sha256(state) -> str:
    """Deterministically bind an in-memory adapter state to its bytes.

    Validation checkpoints are scored before an on-disk bundle exists.  This
    digest gives each such checkpoint a real content identity; the recipe and
    model identities remain separate fields in ``latent_eval.v3``.
    """
    clean = validated_state_clone(state)
    h = hashlib.sha256()
    for name in sorted(clean):
        entry = _tensor_entry(clean[name])
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(entry, sort_keys=True,
                            separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_content_digest(bundle_core: dict) -> str:
    """Digest over identity + recipe + metrics + every tensor byte range."""
    h = hashlib.sha256()
    h.update(json.dumps(
        {k: bundle_core[k] for k in
         ("format_version", "kind", "model_id", "revision", "recipe",
          "metrics")},
        sort_keys=True).encode())
    for name in sorted(bundle_core["tensors"]):
        e = bundle_core["tensors"][name]
        h.update(name.encode())
        h.update(json.dumps({"shape": e["shape"], "dtype": e["dtype"],
                             "sha256": e["sha256"]}, sort_keys=True).encode())
    return h.hexdigest()


def build_adapter_bundle(state, *, model_id, revision, recipe, metrics=None) -> dict:
    """Build the identity-bound v2 bundle; validates everything up front."""
    clean = validated_state_clone(state)
    mid = _require_identity(model_id, "model_id")
    rev = require_pinned_revision(revision)
    rec = validate_recipe(recipe)
    met: dict = {}
    if metrics:
        if not isinstance(metrics, dict):
            raise AdapterBundleSchemaError("metrics must be a dict")
        for k, v in metrics.items():
            if not isinstance(k, str) or not k:
                raise AdapterBundleSchemaError(f"bad metric key {k!r}")
            if v is None:
                continue  # e.g. no best_step yet
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise AdapterBundleSchemaError(
                    f"metric {k!r} is not a number") from e
            if not math.isfinite(f):
                raise NonFiniteMetricError(
                    f"refusing to persist non-finite metric {k}={v!r}")
            met[k] = f
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "kind": BUNDLE_KIND,
        "model_id": mid,
        "revision": rev,
        "recipe": rec,
        "metrics": met,
        "tensors": {name: _tensor_entry(t) for name, t in clean.items()},
    }
    bundle["content_digest"] = compute_content_digest(bundle)
    # attach tensor payloads after digest computation over metadata+bytes
    for name, t in clean.items():
        bundle["tensors"][name]["data"] = t
    return bundle


def save_adapter_bundle(path, state, *, model_id, revision, recipe,
                        metrics=None) -> dict:
    """Atomically persist an identity-bound bundle (tmp + os.replace)."""
    import tempfile

    bundle = build_adapter_bundle(state, model_id=model_id, revision=revision,
                                  recipe=recipe, metrics=metrics)
    _require_torch()
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(bundle, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return bundle


def _decode_and_verify_bundle(path, *, model_id, revision,
                              recipe) -> dict:
    """Decode + fully prevalidate a v2 bundle; return the raw bundle dict.

    Validation order (all fail-closed, nothing outside is mutated):
      decode -> version/kind -> identity (model_id/revision) ->
      recipe equality -> content digest (detects ANY tampering including
      finite value edits) -> metric finiteness -> per-tensor
      shape/dtype/digest/finiteness before anything is returned.
    """
    _require_torch()
    mid = _require_identity(model_id, "model_id")
    rev = require_pinned_revision(revision)
    rec = validate_recipe(recipe)
    p = Path(path)
    try:
        bundle = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - any decode failure is a bad bundle
        raise AdapterBundleError(f"cannot read bundle {p}: {e}") from e

    if not isinstance(bundle, dict):
        raise AdapterBundleSchemaError("bundle is not a metadata dict")
    if bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise AdapterBundleSchemaError(
            f"unsupported bundle format_version {bundle.get('format_version')!r}"
            " (v2 required)")
    if bundle.get("kind") != BUNDLE_KIND:
        raise AdapterBundleSchemaError(f"unknown bundle kind {bundle.get('kind')!r}")

    bid = bundle.get("model_id")
    brev = bundle.get("revision")
    if not isinstance(bid, str) or not isinstance(brev, str):
        raise AdapterBundleSchemaError("bundle identity fields missing")
    if bid != mid or brev != rev:
        raise AdapterBundleIdentityError(
            f"bundle identity mismatch: saved for ({bid!r}, {brev!r}), "
            f"loading into ({mid!r}, {rev!r})")

    brec = bundle.get("recipe")
    try:
        brec_n = validate_recipe(brec)
    except AdapterBundleSchemaError as e:
        raise AdapterBundleSchemaError(f"bundle recipe invalid: {e}") from e
    if brec_n != rec:
        raise AdapterBundleIdentityError(
            f"bundle recipe mismatch: saved for {brec_n}, loading into {rec}")

    stored_digest = bundle.get("content_digest")
    if not isinstance(stored_digest, str) \
            or not _SHA256_RE.fullmatch(stored_digest):
        raise AdapterBundleSchemaError("bundle content_digest missing/malformed")
    core = {k: bundle.get(k) for k in
            ("format_version", "kind", "model_id", "revision", "recipe",
             "metrics")}
    tensors = bundle.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise AdapterBundleSchemaError("bundle tensors missing")
    core_for_digest = dict(core)
    core_for_digest["tensors"] = {
        name: {k: e.get(k) for k in ("shape", "dtype", "sha256")}
        for name, e in tensors.items()
    }
    try:
        recomputed = compute_content_digest(core_for_digest)
    except Exception as e:  # noqa: BLE001 - malformed fields fail the digest
        raise AdapterBundleSchemaError(
            f"bundle content undigestable: {e}") from e
    if recomputed != stored_digest:
        raise AdapterBundleError(
            "bundle content digest mismatch: file was tampered with or "
            f"corrupted (expected {stored_digest}, computed {recomputed})")

    met = bundle.get("metrics")
    if not isinstance(met, dict):
        raise AdapterBundleSchemaError("bundle metrics missing")
    for k, v in met.items():
        if not isinstance(k, str) or isinstance(v, bool) \
                or not isinstance(v, (int, float)):
            raise AdapterBundleSchemaError(f"bad persisted metric {k!r}")
        if not math.isfinite(float(v)):
            raise NonFiniteMetricError(
                f"non-finite persisted metric {k}={v!r}")

    out = {}
    for name, entry in tensors.items():
        if not isinstance(name, str) or not name:
            raise AdapterBundleSchemaError(f"bad tensor key {name!r}")
        if not isinstance(entry, dict) \
                or set(entry) != {"data", "shape", "dtype", "sha256"}:
            raise AdapterBundleSchemaError(f"tensor {name!r}: bad metadata entry")
        data, shape, dtype = entry["data"], entry["shape"], entry["dtype"]
        if not torch.is_tensor(data):
            raise AdapterBundleSchemaError(f"tensor {name!r}: data is not a Tensor")
        if not isinstance(shape, list) or any(
                isinstance(x, bool) or not isinstance(x, int) for x in shape):
            raise AdapterBundleSchemaError(f"tensor {name!r}: bad shape metadata")
        if list(data.shape) != shape:
            raise AdapterBundleSchemaError(
                f"tensor {name!r}: shape {list(data.shape)} != declared {shape}")
        if not isinstance(dtype, str) or dtype != str(data.dtype):
            raise AdapterBundleSchemaError(
                f"tensor {name!r}: dtype {data.dtype} != declared {dtype!r}")
        actual = _tensor_entry(data)["sha256"]
        if actual != entry["sha256"]:
            raise AdapterBundleError(
                f"tensor {name!r}: byte digest mismatch (tampered values)")
        if (data.is_floating_point() or data.is_complex()) \
                and not bool(torch.isfinite(data).all()):
            raise NonFiniteStateError(f"tensor {name!r} has non-finite values")
    return bundle


def load_adapter_bundle(path, *, model_id, revision, recipe) -> dict:
    """Load + fully prevalidate a v2 bundle before returning any tensor.

    Returned tensors are fresh clones on CPU.
    """
    bundle = _decode_and_verify_bundle(path, model_id=model_id,
                                       revision=revision, recipe=recipe)
    return {name: entry["data"].clone()
            for name, entry in bundle["tensors"].items()}


def inspect_adapter_bundle(path, *, model_id, revision, recipe) -> dict:
    """Fully verify a bundle and return its METADATA (no tensor payloads).

    Runs the exact same fail-closed validation as load_adapter_bundle
    (decode/version/kind/identity/recipe/content digest/metrics/
    per-tensor byte digests + finiteness) for artifact validators that
    must prove a checkpoint is a genuine identity-bound bundle without
    materializing it into a runtime.
    """
    bundle = _decode_and_verify_bundle(path, model_id=model_id,
                                       revision=revision, recipe=recipe)
    meta = {k: v for k, v in bundle.items() if k != "tensors"}
    meta["tensor_names"] = sorted(bundle["tensors"])
    return meta


# ---------------------------------------------------------------------------
# standard optimizer-state topology inspection (finite, fail-stop)
# ---------------------------------------------------------------------------

_SCALAR_OK = (int, float, type(None))


def _acc_dtype_for(device: torch.device) -> torch.dtype:
    """Accumulation dtype for a bucket's device-local reduction.

    float64 wherever the backend supports it (CPU/CUDA); backends without
    float64 (e.g. MPS) accumulate in float32 — the same fail-closed
    bias as before (overflow still implies non-finite-scale state).
    """
    if getattr(device, "type", "") == "mps":
        return torch.float32
    return torch.float64


def _bucket_sq_accumulators(tensors: list) -> dict:
    """One device-local fused squared-norm accumulator per (device, dtype).

    Tensors are grouped by their exact ``(device, dtype)`` bucket; each
    bucket is reduced with a single ``torch._foreach_norm`` pass into an
    accumulator CREATED ON THAT BUCKET'S DEVICE (in that device's
    supported accumulation dtype). No CPU or device-unspecified
    accumulator ever touches accelerator norms, and the host reads at
    most one scalar per bucket — never one per tensor/parameter.
    NaN/Inf survive squared-sum reductions, so a non-finite accumulator
    implies some input was non-finite.
    """
    if not tensors:
        return {}
    groups: dict = {}
    for t in tensors:
        if not torch.is_tensor(t):
            raise GuardedStepFaultError(
                f"non-tensor {type(t).__name__} in fused reduction domain")
        groups.setdefault((str(t.device), str(t.dtype)), []).append(t)
    accs = {}
    for key, group in groups.items():
        dev = group[0].device
        acc_dtype = _acc_dtype_for(dev)
        acc = torch.zeros((), device=dev, dtype=acc_dtype)
        for n in torch._foreach_norm(group):
            n_up = n.to(acc_dtype)
            acc = acc + n_up * n_up
        accs[key] = acc
    return accs


def _all_finite_fused(tensors: list) -> bool:
    """Finiteness of arbitrarily many tensors across devices/dtypes.

    Bounded synchronization contract: at most ONE host decision per
    (device, dtype) bucket; the reduction itself is fully device-local.
    """
    for acc in _bucket_sq_accumulators(tensors).values():
        if not math.isfinite(float(acc.sqrt())):  # single host read/bucket
            return False
    return True


def _fused_total_norm(tensors: list) -> float:
    """Global L2 norm across every device/dtype bucket.

    Mirrors clip_grad_norm_'s squared-sum computation while keeping the
    host read bounded at one per bucket. A non-finite input necessarily
    yields a non-finite total.
    """
    total = 0.0
    for acc in _bucket_sq_accumulators(tensors).values():
        total += float(acc)  # single host read per bucket
    return math.sqrt(total)


def _inspect_state_tree(obj, where: str, seen: set[int],
                        float_out: list | None = None) -> None:
    """Require standard AdamW/SGD-style state topology.

    Supported leaves: tensors, ints/floats/None. Containers:
    dicts/lists/tuples (the standard PyTorch topology). Anything else —
    custom objects, hostile __getattr__ carriers, cyclic references — is
    fatal fail-stop, never silently treated as finite. Floating/complex
    tensor leaves are appended to float_out for one fused reduction
    upstream instead of a per-tensor kernel launch here.
    """
    if torch.is_tensor(obj):
        if obj.is_floating_point() or obj.is_complex():
            if float_out is not None:
                float_out.append(obj)
        return
    if obj is None or isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise NonFiniteTrainingStateError(
                f"non-finite optimizer-state scalar at {where}")
        return
    if isinstance(obj, dict):
        if id(obj) in seen:
            raise OptimizerStateInspectionError(
                f"cyclic optimizer-state container at {where}")
        seen.add(id(obj))
        for k, v in obj.items():
            if not isinstance(k, (str, int)) :
                raise OptimizerStateInspectionError(
                    f"non-standard optimizer-state key {k!r} at {where}")
            _inspect_state_tree(v, f"{where}[{k!r}]", seen, float_out)
        seen.discard(id(obj))
        return
    if isinstance(obj, (list, tuple)):
        if id(obj) in seen:
            raise OptimizerStateInspectionError(
                f"cyclic optimizer-state container at {where}")
        seen.add(id(obj))
        for i, v in enumerate(obj):
            _inspect_state_tree(v, f"{where}[{i}]", seen, float_out)
        seen.discard(id(obj))
        return
    raise OptimizerStateInspectionError(
        f"uninspectable optimizer-state entry of type "
        f"{type(obj).__name__} at {where}; only standard mappings/"
        f"sequences/scalars/tensors are supported (fail-stop)")


def validate_optimizer_state_standard_and_finite(optimizer) -> None:
    """Post-step check over optimizer.state for EVERY owned parameter.

    Topology must be the standard mappings/sequences/scalars/tensors
    layout (anything else is fatal fail-stop); finiteness of all floating
    state tensors is verified in ONE device-bucketed fused reduction.
    ANY exception raised by the inspection itself surfaces as
    OptimizerStateInspectionError (a fatal subclass) preserving cause —
    a faulting inspection can never leak a raw foreign exception.
    """
    _require_torch()
    try:
        state = getattr(optimizer, "state", None)
        if state is None:
            raise OptimizerStateInspectionError(
                "optimizer has no .state mapping")
        if not isinstance(state, dict):
            raise OptimizerStateInspectionError(
                f"optimizer.state is {type(state).__name__}, not a dict")
        float_tensors: list = []
        for p, entry in state.items():
            if not torch.is_tensor(p) and not hasattr(p, "grad"):
                raise OptimizerStateInspectionError(
                    f"optimizer.state keyed by non-parameter {p!r}")
            # one walk enforces the standard topology AND collects leaves
            _inspect_state_tree(entry, f"state[{id(p):#x}]", set(),
                                float_tensors)
        if not _all_finite_fused(float_tensors):
            raise NonFiniteTrainingStateError(
                "non-finite optimizer-state tensor after optimizer.step")
    except FatalRunInvalidError:
        raise
    except Exception as e:  # noqa: BLE001 - inspection faults are fatal
        raise OptimizerStateInspectionError(
            f"optimizer-state inspection faulted "
            f"({type(e).__name__}: {e}); fail-stop") from e


# ---------------------------------------------------------------------------
# fail-stop optimizer stepping (NO snapshot, NO rollback, NO retry)
# ---------------------------------------------------------------------------

def owned_parameters(optimizer, params) -> list:
    """Deterministic union of optimizer groups' params and caller params.

    First occurrence wins; live Parameter objects are never copied or
    replaced, so identity survives. Caller-omitted group members stay fully
    covered by pre/post checks.
    """
    out, seen = [], set()

    def add(p):
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)

    for group in getattr(optimizer, "param_groups", ()) or ():
        for p in group.get("params", ()):  # standard groups are dicts
            add(p)
    for p in params or ():
        add(p)
    return out


def guarded_optimizer_step(optimizer, loss, params, clip_norm) -> None:
    """Step only when everything is finite; otherwise kill the run.

    Cheap fail-stop contract (deliberately NO snapshot/rollback):
      PRE-STEP (nothing mutated yet): finite loss tensor, positive finite
      clip norm, finite owned parameters (union of optimizer groups and
      the caller's iterable), and ONE device-bucketed gradient-norm pass
      whose non-finite result implies some gradient (or the norm itself)
      is non-finite — NaN/Inf cannot survive a squared-sum reduction.
      CLIP: gradients scaled at most once (single fused pass), then
      rechecked with the same bounded helper (<=1 host decision per
      device/dtype bucket, never one per gradient).
      STEP: optimizer.step(); any exception is fatal.
      POST: every owned parameter AND all standard optimizer-state tensors
      rechecked for finiteness through the same bounded helper.
    ANY clipping, step, parameter-postcheck, optimizer-state-inspection or
    helper fault after the guarded operation begins surfaces as
    FatalRunInvalidError (or an explicit fatal subclass) preserving the
    original cause. The caller must terminate the run, must not retry this
    optimizer, and must not write success artifacts from the possibly
    mutated in-memory state.
    """
    _require_torch()
    try:
        _guarded_step_checked(optimizer, loss, params, clip_norm)
    except FatalRunInvalidError:
        raise
    except Exception as e:  # noqa: BLE001 - any guarded-op fault is fatal
        raise GuardedStepFaultError(
            f"guarded optimizer step faulted ({type(e).__name__}: {e}); "
            "run invalid, no rollback attempted") from e


def _guarded_step_checked(optimizer, loss, params, clip_norm) -> None:
    """guarded_optimizer_step body; every raise here is fatal."""
    # -- pre-step validation: nothing has been mutated yet -------------------
    if loss is None or not torch.is_tensor(loss) \
            or not bool(torch.isfinite(loss.detach()).all()):
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: non-finite loss")
    try:
        clip = float(clip_norm)
    except (TypeError, ValueError) as e:
        raise NonFiniteTrainingStateError(
            f"refusing optimizer.step: invalid clip norm {clip_norm!r}") from e
    if not math.isfinite(clip) or clip <= 0.0:
        raise NonFiniteTrainingStateError(
            f"refusing optimizer.step: invalid clip norm {clip_norm!r}")
    owned = owned_parameters(optimizer, list(params))
    if not owned:
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: no owned parameters")
    for p in owned:
        if not torch.is_tensor(p):
            raise NonFiniteTrainingStateError(
                "refusing optimizer.step: non-parameter in owned set")
    # ONE device-bucketed reduction validates every owned parameter
    if not _all_finite_fused([p.detach() for p in owned]):
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: non-finite parameter")
    grads = [p.grad for p in owned
             if p.grad is not None and torch.is_tensor(p.grad)]
    if not grads:
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: no gradients attached (call backward)")
    # ONE device-bucketed reduction: a non-finite gradient necessarily
    # produces a non-finite total norm, so this single pass validates
    # grads + norm. Mirrors clip_grad_norm_'s own computation, so no
    # second pass is due; host reads stay bounded per device/dtype bucket.
    total_norm = _fused_total_norm(grads)
    if not math.isfinite(total_norm):
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: non-finite gradient or gradient norm "
            "(clipping would silently zero it)")

    # -- clip (single fused pass) + mandated recheck --------------------------
    clip_coef = clip / (total_norm + 1e-6)
    if clip_coef < 1.0:
        torch._foreach_mul_(grads, clip_coef)
        # scaling finite tensors by <1 cannot create NaN/Inf, but the
        # recheck is contract-mandated evidence against hostile grad impls;
        # it uses the SAME bounded bucketed helper (no per-gradient sync)
        if not _all_finite_fused(grads):
            raise NonFiniteTrainingStateError(
                "gradients became non-finite after clipping")

    # -- step (no snapshot is taken; recovery is from the committed bundle) --
    try:
        optimizer.step()
    except BaseException as e:  # noqa: BLE001 - step faults are fatal
        raise NonFiniteTrainingStateError(
            f"optimizer.step raised; run invalid, no rollback attempted "
            f"(recover from last committed checkpoint in a fresh process): "
            f"{e}") from e

    # -- post-step validation -------------------------------------------------
    if not _all_finite_fused([p.detach() for p in owned]):
        raise NonFiniteTrainingStateError(
            "parameters became non-finite after optimizer.step")
    validate_optimizer_state_standard_and_finite(optimizer)
