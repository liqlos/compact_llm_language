"""Finite-integrity checkpointing for the localized scoring runtime.

Everything in this module fails closed:

* adapter payloads are saved atomically (temp file + fsync + os.replace)
  and carry explicit, immutable model provenance (repo + revision);
* loading prevalidates format, metadata provenance, the complete exact
  key set, tensor types, shapes, dtypes and finiteness BEFORE any runtime
  tensor is touched -- a rejected payload leaves state unchanged;
* the best-checkpoint tracker accepts only finite validation metrics over
  finite fp32 adapter states, stores detached clones, and refuses to
  substitute final training state;
* every optimizer update is guarded: non-finite loss / parameters /
  gradients / clip norm / post-update parameters can never reach or
  survive the optimizer's mutation step;
* persisted metrics are walked for NaN/Inf before any JSON is written.

All primitives are pure CPU torch + stdlib; no model I/O happens here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - import-safety without lab group
    torch = None


ADAPTER_FORMAT_VERSION = "rcc.localized-adapter/v1"
ADAPTER_FILENAME = "best_params.pt"

FP32 = "torch.float32"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class AdapterValidationError(RuntimeError):
    """Adapter payload/state failed structural or finiteness validation."""


class AdapterProvenanceError(RuntimeError):
    """Adapter metadata is missing, incomplete, or inconsistent with the
    resolved (repository, immutable revision) identity of the runtime."""


class NonFiniteTrainingState(RuntimeError):
    """Optimizer update path encountered non-finite parameters or an
    invalid clip configuration; the update was not applied."""


class NoAcceptedBestCheckpoint(RuntimeError):
    """No finite validation observation was ever accepted; refusing to
    substitute final training state for a best checkpoint."""


class NonFiniteMetricError(RuntimeError):
    """A metric destined for persistent storage contained NaN or Inf."""


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - import-safety without lab group
        raise RuntimeError("torch required for checkpointing")


# ---------------------------------------------------------------------------
# spec / state validation
# ---------------------------------------------------------------------------

def dtype_name(t: "torch.Tensor") -> str:
    return str(t.dtype)


def normalize_spec(spec: dict) -> dict:
    """Coerce a raw {key: {'shape': .., 'dtype': ..}} spec into canonical
    {key: {'shape': tuple[int,...], 'dtype': str}} form."""
    out = {}
    for key, entry in spec.items():
        shape = tuple(int(x) for x in entry["shape"])
        dt = str(entry["dtype"])
        if not shape or any(s <= 0 for s in shape):
            raise AdapterValidationError(f"{key}: bad spec shape {shape}")
        out[str(key)] = {"shape": shape, "dtype": dt}
    return out


def validate_adapter_state(state, spec: dict) -> None:
    """Full prevalidation of a candidate adapter state against `spec`.

    Checks, without mutating anything:
      * exact key-set equality (missing AND extra keys are rejected)
      * every value is a torch.Tensor
      * shapes match exactly
      * dtypes match exactly (fp32 masters by contract)
      * all values are finite (no NaN/Inf anywhere)

    Raises AdapterValidationError collecting every problem found.
    """
    _require_torch()
    spec = normalize_spec(spec)
    problems: list[str] = []
    if not isinstance(state, dict):
        raise AdapterValidationError(
            f"adapter state must be a dict, got {type(state)!r}")
    keys_state, keys_spec = set(state), set(spec)
    for k in sorted(keys_spec - keys_state):
        problems.append(f"missing key {k!r}")
    for k in sorted(keys_state - keys_spec):
        problems.append(f"unexpected key {k!r}")
    for k in sorted(keys_state & keys_spec):
        v = state[k]
        want = spec[k]
        if not isinstance(v, torch.Tensor):
            problems.append(f"{k}: expected tensor, got {type(v)!r}")
            continue
        got_shape = tuple(v.shape)
        if got_shape != want["shape"]:
            problems.append(
                f"{k}: shape {got_shape} != expected {want['shape']}")
        if dtype_name(v) != want["dtype"]:
            problems.append(
                f"{k}: dtype {dtype_name(v)} != expected {want['dtype']}")
        else:
            finite = bool(torch.isfinite(v).all().item())
            if not finite:
                problems.append(f"{k}: non-finite values (NaN/Inf)")
    if problems:
        raise AdapterValidationError(
            "adapter state rejected: " + "; ".join(problems))


def require_finite_state(state) -> None:
    """Every value must be a finite tensor (dtype/keys unchecked)."""
    _require_torch()
    if not isinstance(state, dict):
        raise AdapterValidationError("adapter state must be a dict")
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            raise AdapterValidationError(f"{k}: expected tensor")
        if not bool(torch.isfinite(v).all().item()):
            raise AdapterValidationError(f"{k}: non-finite values (NaN/Inf)")


def require_fp32_trainables(params) -> None:
    """The training contract is fp32 masters over any backbone precision."""
    _require_torch()
    for i, p in enumerate(params):
        if not isinstance(p, torch.Tensor) or str(p.dtype) != FP32:
            got = None if not isinstance(p, torch.Tensor) else str(p.dtype)
            raise AdapterValidationError(
                f"trainable parameter #{i} must be fp32, got {got!r}")


# ---------------------------------------------------------------------------
# provenance metadata
# ---------------------------------------------------------------------------

def build_adapter_metadata(*, model_repo, model_revision, spec: dict,
                           extra: dict | None = None) -> dict:
    """Metadata bound to an EXPLICIT repository + immutable revision.

    Refuses to invent defaults: empty/None identity raises. `spec` is the
    runtime adapter spec; it is embedded so loaders can cross-check the
    artifact against itself before touching a runtime.
    """
    repo = str(model_repo).strip() if model_repo is not None else ""
    rev = str(model_revision).strip() if model_revision is not None else ""
    if not repo or not rev:
        raise AdapterProvenanceError(
            "refusing to bind adapter metadata: model repository and "
            "immutable revision must both be explicit and non-empty "
            f"(got repo={model_repo!r}, revision={model_revision!r})")
    md = {
        "format": ADAPTER_FORMAT_VERSION,
        "model_repo": repo,
        "model_revision": rev,
        "spec": {str(k): {"shape": list(v["shape"]), "dtype": str(v["dtype"])}
                 for k, v in normalize_spec(spec).items()},
    }
    if extra:
        require_finite_json(extra, where="adapter metadata extra")
        md["extra"] = extra
    return md


def spec_from_metadata(md: dict) -> dict:
    try:
        raw = md["spec"]
        return normalize_spec({k: {"shape": v["shape"], "dtype": v["dtype"]}
                               for k, v in raw.items()})
    except (KeyError, TypeError, ValueError) as e:
        raise AdapterValidationError(
            f"adapter metadata has no usable tensor spec: {e}") from e


def validate_provenance(md: dict, *, model_repo, model_revision) -> None:
    """Exact-match check of payload metadata against resolved identity.

    Both sides must be present and identical; nothing is defaulted."""
    if not isinstance(md, dict):
        raise AdapterProvenanceError("adapter metadata must be a dict")
    want_repo = "" if model_repo is None else str(model_repo).strip()
    want_rev = "" if model_revision is None else str(model_revision).strip()
    if not want_repo or not want_rev:
        raise AdapterProvenanceError(
            "runtime model identity is not bound to an explicit "
            "(repository, immutable revision); refusing to load an "
            "adapter without provenance")
    got_repo = md.get("model_repo")
    got_rev = md.get("model_revision")
    if got_repo != want_repo or got_rev != want_rev:
        raise AdapterProvenanceError(
            "adapter provenance mismatch: payload binds "
            f"(repo={got_repo!r}, revision={got_rev!r}) but runtime "
            f"resolved (repo={want_repo!r}, revision={want_rev!r})")


# ---------------------------------------------------------------------------
# atomic persistence
# ---------------------------------------------------------------------------

def check_adapter_payload(obj) -> dict:
    """Structural + self-consistency validation of a saved adapter payload.

    Payload layout: {"metadata": {...}, "state": {key: Tensor}}. The state
    is validated against the spec embedded in its own metadata, so a
    truncated/tampered artifact is rejected before anyone loads it.
    """
    if not isinstance(obj, dict) or set(obj.keys()) != {"metadata", "state"}:
        raise AdapterValidationError(
            "adapter payload must be a dict with exactly the keys "
            "{'metadata', 'state'}, got "
            f"{sorted(obj) if isinstance(obj, dict) else type(obj)!r}")
    md = obj["metadata"]
    if not isinstance(md, dict) or md.get("format") != ADAPTER_FORMAT_VERSION:
        raise AdapterValidationError(
            f"unsupported adapter metadata/format: {md!r}")
    validate_adapter_state(obj["state"], spec_from_metadata(md))
    return obj


def save_adapter_atomic(path, state: dict, metadata: dict) -> None:
    """Atomically persist {"metadata", "state"} at `path`.

    The state is validated first (finite fp32 tensors per the embedded
    spec), written to a temp file in the same directory, fsynced, then
    moved into place with os.replace. A crash mid-write can never leave a
    partially visible adapter at `path`.
    """
    _require_torch()
    path = Path(path)
    validate_adapter_state(state, spec_from_metadata(metadata))
    require_finite_json(metadata, where="adapter metadata")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            torch.save({"metadata": metadata, "state": {
                k: v.detach().to("cpu").clone()
                for k, v in state.items()}}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_adapter_payload(path) -> dict:
    """Load and fully validate an adapter payload from disk (CPU tensors)."""
    _require_torch()
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    return check_adapter_payload(obj)


def atomic_write_json(path, obj) -> None:
    """Persist metrics/config as JSON after rejecting any NaN/Inf number."""
    require_finite_json(obj)
    tmp = Path(path).with_name(f".{Path(path).name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, Path(path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def require_finite_json(obj, *, where: str = "metrics") -> None:
    """Walk a JSON-bound structure; reject any non-finite float."""
    def walk(node, path):
        if isinstance(node, bool) or node is None or isinstance(node, str):
            return
        if isinstance(node, float):
            if not math.isfinite(node):
                raise NonFiniteMetricError(
                    f"{where}: non-finite value at {path or '<root>'}")
            return
        if isinstance(node, int):
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
            return
        if isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
            return
        raise NonFiniteMetricError(
            f"{where}: value of unsupported type {type(node)!r} "
            f"at {path or '<root>'}")

    walk(obj, "")


# ---------------------------------------------------------------------------
# finite-only best-checkpoint tracker
# ---------------------------------------------------------------------------

class FiniteBestTracker:
    """Accepts ONLY finite validation metrics over finite adapter states.

    Accepted states are stored as detached CPU clones, so later in-place
    mutation of live training weights cannot rewrite history. There is no
    fallback path that returns final/current state: if nothing finite was
    ever accepted, require_best() raises.
    """

    def __init__(self) -> None:
        self._state: dict | None = None
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.rejections: list[dict] = []
        self.observations = 0

    @property
    def has_best(self) -> bool:
        return self._state is not None

    def observe(self, metric, state, step: int) -> bool:
        """Record one validation observation; True iff it became the best.

        Rejections (recorded, never fatal here): non-finite metric,
        non-finite/non-tensor state, non-improving metric."""
        _require_torch()
        self.observations += 1
        step = int(step)
        try:
            m = float(metric)
        except (TypeError, ValueError):
            self.rejections.append(
                {"step": step, "reason": "metric not numeric"})
            return False
        if not math.isfinite(m):
            self.rejections.append(
                {"step": step, "reason": "non-finite validation metric"})
            return False
        try:
            require_finite_state(state)
        except AdapterValidationError as e:
            self.rejections.append(
                {"step": step, "reason": f"non-finite adapter state: {e}"})
            return False
        if self._state is not None and not (m > float(self.best_metric)):
            return False
        self._state = {k: v.detach().to("cpu").clone()
                       for k, v in state.items()}
        self.best_metric = m
        self.best_step = step
        return True

    def require_best(self) -> dict:
        """The selected best state (fresh clone). Never substitutes final
        state: raises if no finite observation was ever accepted."""
        if self._state is None:
            raise NoAcceptedBestCheckpoint(
                "no finite validation metric/state was ever accepted; "
                "refusing to substitute final training state")
        return {k: v.clone() for k, v in self._state.items()}


# ---------------------------------------------------------------------------
# fail-closed optimizer update guard
# ---------------------------------------------------------------------------

APPLIED = "applied"
SKIPPED_NONFINITE_LOSS = "skipped_nonfinite_loss"
SKIPPED_NONFINITE_GRADIENT = "skipped_nonfinite_gradient"
SKIPPED_NONFINITE_GRAD_NORM = "skipped_nonfinite_grad_norm"


def _assert_params_finite(params, stage: str) -> None:
    for i, p in enumerate(params):
        if not bool(torch.isfinite(p.detach()).all().item()):
            raise NonFiniteTrainingState(
                f"{stage}: non-finite parameter tensor #{i}; "
                "refusing to continue")


def finite_guarded_step(opt, loss, params, *, clip_norm: float) -> str:
    """One optimizer update that can never apply corruption.

    Order of gates (criterion C1):
      1. configured clip norm must be finite and positive (config error ->
         raise; this is a programming/config fault, not skippable data)
      2. non-finite loss                -> skip update, params untouched
      3. non-finite parameters          -> raise NonFiniteTrainingState
      4. non-finite gradients           -> skip update, grads discarded
      5. clip_grad_norm_(params, clip) applied; non-finite total norm
                                         -> skip update
      6. opt.step(); post-update params non-finite -> raise

    Returns APPLIED or the skip reason string.
    """
    _require_torch()
    clip = float(clip_norm)
    if not math.isfinite(clip) or clip <= 0.0:
        raise NonFiniteTrainingState(
            "configured clip norm must be finite and positive, "
            f"got {clip_norm!r}")

    lv = float(loss.detach())
    if not math.isfinite(lv):
        opt.zero_grad(set_to_none=True)
        return SKIPPED_NONFINITE_LOSS

    _assert_params_finite(params, stage="pre-update")

    opt.zero_grad(set_to_none=True)
    loss.backward()

    for i, p in enumerate(params):
        if p.grad is not None and not bool(
                torch.isfinite(p.grad).all().item()):
            opt.zero_grad(set_to_none=True)
            return SKIPPED_NONFINITE_GRADIENT

    total_norm = torch.nn.utils.clip_grad_norm_(list(params), clip)
    if not bool(torch.isfinite(torch.as_tensor(total_norm)).all().item()):
        opt.zero_grad(set_to_none=True)
        return SKIPPED_NONFINITE_GRAD_NORM

    opt.step()

    for i, p in enumerate(params):
        if not bool(torch.isfinite(p.detach()).all().item()):
            raise NonFiniteTrainingState(
                f"post-update: non-finite parameter tensor #{i} after "
                "optimizer step; training state is corrupt")
    return APPLIED
