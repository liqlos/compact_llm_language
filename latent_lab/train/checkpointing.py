"""Runtime-integrity primitives for latent training runs.

Covers three fail-closed contracts:
  * BestCheckpointTracker — accepts only finite validation metrics with
    finite tensors, deep-clones the accepted state, and never falls back
    to the final state when nothing was accepted;
  * guarded_optimizer_step — refuses optimizer.step() unless the loss,
    every gradient, every parameter and the post-clip total norm are
    finite, re-checking gradients after clipping and parameters after
    the update;
  * identity-bound adapter bundles at best_params.pt — the bundle pins
    model_id + immutable revision and every tensor is prevalidated
    (key set, tensor type, shape, dtype, finiteness) before any copy,
    so a failed load leaves the runtime untouched.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - import-safety without lab group
    torch = None

BUNDLE_FORMAT = "latent_lab.adapter_bundle"
BUNDLE_VERSION = 1
BUNDLE_FILENAME = "best_params.pt"


class NonFiniteMetricError(ValueError):
    """A persisted/tracked metric is NaN or infinite."""


class NonFiniteStateError(ValueError):
    """A checkpoint state contains non-finite or non-tensor entries."""


class NonFiniteTrainingStateError(RuntimeError):
    """Loss/grads/params/clip-norm were non-finite; step refused."""


class IdentityMismatchError(ValueError):
    """Bundle identity does not match the requested model_id/revision."""


class AdapterBundleError(ValueError):
    """Malformed adapter bundle (schema, keys, shape or dtype)."""


def require_finite_metric(name: str, value) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise NonFiniteMetricError(
            f"{name} is not finite: {value!r}")
    return v


def validate_state_tensors(state, *, where="checkpoint state") -> None:
    """Every entry must be a str -> torch.Tensor; floats must be finite."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch required")
    if not isinstance(state, dict) or not state:
        raise TypeError(f"{where}: expected a non-empty dict of tensors")
    for key, t in state.items():
        if not isinstance(key, str):
            raise TypeError(f"{where}: keys must be str, got {type(key)!r}")
        if not torch.is_tensor(t):
            raise TypeError(
                f"{where}[{key!r}]: expected torch.Tensor, got {type(t)!r}")
        if t.is_floating_point() and not bool(torch.isfinite(t).all()):
            raise NonFiniteStateError(f"{where}[{key!r}] has non-finite values")


def clone_state(state: dict) -> dict:
    """Validated deep copy on CPU; later mutation of sources is invisible."""
    validate_state_tensors(state)
    return {k: t.detach().to("cpu").clone() for k, t in state.items()}


class BestCheckpointTracker:
    """Tracks the best finite validation metric; clones accepted state."""

    def __init__(self, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self._metric: float | None = None
        self._step: int | None = None
        self._state: dict | None = None

    @property
    def has_best(self) -> bool:
        return self._state is not None

    @property
    def best_metric(self) -> float | None:
        return self._metric

    @property
    def best_step(self) -> int | None:
        return self._step

    @property
    def state(self) -> dict | None:
        return self._state

    def update(self, metric, state: dict, step: int) -> bool:
        """Accept (metric, state) only if finite; clone on acceptance."""
        m = require_finite_metric("validation metric", metric)
        candidate = clone_state(state)
        better = not self.has_best or (
            m > self._metric if self.mode == "max" else m < self._metric)
        if better:
            self._metric, self._step, self._state = m, int(step), candidate
            return True
        return False

    def require_state(self) -> dict:
        """Selected best state; NEVER falls back to the final state."""
        if self._state is None:
            raise RuntimeError(
                "no checkpoint with finite validation metric was accepted; "
                "refusing to fall back to the final state")
        return dict(self._state)


# ---------------------------------------------------------------------------
# guarded optimizer step
# ---------------------------------------------------------------------------

def _grads_finite(params) -> bool:
    return all(torch.isfinite(p.grad).all()
               for p in params if p.grad is not None)


def _params_finite(params) -> bool:
    return all(torch.isfinite(p).all() for p in params)


def guarded_optimizer_step(loss, opt, params, *, max_norm: float):
    """clip + step only when loss/grads/params/norm are all finite.

    Order of defense: loss -> gradients -> parameters -> clip(args.clip)
    -> clip-norm -> gradient recheck after clipping -> opt.step()
    -> parameter recheck after the update. Any failure raises before or
    immediately after the mutation point; no partial state is consumed.
    """
    params = list(params)
    if loss is not None:
        lval = float(loss.detach())
        if not math.isfinite(lval):
            raise NonFiniteTrainingStateError(
                f"non-finite loss {lval}; optimizer step refused")
    if not _grads_finite(params):
        raise NonFiniteTrainingStateError(
            "non-finite gradient(s); optimizer step refused")
    if not _params_finite(params):
        raise NonFiniteTrainingStateError(
            "non-finite parameter(s) before step; optimizer step refused")
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if not torch.isfinite(total_norm).all():
        raise NonFiniteTrainingStateError(
            f"non-finite clip norm {float(total_norm)}; step refused")
    if not _grads_finite(params):
        raise NonFiniteTrainingStateError(
            "gradients became non-finite after clipping; step refused")
    opt.step()
    if not _params_finite(params):
        raise NonFiniteTrainingStateError(
            "parameters became non-finite after update")
    return total_norm


# ---------------------------------------------------------------------------
# identity-bound adapter bundles
# ---------------------------------------------------------------------------

def save_adapter_bundle(out_dir, state: dict, *, model_id: str,
                        revision: str) -> Path:
    """Write {format, version, model_id, revision, tensors} atomically."""
    validate_state_tensors(state, where="adapter bundle")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": BUNDLE_FORMAT,
        "bundle_version": BUNDLE_VERSION,
        "model_id": str(model_id),
        "revision": str(revision),
        "tensors": {k: t.detach().to("cpu").clone()
                    for k, t in state.items()},
    }
    final = out / BUNDLE_FILENAME
    tmp = out / (BUNDLE_FILENAME + ".tmp")
    torch.save(bundle, tmp)
    os.replace(tmp, final)
    return final


def _load_raw_bundle(adapter_dir) -> dict:
    path = Path(adapter_dir) / BUNDLE_FILENAME
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as e:
        raise AdapterBundleError(f"missing adapter bundle: {path}") from e
    except Exception as e:  # noqa: BLE001 — corrupt archive is a bundle error
        raise AdapterBundleError(f"unreadable adapter bundle {path}: {e}") from e
    if not isinstance(obj, dict):
        raise AdapterBundleError("bundle root must be a dict")
    if obj.get("format") != BUNDLE_FORMAT:
        raise AdapterBundleError(
            f"bad bundle format {obj.get('format')!r}; expected "
            f"{BUNDLE_FORMAT!r}")
    if obj.get("bundle_version") != BUNDLE_VERSION:
        raise AdapterBundleError(
            f"unsupported bundle version {obj.get('bundle_version')!r}")
    return obj


def load_adapter_bundle(adapter_dir, *, model_id: str, revision: str,
                        expected: dict | None = None) -> dict:
    """Validate everything, THEN return cloned CPU tensors.

    `expected` maps key -> (shape tuple, torch.dtype); when given, the key
    set plus every shape/dtype/finiteness constraint is checked before any
    tensor is handed back, so a rejected load mutates nothing downstream.
    """
    obj = _load_raw_bundle(adapter_dir)
    bid, brv = obj.get("model_id"), obj.get("revision")
    if bid != str(model_id) or brv != str(revision):
        raise IdentityMismatchError(
            f"bundle identity ({bid!r}, {brv!r}) does not match runtime "
            f"identity ({str(model_id)!r}, {str(revision)!r})")
    tensors = obj.get("tensors")
    validate_state_tensors(tensors, where="bundle tensors")
    if expected is not None:
        want_keys, got_keys = set(expected), set(tensors)
        missing, extra = want_keys - got_keys, got_keys - want_keys
        if missing or extra:
            raise AdapterBundleError(
                f"bundle key mismatch; missing={sorted(missing)} "
                f"extra={sorted(extra)}")
        for key, (shape, dtype) in expected.items():
            t = tensors[key]
            if tuple(t.shape) != tuple(shape):
                raise AdapterBundleError(
                    f"{key!r}: shape {tuple(t.shape)} != expected "
                    f"{tuple(shape)}")
            if t.dtype != dtype:
                raise AdapterBundleError(
                    f"{key!r}: dtype {t.dtype} != expected {dtype}")
    return {k: t.clone() for k, t in tensors.items()}
