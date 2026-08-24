"""Runtime-integrity primitives.

Three responsibilities live here:
  * BestCheckpointTracker — accepts only finite validation metrics over
    finite states, clones the accepted state, and can reload the selected
    best state; it never falls back to the final training state.
  * Identity-bound adapter bundles — the on-disk best_params.pt carries the
    (model_id, revision) it was trained against plus per-tensor metadata;
    revisions must be pinned immutable 40-hex commit ids (no network);
    loads prevalidate every key, tensor type, shape, dtype and finiteness
    before any tensor is copied, so failed loads are atomic.
  * guarded_optimizer_step — transactional fail-closed stepping: no
    optimizer.step() unless loss, the clip configuration, parameters,
    gradients and the clip norm are all finite, with rechecks after
    clipping and after the update; supplied params must exactly cover the
    optimizer-owned ones; any post-step corruption rolls the parameters,
    gradients and the complete optimizer state AND param-group structure
    (group count/order, fields and per-group Parameter identities/order)
    back bit-exactly.
"""

from __future__ import annotations

import copy
import math
import os
import re
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


class NonFiniteTrainingStateError(CheckpointError):
    """Loss/gradients/parameters/clip-norm were non-finite at step time."""


class OptimizerParamCoverageError(CheckpointError):
    """Supplied parameters did not exactly cover the optimizer-owned ones.

    ``optimizer.step()`` updates every Parameter in ``param_groups``
    regardless of what the caller passes, so a call whose supplied
    parameters omit (or add beyond) the optimizer-owned identities is
    rejected BEFORE any mutation: no owned Parameter may be updated
    without snapshot, post-check and rollback coverage.
    """


class AdapterBundleError(CheckpointError):
    """An adapter bundle failed structural or content validation."""


class AdapterBundleIdentityError(AdapterBundleError):
    """Bundle was produced for a different (model_id, revision)."""


class AdapterBundleSchemaError(AdapterBundleError):
    """Bundle violated the metadata schema (key/type/shape/dtype)."""


BUNDLE_KIND = "latent_lab.adapter_bundle"
BUNDLE_FORMAT_VERSION = 1


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch required for checkpointing")


def assert_all_finite(obj, *, where: str = "state") -> None:
    """Recursively require every floating tensor in obj to be finite."""
    _require_torch()
    if torch.is_tensor(obj):
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
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
        """Accept (score, state) only if both are finite; clone on improve."""
        try:
            s = float(score)
        except (TypeError, ValueError) as e:
            raise NonFiniteMetricError(f"metric {score!r} is not a number") from e
        if not math.isfinite(s):
            raise NonFiniteMetricError(f"validation metric {score!r} is not finite")
        clean = validated_state_clone(state)
        improved = self._score is None or s > self._score
        if improved:
            self._state = clean
            self._score = s
            self._step = None if step is None else float(step)
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

    def save(self, path, *, model_id, revision):
        """Persist the selected best state as an identity-bound bundle."""
        state = self.best_state()
        return save_adapter_bundle(
            path, state, model_id=model_id, revision=revision,
            metrics={"best_score": self._score, "best_step": self._step})


# ---------------------------------------------------------------------------
# identity-bound adapter bundles
# ---------------------------------------------------------------------------

def _require_identity(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterBundleIdentityError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if name == "revision":
        return require_pinned_revision(normalized, name=name)
    return normalized


_PINNED_REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def require_pinned_revision(value, *, name: str = "revision") -> str:
    """Require an immutable, pinned commit-style revision (40 hex chars).

    Mutable refs — branches (``main``), tags, ``latest``, short shas and
    any other unpinned string — are rejected WITHOUT any network access,
    before loading/training/saving. The accepted value is normalized
    consistently (trimmed, lowercased) so reports and adapter metadata
    store and compare exactly that pinned revision.
    """
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if not _PINNED_REVISION_RE.fullmatch(normalized):
        raise AdapterBundleIdentityError(
            f"{name} must be a pinned immutable 40-hex commit revision; "
            f"got mutable/unpinned {value!r}")
    return normalized


def build_adapter_bundle(state, *, model_id, revision, metrics=None) -> dict:
    """Build the identity-bound bundle; validates everything up front."""
    clean = validated_state_clone(state)
    mid = _require_identity(model_id, "model_id")
    rev = _require_identity(revision, "revision")
    met: dict = {}
    if metrics:
        if not isinstance(metrics, dict):
            raise AdapterBundleSchemaError("metrics must be a dict")
        for k, v in metrics.items():
            if not isinstance(k, str) or not k:
                raise AdapterBundleSchemaError(f"bad metric key {k!r}")
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise AdapterBundleSchemaError(
                    f"metric {k!r} is not a number") from e
            if not math.isfinite(f):
                raise NonFiniteMetricError(
                    f"refusing to persist non-finite metric {k}={v!r}")
            met[k] = f
    tensors = {
        name: {"data": t, "shape": list(t.shape), "dtype": str(t.dtype)}
        for name, t in clean.items()
    }
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "kind": BUNDLE_KIND,
        "model_id": mid,
        "revision": rev,
        "metrics": met,
        "tensors": tensors,
    }


def save_adapter_bundle(path, state, *, model_id, revision, metrics=None) -> dict:
    """Atomically persist an identity-bound bundle (tmp file + os.replace)."""
    bundle = build_adapter_bundle(state, model_id=model_id, revision=revision,
                                  metrics=metrics)
    _require_torch()
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(bundle, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return bundle


def load_adapter_bundle(path, *, model_id, revision) -> dict:
    """Load + fully prevalidate a bundle before returning any tensor.

    Nothing outside this function is mutated; the returned tensors are fresh
    clones, so a failure anywhere leaves the caller's model untouched.
    """
    _require_torch()
    mid = _require_identity(model_id, "model_id")
    rev = _require_identity(revision, "revision")
    p = Path(path)
    try:
        bundle = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - any decode failure is a bad bundle
        raise AdapterBundleError(f"cannot read bundle {p}: {e}") from e

    if not isinstance(bundle, dict):
        raise AdapterBundleSchemaError("bundle is not a metadata dict")
    if bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise AdapterBundleSchemaError(
            f"unsupported bundle format_version {bundle.get('format_version')!r}")
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

    tensors = bundle.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise AdapterBundleSchemaError("bundle tensors missing")

    out = {}
    for name, entry in tensors.items():
        if not isinstance(name, str) or not name:
            raise AdapterBundleSchemaError(f"bad tensor key {name!r}")
        if not isinstance(entry, dict) or set(entry) != {"data", "shape", "dtype"}:
            raise AdapterBundleSchemaError(f"tensor {name!r}: bad metadata entry")
        data, shape, dtype = entry["data"], entry["shape"], entry["dtype"]
        if not torch.is_tensor(data):
            raise AdapterBundleSchemaError(f"tensor {name!r}: data is not a Tensor")
        if not isinstance(shape, list) or any(
                not isinstance(x, int) for x in shape):
            raise AdapterBundleSchemaError(f"tensor {name!r}: bad shape metadata")
        if list(data.shape) != shape:
            raise AdapterBundleSchemaError(
                f"tensor {name!r}: shape {list(data.shape)} != declared {shape}")
        if not isinstance(dtype, str) or dtype != str(data.dtype):
            raise AdapterBundleSchemaError(
                f"tensor {name!r}: dtype {data.dtype} != declared {dtype!r}")
        if data.is_floating_point() and not bool(torch.isfinite(data).all()):
            raise NonFiniteStateError(f"tensor {name!r} has non-finite values")
        out[name] = data.clone()
    return out


# ---------------------------------------------------------------------------
# fail-closed optimizer stepping
# ---------------------------------------------------------------------------

def _deep_clone_tree(obj):
    """Exact deep copy of a tensor tree (optimizer state shapes)."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _deep_clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_clone_tree(v) for v in obj)
    return obj


def _tree_is_finite(obj) -> bool:
    if torch.is_tensor(obj):
        return not obj.is_floating_point() or bool(torch.isfinite(obj).all())
    if isinstance(obj, dict):
        return all(_tree_is_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_tree_is_finite(v) for v in obj)
    return True


def _restore_optimizer_state(optimizer, snap) -> None:
    """Roll optimizer.state back to the snapshot bit-exactly, in place."""
    state = optimizer.state
    for key in list(state.keys()):
        if key not in snap:
            del state[key]
    for key, snap_entry in snap.items():
        live_entry = state.get(key) if hasattr(state, "get") else None
        if not isinstance(live_entry, dict) or \
                not isinstance(snap_entry, dict):
            state[key] = _deep_clone_tree(snap_entry)
            continue
        for name in list(live_entry.keys()):
            if name not in snap_entry:
                del live_entry[name]
        for name, val in snap_entry.items():
            live_val = live_entry.get(name)
            if torch.is_tensor(live_val) and torch.is_tensor(val):
                with torch.no_grad():
                    live_val.copy_(val)
            else:
                live_entry[name] = _deep_clone_tree(val)


def _snapshot_param_groups(optimizer) -> list:
    """Snapshot the COMPLETE ``param_groups`` structure.

    Per group (in order): a reference to the original group dict, deep
    copies of every non-``params`` field, a reference to the original
    ``params`` list object and the ordered list of the live Parameter
    objects themselves. Parameters are never copied or replaced — only
    their identities are recorded.
    """
    snap = []
    for group in optimizer.param_groups:
        entry = {
            "group": group,
            "fields": {
                k: (v.detach().clone() if torch.is_tensor(v)
                    else copy.deepcopy(v))
                for k, v in group.items() if k != "params"},
            "has_params": "params" in group,
            "params_list": group.get("params"),
            "param_refs": list(group.get("params", ())),
        }
        snap.append(entry)
    return snap


def _restore_param_groups(optimizer, snap) -> None:
    """Restore the COMPLETE param-group structure IN PLACE: exact group
    count/order (the original group dict objects themselves), every field
    value, and each group's ``params`` list holding the original live
    Parameter identities in the original order. Hostile additions,
    removals, swaps and reordering of groups or of parameters within
    groups are all undone; nothing is ever deep-copied or replaced at the
    Parameter level; fields injected after the snapshot are removed."""
    rebuilt = [entry["group"] for entry in snap]
    if isinstance(optimizer.param_groups, list):
        optimizer.param_groups[:] = rebuilt
    else:
        optimizer.param_groups = rebuilt
    for entry in snap:
        group = entry["group"]
        fields = entry["fields"]
        for key in list(group.keys()):
            if key != "params" and key not in fields:
                del group[key]
        for key, val in fields.items():
            group[key] = (val.detach().clone() if torch.is_tensor(val)
                          else copy.deepcopy(val))
        if entry["has_params"]:
            params_list = entry["params_list"]
            params_list[:] = entry["param_refs"]
            group["params"] = params_list


def _optimizer_owned_params(optimizer) -> list:
    """Unique live Parameters owned by the optimizer, in deterministic
    pre-step group order (first occurrence wins for cross-group sharing)."""
    owned, seen = [], set()
    for group in optimizer.param_groups:
        for p in group.get("params", ()):
            if id(p) not in seen:
                seen.add(id(p))
                owned.append(p)
    return owned


def guarded_optimizer_step(optimizer, loss, params, clip_norm) -> None:
    """Step only when everything is finite; roll back completely otherwise.

    Transactional contract:
      * PRE-STEP (before any mutation): finite loss, positive finite clip,
        finite parameters, finite gradients and a finite total gradient
        norm. Any rejection here performs no step and mutates nothing.
      * The caller-supplied ``params`` iterable is never trusted as
        exhaustive: the supplied Parameter identities must EXACTLY cover
        the optimizer-owned ones (``optimizer.step()`` updates every group
        member whether or not it was supplied). Otherwise the call is
        rejected before any mutation. The transaction set is the deduped,
        optimizer-owned parameter list in pre-step group order, so every
        owned Parameter — including duplicates across groups, which are
        handled deterministically by first occurrence — has snapshot,
        post-check and rollback coverage.
      * Gradients are rechecked after clipping; a post-clip rejection also
        restores the pre-clip gradients.
      * Every trainable parameter, every pre-clip gradient (including the
        ``None`` versus tensor distinction and exact bytes), a deep exact
        copy of the optimizer state and the complete ``param_groups``
        structure — exact group count/order, every field value and each
        group's ``params`` list with the original live Parameter
        identities in the original order — are snapshotted before
        ``optimizer.step()``. If the step raises, or any post-step
        parameter or optimizer-state tensor is non-finite, all of it —
        parameters, gradients, complete optimizer state including injected
        fields, and the full param-group topology and values — is restored
        exactly while keeping the live Parameter identities (and the
        original optimizer object), then the error is raised. The
        optimizer remains usable after rollback.
    """
    _require_torch()
    # -- pre-step validation: nothing has been mutated yet -------------------
    if loss is None or not torch.is_tensor(loss) \
            or not bool(torch.isfinite(loss.detach()).all()):
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: non-finite loss")
    clip = float(clip_norm)
    if not math.isfinite(clip) or clip <= 0.0:
        raise NonFiniteTrainingStateError(
            f"refusing optimizer.step: invalid clip norm {clip_norm!r}")
    supplied = list(params)
    supplied_ids = set()
    for p in supplied:
        supplied_ids.add(id(p))
    owned = _optimizer_owned_params(optimizer)
    owned_ids = {id(p) for p in owned}
    if supplied_ids != owned_ids:
        missing = sum(1 for i in owned_ids if i not in supplied_ids)
        unknown = len(supplied) - len(supplied_ids & owned_ids)
        raise OptimizerParamCoverageError(
            "refusing optimizer.step: supplied params do not exactly cover "
            f"the optimizer-owned parameters ({missing} omitted, "
            f"{unknown} not owned); optimizer.step() would update "
            "Parameters without snapshot/rollback coverage")
    params = owned  # authoritative deterministic transaction set
    for p in params:
        if not bool(torch.isfinite(p.detach()).all()):
            raise NonFiniteTrainingStateError(
                "refusing optimizer.step: non-finite parameter")
    pairs = [(p, p.grad) for p in params if p.grad is not None]
    for _, g in pairs:
        if not bool(torch.isfinite(g).all()):
            raise NonFiniteTrainingStateError(
                "refusing optimizer.step: non-finite gradient")
    total_sq = torch.zeros((), dtype=torch.float32)
    for _, g in pairs:
        total_sq = total_sq + g.detach().float().pow(2).sum()
    if not math.isfinite(float(torch.sqrt(total_sq))):
        raise NonFiniteTrainingStateError(
            "refusing optimizer.step: non-finite gradient norm "
            "(clip would silently zero it)")

    grad_snap = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in params}

    def restore_grads() -> None:
        with torch.no_grad():
            for p in params:
                saved = grad_snap[id(p)]
                if saved is None:
                    p.grad = None
                elif p.grad is None:
                    p.grad = saved.clone()
                else:
                    p.grad.copy_(saved)

    torch.nn.utils.clip_grad_norm_(params, clip)
    for p, _ in pairs:  # recheck after clipping
        if not bool(torch.isfinite(p.grad).all()):
            restore_grads()
            raise NonFiniteTrainingStateError(
                "gradients became non-finite after clipping")

    # -- transaction: snapshot everything the step may corrupt ---------------
    param_snap = [p.detach().clone() for p in params]
    state_snap = _deep_clone_tree(optimizer.state)
    groups_snap = _snapshot_param_groups(optimizer)

    def restore_params() -> None:
        with torch.no_grad():
            for p, saved in zip(params, param_snap):
                p.copy_(saved)

    def rollback() -> None:
        restore_params()
        _restore_optimizer_state(optimizer, state_snap)
        _restore_param_groups(optimizer, groups_snap)
        restore_grads()

    try:
        optimizer.step()
    except BaseException:
        rollback()
        raise
    for p in params:  # recheck after update
        if not bool(torch.isfinite(p.detach()).all()):
            rollback()
            raise NonFiniteTrainingStateError(
                "parameters became non-finite after optimizer.step")
    if not _tree_is_finite(optimizer.state):
        rollback()
        raise NonFiniteTrainingStateError(
            "optimizer state became non-finite after optimizer.step")
