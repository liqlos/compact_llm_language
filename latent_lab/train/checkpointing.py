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
    clipping and after the update; any post-step corruption rolls the
    parameters, the complete optimizer state (the full reachable
    tensor/storage graph: every tensor's values AND its exact
    shape/dtype/device/layout/stride/storage-offset metadata, unique
    backing storages cloned across their whole extent, alias views kept
    sharing one storage, repeated references kept identical, and the
    original ``optimizer.state`` mapping object) and the full param-group
    topology (the original outer list
    object, group count/order, fields, per-group Parameter identities
    and order) back bit-exactly, without ever masking or replacing the
    original step exception. A state graph that cannot be captured
    exactly (tensor subclasses, meta tensors, unsupported devices,
    unsupported layouts, conjugate/neg bits, cross-dtype shared storage,
    or a unique-storage byte total above the configured budget) fails
    closed during preflight — before clipping or stepping. The
    transaction set is derived from the optimizer's own groups, so
    Parameters omitted by the caller stay covered.

    PERFORMANCE BOUNDARY: the transaction never serializes state to CPU
    and never copies per-leaf; it pays exactly ONE on-device clone per
    unique backing storage plus per-tensor metadata/finiteness reads.
    Paid Vast/CUDA training remains blocked until a target-CUDA A/B
    benchmark shows <= 5% median AND p95 step-time regression versus the
    unguarded baseline; no CUDA measurements are claimed anywhere here.
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


class AdapterBundleError(CheckpointError):
    """An adapter bundle failed structural or content validation."""


class AdapterBundleIdentityError(AdapterBundleError):
    """Bundle was produced for a different (model_id, revision)."""


class AdapterBundleSchemaError(AdapterBundleError):
    """Bundle violated the metadata schema (key/type/shape/dtype)."""


class OptimizerStateGraphError(CheckpointError):
    """The reachable optimizer-state tensor/storage graph cannot be
    captured and restored exactly; the transaction refuses to start."""


BUNDLE_KIND = "latent_lab.adapter_bundle"
BUNDLE_FORMAT_VERSION = 1

# -- stable OptimizerStateGraphError reason codes ---------------------------
# Every fail-closed rejection carries one bracketed code as the first token
# of its message so callers/tests can dispatch on it stably.
REASON_TENSOR_SUBCLASS = "tensor-subclass"
REASON_META_DEVICE = "meta-device"
REASON_UNSUPPORTED_DEVICE = "unsupported-device"
REASON_SPARSE_LAYOUT = "sparse-layout"
REASON_NESTED_LAYOUT = "nested-layout"
REASON_MKLDNN_LAYOUT = "mkldnn-layout"
REASON_UNSUPPORTED_LAYOUT = "unsupported-layout"
REASON_QUANTIZED = "quantized-tensor"
REASON_CONJUGATE_VIEW = "conjugate-view"
REASON_NEG_VIEW = "neg-bit-view"
REASON_CROSS_DTYPE_STORAGE = "cross-dtype-shared-storage"
REASON_BUDGET_EXCEEDED = "state-byte-budget-exceeded"
REASON_POST_STEP_UNINSPECTABLE = "post-step-uninspectable-state"

# Device types on which an exact whole-extent storage clone is known to
# work. Anything else fails closed during preflight (documented, stable
# ``unsupported-device`` reason) rather than risking a lossy transaction.
_SUPPORTED_STATE_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})

# Default ceiling on the SUM of unique backing-storage bytes the state
# snapshot may clone (16 GiB). This accommodates honest fp32 AdamW state —
# exp_avg + exp_avg_sq (+ max_exp_avg_sq for amsgrad) for ~1B parameters,
# including hidden storage extents — while bounding the damage of a tiny
# view silently aliasing a huge backing store. Override per call with the
# ``state_byte_budget`` keyword.
DEFAULT_STATE_BYTE_BUDGET_BYTES = 1 << 34


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

def _collect_state_tensors(obj, out, active, done) -> None:
    """Collect every tensor reachable from ``obj`` through dict/list/tuple
    containers; cyclic container graphs raise (they cannot be snapshotted
    deterministically). Shared substructures are visited once."""
    if torch.is_tensor(obj):
        out.append(obj)
        return
    oid = id(obj)
    if oid in done:
        return
    if oid in active:
        raise OptimizerStateGraphError(
            "cyclic container graph in optimizer state")
    active.add(oid)
    try:
        if isinstance(obj, dict):
            for v in obj.values():
                _collect_state_tensors(v, out, active, done)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect_state_tensors(v, out, active, done)
    finally:
        active.discard(oid)
        done.add(oid)


def _capturable_rejection_reason(t, *, allow_parameter: bool = False):
    """Stable reason code (or ``None``) for why tensor ``t`` cannot be
    captured and restored EXACTLY as the same semantic type.

    Tensor subclasses are rejected because the snapshot machinery rebuilds
    plain ``torch.Tensor`` leaves; a subclass instance would silently lose
    its semantic type on rollback (``allow_parameter`` relaxes this ONLY
    for live Parameters, whose identity — not class — the parameter
    rollback preserves)."""
    if type(t) is not torch.Tensor:
        if allow_parameter and type(t) is torch.nn.Parameter:
            pass
        else:
            return REASON_TENSOR_SUBCLASS
    if t.is_sparse:
        return REASON_SPARSE_LAYOUT
    if t.is_nested:
        return REASON_NESTED_LAYOUT
    if t.layout != torch.strided:
        if t.layout == torch.mkldnn:
            return REASON_MKLDNN_LAYOUT
        return REASON_UNSUPPORTED_LAYOUT
    if t.is_quantized:
        return REASON_QUANTIZED
    if t.is_meta:
        return REASON_META_DEVICE
    if t.device.type not in _SUPPORTED_STATE_DEVICE_TYPES:
        return REASON_UNSUPPORTED_DEVICE
    if t.is_conj():
        return REASON_CONJUGATE_VIEW
    if t.is_neg():
        return REASON_NEG_VIEW
    return None


def _storage_identity_key(t) -> tuple:
    """Device-safe unique-storage identity.

    Keyed by ``(device type, device index, storage _cdata)`` — never by
    ``data_ptr`` alone: raw addresses from different device address spaces
    can collide numerically, while ``_cdata`` is process-global per
    allocation and keeps even distinct EMPTY storages separate."""
    dev = t.device
    return (dev.type, dev.index, t.untyped_storage()._cdata)


def _require_capturable_state_graph(tensors) -> None:
    """Fail closed when the reachable tensor/storage graph cannot be
    captured and restored exactly: tensor subclasses, meta tensors,
    unsupported devices, unsupported layouts (sparse/nested/quantized/
    mkldnn), conjugate/neg bitwise views, or ONE backing storage viewed
    at TWO dtypes."""
    seen_dtype = {}
    for t in tensors:
        reason = _capturable_rejection_reason(t)
        if reason is not None:
            raise OptimizerStateGraphError(
                f"[{reason}] optimizer-state tensor (shape={tuple(t.shape)}, "
                f"dtype={t.dtype}, device={t.device}, layout={t.layout}) "
                f"cannot be captured and restored exactly")
        key = _storage_identity_key(t)
        known = seen_dtype.get(key)
        if known is None:
            seen_dtype[key] = t.dtype
        elif known is not t.dtype:
            raise OptimizerStateGraphError(
                f"[{REASON_CROSS_DTYPE_STORAGE}] cross-dtype shared storage "
                f"in optimizer state ({known} vs {t.dtype}) cannot be "
                f"captured exactly")


def _unique_storage_bytes(tensors) -> int:
    """Overflow-safe sum of FULL backing-storage extents over UNIQUE
    storages (Python ints cannot overflow; each storage counted once via
    its device-safe identity key). A 2-element view over a 4 GiB backing
    store therefore counts 4 GiB — what a snapshot must actually clone."""
    seen, total = set(), 0
    for t in tensors:
        key = _storage_identity_key(t)
        if key in seen:
            continue
        seen.add(key)
        total += int(t.untyped_storage().nbytes())
    return total


def _require_state_bytes_within_budget(tensors, budget) -> int:
    """Fail closed BEFORE any clone when unique storage bytes exceed the
    configured snapshot budget; returns the counted total."""
    counted = _unique_storage_bytes(tensors)
    if counted > budget:
        raise OptimizerStateGraphError(
            f"[{REASON_BUDGET_EXCEEDED}] optimizer-state graph needs "
            f"{counted} unique storage bytes, exceeding the configured "
            f"state_byte_budget of {budget} bytes; refusing to clone")
    return counted


def _tensor_values_finite(t) -> bool:
    """Finiteness including COMPLEX tensors: NaN/Inf in either component
    makes a complex tensor non-finite (plain ``is_floating_point()``
    checks miss complex dtypes entirely)."""
    if t.is_floating_point() or t.is_complex():
        return bool(torch.isfinite(t).all())
    return True


def _require_state_finite(tensors) -> None:
    """Reject pre-existing non-finite optimizer-state values (complex
    included) BEFORE any mutation."""
    for i, t in enumerate(tensors):
        if not _tensor_values_finite(t):
            kind = "complex " if t.is_complex() else ""
            raise NonFiniteTrainingStateError(
                f"refusing optimizer.step: non-finite {kind}"
                f"optimizer-state tensor #{i} (dtype={t.dtype})")


def _rebuild_view(storage, template) -> torch.Tensor:
    """Fresh detached tensor viewing ``storage`` with the template's exact
    dtype/device/shape/strides/storage-offset (no value copy)."""
    return torch.empty(0, dtype=template.dtype,
                       device=template.device).set_(
        storage, template.storage_offset(),
        tuple(template.shape), tuple(template.stride()))


def _capture_optimizer_state(state, *, byte_budget: int) -> tuple:
    """Structural snapshot of ``optimizer.state`` that preserves the FULL
    reachable storage graph: one cloned storage per unique backing
    storage cloned across its WHOLE extent (not just the visible region),
    every tensor rebuilt as an exact view (dtype/device/layout/shape/
    strides/storage-offset), views of one storage kept sharing the
    snapshot storage, and repeated references to the same Tensor object
    kept as ONE snapshot object. Returns ``(original_mapping_object,
    snapshot_tree)`` so rollback can also undo a hostile rebind of the
    mapping itself. Raises ``OptimizerStateGraphError`` before ANY clone
    when the graph is not exactly capturable (subclasses, meta tensors,
    unsupported devices/layouts, conjugate/neg bits, cross-dtype shared
    storage) or when its unique storage bytes exceed ``byte_budget``;
    raises ``NonFiniteTrainingStateError`` for pre-existing non-finite
    state values, complex included — all strictly before any mutation."""
    tensors: list = []
    _collect_state_tensors(state, tensors, set(), set())
    _require_capturable_state_graph(tensors)
    _require_state_bytes_within_budget(tensors, byte_budget)
    _require_state_finite(tensors)
    snap_tensors: dict = {}
    snap_storages: dict = {}

    def rec(obj):
        if torch.is_tensor(obj):
            got = snap_tensors.get(id(obj))
            if got is not None:
                return got
            st = obj.untyped_storage()
            key = _storage_identity_key(obj)
            snap_st = snap_storages.get(key)
            if snap_st is None:
                snap_st = st.clone()
                snap_storages[key] = snap_st
            out = _rebuild_view(snap_st, obj)
            snap_tensors[id(obj)] = out
            return out
        if isinstance(obj, dict):
            return {k: rec(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rec(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(rec(v) for v in obj)
        return obj

    return state, rec(state)


def _tree_is_finite(obj) -> bool:
    if torch.is_tensor(obj):
        return _tensor_values_finite(obj)
    if isinstance(obj, dict):
        return all(_tree_is_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_tree_is_finite(v) for v in obj)
    return True


def _inplace_exact_restore_possible(live, snap) -> bool:
    """True only when copying ``snap`` through the live tensor object
    cannot change any observable metadata — i.e. both are strided tensors
    with identical shape, dtype, device, layout, strides and storage
    offset. Anything else must be rebound from the pristine snapshot."""
    if snap.layout != torch.strided or live.layout != torch.strided:
        return False
    return (tuple(live.shape) == tuple(snap.shape)
            and live.dtype == snap.dtype
            and live.device == snap.device
            and tuple(live.stride()) == tuple(snap.stride())
            and live.storage_offset() == snap.storage_offset())


def _restore_optimizer_state(optimizer, snap) -> None:
    """Roll ``optimizer.state`` back to ``snap`` (from
    ``_capture_optimizer_state``) exactly, storage graph included.

    Every snapshot leaf is materialised against a fresh clone of its
    snapshot storage — one clone per unique storage, so aliasing inside
    the snapshot survives restoration while the pristine snapshot itself
    is never aliased by live state. A live tensor object that still has
    the snapshot's dtype/device and a strided layout is rebound IN PLACE
    via ``Tensor.set_``, keeping object identity while restoring its
    exact geometry (non-zero storage offsets, zero/non-dense strides,
    full backing-storage extent); anything else (swapped dtype/device,
    replaced or missing objects) is replaced by a freshly built exact
    view. Repeated snapshot references are reinstated as the SAME
    installed object. Hostilely injected keys are removed, removed keys
    reinstated in kind, and a hostile rebind of the ``optimizer.state``
    mapping itself is undone before its contents are restored in place.
    ``Tensor.copy_`` is never used here, so rollback cannot cast dtypes,
    coerce shapes or fail on a size mismatch."""
    state_obj, snap_tree = snap
    if optimizer.state is not state_obj:
        optimizer.state = state_obj          # undo hostile mapping rebind
    state = optimizer.state
    fresh_storages: dict = {}
    installed: dict = {}

    def fresh_storage(snap_t):
        key = _storage_identity_key(snap_t)
        st = fresh_storages.get(key)
        if st is None:
            st = snap_t.untyped_storage().clone()
            fresh_storages[key] = st
        return st

    def restore_leaf(live, snap_t):
        memo = installed.get(id(snap_t))
        if memo is not None:
            return memo                      # repeated reference: same obj
        if (torch.is_tensor(live)
                and _capturable_rejection_reason(live) is None
                and live.dtype == snap_t.dtype
                and live.device == snap_t.device):
            # in-place rebind ONLY for a live leaf that is itself an exact-
            # inspectable plain dense tensor; anything the step injected
            # (sparse/meta/quantized/subclass/conj/neg) is replaced by a
            # fresh exact view instead of being set_ through.
            with torch.no_grad():
                live.set_(fresh_storage(snap_t), snap_t.storage_offset(),
                          tuple(snap_t.shape), tuple(snap_t.stride()))
            out = live
        else:
            out = _rebuild_view(fresh_storage(snap_t), snap_t)
        installed[id(snap_t)] = out
        return out

    def rec(live, snap_v):
        if torch.is_tensor(snap_v):
            return restore_leaf(live, snap_v)
        if isinstance(snap_v, dict):
            if isinstance(live, dict):
                for k in list(live.keys()):
                    if k not in snap_v:
                        del live[k]          # hostile injection: undone
                for k, v in snap_v.items():
                    live[k] = rec(live.get(k), v)
                return live                  # mapping object kept alive
            return {k: rec(None, v) for k, v in snap_v.items()}
        if isinstance(snap_v, list):
            if isinstance(live, list) and len(live) == len(snap_v):
                for i, v in enumerate(snap_v):
                    live[i] = rec(live[i], v)
                return live
            return [rec(None, v) for v in snap_v]
        if isinstance(snap_v, tuple):
            return tuple(rec(None, v) for v in snap_v)
        return snap_v

    rec(state, snap_tree)


def _optimizer_owned_params(optimizer) -> list:
    """Authoritative pre-step Parameter set: every tensor owned by the
    optimizer's ``param_groups``, in deterministic first-occurrence order.
    The caller-supplied iterable is never trusted as exhaustive."""
    owned, seen = [], set()
    for group in optimizer.param_groups:
        if isinstance(group, dict):
            plist = group.get("params", ())
        else:  # exotic non-dict group objects
            plist = getattr(group, "params", ())
        if not isinstance(plist, (list, tuple)):
            continue
        for p in plist:
            if torch.is_tensor(p) and id(p) not in seen:
                seen.add(id(p))
                owned.append(p)
    return owned


def _snapshot_param_groups(optimizer) -> tuple:
    """Structural snapshot of ``param_groups`` sufficient to rebuild the
    exact pre-step topology, INCLUDING the identity of the original outer
    ``param_groups`` list object itself. Per group it records the live
    group dict (by reference), deep copies of every non-``params`` field,
    whether a ``params`` entry existed, the live params list object (by
    reference) and a copy of its Parameter order. Parameter objects
    themselves are never copied or replaced."""
    groups_obj = optimizer.param_groups
    outer_ref = groups_obj if isinstance(groups_obj, list) else None
    snap = []
    for group in groups_obj:
        fields = {
            k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in group.items() if k != "params"}
        had_params = "params" in group
        plist = group["params"] if had_params else ()
        params_ref = plist if isinstance(plist, list) else None
        snap.append((group, fields, had_params, params_ref, list(plist)))
    return outer_ref, snap


def _restore_param_groups(optimizer, snap) -> None:
    """Rebuild ``param_groups`` IN PLACE from the snapshot: exact original
    group count/order, the original group dicts with their original field
    values, and each group's original live Parameter objects in their
    exact original order — plus the original OUTER list object: if a
    hostile step rebound ``optimizer.param_groups`` to a different list,
    the snapshot's list is reinstated by assignment before its contents
    are restored in place, so any optimizer/subclass reference to the
    original list stays authoritative. Hostile additions (groups or
    per-group params) are undone by removal, hostile removals/reordering
    by reinsertion of the original objects; no Parameter object is ever
    replaced."""
    outer_ref, group_snaps = snap
    live = optimizer.param_groups
    if outer_ref is not None and live is not outer_ref:
        optimizer.param_groups = outer_ref     # undo hostile outer rebind
        live = outer_ref
    present = {id(g) for g in live}
    rebuilt = []
    for group, fields, had_params, params_ref, params_order in group_snaps:
        if id(group) not in present:      # hostile group removal: reinsert
            live.append(group)
            present.add(id(group))
        for key in list(group.keys()):
            if key != "params" and key not in fields:
                del group[key]            # field injected mid-step
        for key, val in fields.items():
            group[key] = (val.detach().clone() if torch.is_tensor(val)
                          else copy.deepcopy(val))
        if had_params:
            if params_ref is None:        # non-list container: restore contents
                group["params"] = params_order
            else:
                if group.get("params") is not params_ref:
                    group["params"] = params_ref   # hostile list replacement
                params_ref[:] = params_order       # identities + order, in place
        elif "params" in group:
            del group["params"]
        rebuilt.append(group)
    live[:] = rebuilt               # drop injected groups, restore order


def _require_post_step_transaction_valid(params, optimizer) -> None:
    """Post-step validation that can never silently pass an uninspectable
    transaction: every live Parameter and every reachable optimizer-state
    tensor must still be exactly inspectable (no sparse/meta/quantized/
    nested/subclass/conj/neg swap) AND finite, complex values included.
    Raises ``OptimizerStateGraphError`` (stable
    ``[post-step-uninspectable-state]`` reason) for uninspectable leaves
    and ``NonFiniteTrainingStateError`` for non-finite values; callers
    must roll back the whole transaction on ANY exception from here."""
    for p in params:
        reason = _capturable_rejection_reason(p, allow_parameter=True)
        if reason is not None:
            raise OptimizerStateGraphError(
                f"[{REASON_POST_STEP_UNINSPECTABLE}] parameter became "
                f"uninspectable ({reason}) after optimizer.step")
        if not _tensor_values_finite(p.detach()):
            raise NonFiniteTrainingStateError(
                "parameters became non-finite after optimizer.step")
    tensors: list = []
    _collect_state_tensors(optimizer.state, tensors, set(), set())
    for t in tensors:
        reason = _capturable_rejection_reason(t)
        if reason is not None:
            raise OptimizerStateGraphError(
                f"[{REASON_POST_STEP_UNINSPECTABLE}] optimizer-state tensor "
                f"became uninspectable ({reason}) after optimizer.step")
    for t in tensors:
        if not _tensor_values_finite(t):
            raise NonFiniteTrainingStateError(
                "optimizer state became non-finite after optimizer.step")


def guarded_optimizer_step(optimizer, loss, params, clip_norm, *,
                           state_byte_budget=None) -> None:
    """Step only when everything is finite; roll back completely otherwise.

    ``state_byte_budget``: optional ceiling (int bytes) on the SUM of
    unique backing-storage bytes the state snapshot may clone. ``None``
    uses ``DEFAULT_STATE_BYTE_BUDGET_BYTES`` (16 GiB — sized for honest
    fp32 AdamW state up to ~1B parameters including amsgrad and hidden
    extents). A tiny view over a huge backing store counts its FULL
    extent, so pathological aliasing cannot silently clone unbounded
    memory; exceedance fails closed with the stable
    ``[state-byte-budget-exceeded]`` reason before clipping/stepping.
    Must be ``None`` or a non-negative ``int`` (``bool`` rejected);
    invalid values raise ``ValueError`` before any mutation.

    Transactional contract:
      * PRE-STEP (before any mutation): finite loss, positive finite clip,
        finite parameters, finite gradients and a finite total gradient
        norm. Any rejection here performs no step and mutates nothing.
      * STATE-GRAPH PREFLIGHT: the reachable optimizer-state tensor/
        storage graph must be exactly capturable — no tensor subclasses,
        meta tensors, unsupported devices or layouts (sparse/nested/
        quantized/mkldnn), no conjugate/neg views, every strided dense
        tensor at one dtype per device-keyed unique storage, unique
        storage bytes within budget, and all state values finite
        (complex NaN/Inf included) — or the whole call raises
        ``OptimizerStateGraphError`` / ``NonFiniteTrainingStateError``
        BEFORE clipping or stepping, claiming no lossy transaction.
      * COMPLETE SNAPSHOT BEFORE MUTATION: state, gradients, parameters
        AND the full param-group topology/deepcopy all complete BEFORE
        ``clip_grad_norm_`` runs; if any snapshot operation fails (e.g. a
        hostile group field whose deepcopy raises), gradients were never
        clipped and no object was touched.
      * Gradients are rechecked after clipping; a post-clip rejection also
        restores the pre-clip gradients.
      * Every trainable parameter, every pre-clip gradient (including the
        ``None`` versus tensor distinction and exact bytes), an exact
        snapshot of the optimizer state's full storage graph and the
        complete ``param_groups``
        topology (the original outer list object, group count/order,
        every group field, and each group's live Parameter identities in
        exact order) are snapshotted before ``optimizer.step()``. The
        state snapshot clones each unique backing storage across its
        WHOLE extent — exactly ONE on-device clone per unique storage,
        keyed by ``(device type/index, _cdata)``, never by ``data_ptr``
        alone — rebuilds every leaf with exact shape/dtype/device/layout/
        strides/storage-offset, keeps views of one storage sharing one
        snapshot storage, keeps repeated references to the same Tensor
        object as ONE object, and preserves the original
        ``optimizer.state`` mapping object. If the step raises, or any
        post-step parameter or optimizer-state tensor is non-finite
        (complex included) or has become uninspectable (sparse/meta/
        quantized/subclass/...), or validation itself crashes, ALL of it
        — parameters, gradients, the complete state storage graph
        (aliasing, extents, injected fields removed) and the full
        param-group structure down to outer-list identity — is restored
        exactly while keeping the live Parameter identities; only then is
        the stable guard error raised (``NonFiniteTrainingStateError``,
        or ``OptimizerStateGraphError`` chaining the validation failure),
        so a failed validation can never leave an applied update behind.
        A state leaf whose dtype/device was swapped mid-step is replaced
        by a fresh exact view rather than copied through, so rollback
        itself can never raise a size/dtype mismatch. A hostile rebind of
        the optimizer.state mapping is undone. The original step
        exception is always re-raised intact after an exact rollback
        (rollback failures chain onto it as __cause__, never replacing
        it). The optimizer remains usable after rollback.
      * The caller-supplied ``params`` iterable is never trusted as
        exhaustive: the authoritative transaction/finite-check set is the
        deterministic first-occurrence union of the supplied Parameters
        with every Parameter owned by the optimizer's param_groups, so no
        optimizer-owned Parameter can be updated by ``optimizer.step()``
        without snapshot, post-check and rollback coverage.
      * PERFORMANCE BOUNDARY: no CPU serialization, no per-leaf copies;
        exactly one on-device clone per unique backing storage. Paid
        Vast/CUDA training remains blocked until a target-CUDA A/B shows
        <= 5% median AND p95 step-time regression; none are claimed.
    """
    _require_torch()
    # -- configuration validation: nothing has been mutated yet --------------
    if state_byte_budget is None:
        budget = DEFAULT_STATE_BYTE_BUDGET_BYTES
    elif (isinstance(state_byte_budget, bool)
          or not isinstance(state_byte_budget, int)
          or state_byte_budget < 0):
        raise ValueError(
            "state_byte_budget must be None or a non-negative int "
            f"(bytes); got {state_byte_budget!r}")
    else:
        budget = state_byte_budget
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
    params, seen = [], set()
    for p in supplied + _optimizer_owned_params(optimizer):
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
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

    # -- transaction preflight + COMPLETE snapshot: still nothing mutated ----
    # The optimizer-state storage graph must be exactly capturable BEFORE
    # clipping or stepping: an exotic graph that could not be restored
    # verbatim must fail closed here rather than claim a lossy transaction.
    state_snap = _capture_optimizer_state(optimizer.state, byte_budget=budget)

    grad_snap = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in params}
    param_snap = [p.detach().clone() for p in params]
    groups_snap = _snapshot_param_groups(optimizer)

    def restore_grads() -> None:
        with torch.no_grad():
            for p in params:
                saved = grad_snap[id(p)]
                if saved is None:
                    p.grad = None
                elif p.grad is None or not _inplace_exact_restore_possible(
                        p.grad, saved):
                    p.grad = saved.clone()   # hostile grad rebinding: rebind
                else:
                    p.grad.copy_(saved)

    torch.nn.utils.clip_grad_norm_(params, clip)
    for p, _ in pairs:  # recheck after clipping
        if not bool(torch.isfinite(p.grad).all()):
            restore_grads()
            raise NonFiniteTrainingStateError(
                "gradients became non-finite after clipping")

    # -- transaction: everything is snapshotted; step may corrupt it --------

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
    except BaseException as step_error:
        try:
            rollback()
        except BaseException as rollback_error:  # never mask the original
            raise step_error from rollback_error
        raise
    # -- POST-STEP VALIDATION IS TRANSACTIONAL -------------------------------
    # A successful step whose validation fails (non-finite values,
    # uninspectable injected leaves, or a crashing check) must never leave
    # the applied update behind: roll back the WHOLE transaction first,
    # then raise the stable guard error.
    try:
        _require_post_step_transaction_valid(params, optimizer)
    except BaseException as validation_error:
        if isinstance(validation_error,
                      (OptimizerStateGraphError, NonFiniteTrainingStateError)):
            guard = validation_error
        else:
            guard = OptimizerStateGraphError(
                "post-step validation crashed after a successful "
                f"optimizer.step ({validation_error!r}); rolled back the "
                "whole transaction")
            guard.__cause__ = validation_error
        try:
            rollback()
        except BaseException as rollback_error:
            guard.__cause__ = rollback_error
            raise guard
        raise guard
