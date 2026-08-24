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
    parameters, the complete optimizer state (every unique backing
    storage's full bytes AND every tensor leaf's exact
    shape/dtype/device/layout/stride/offset metadata, structurally
    exact) and the full param-group topology (the original outer list
    object, group count/order, fields, per-group Parameter identities
    and order) back bit-exactly, without ever masking or replacing the
    original step exception. The optimizer state is snapshotted as a
    dense-strided on-device storage GRAPH — unique UntypedStorages are
    cloned once each (under a documented byte budget) so aliasing,
    overlapping, expanded and cross-dtype views plus hidden backing
    bytes survive rollback exactly; unsupported state graphs are
    rejected fail-closed BEFORE any mutation. The transaction set is
    derived from the optimizer's own groups, so Parameters omitted by
    the caller stay covered.
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


class UnsupportedOptimizerStateError(CheckpointError):
    """The optimizer-state value graph cannot be snapshotted/restored
    exactly (non-strided/sparse/nested/quantized/meta layout, unsupported
    tensor subclass or device, invalid storage bounds, unidentifiable
    storage). Raised fail-closed BEFORE any mutation; there is no silent
    per-leaf clone fallback."""


class StateStorageBudgetExceededError(CheckpointError):
    """Total unique optimizer-state backing-storage bytes exceeded the
    configured snapshot budget. Raised before any mutation/cloning."""


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

# Unique-storage snapshot budget: the maximum total number of backing
# bytes (sum over UNIQUE UntypedStorages, not per leaf) that one guarded
# step may clone for rollback. Override per call via
# ``guarded_optimizer_step(..., state_byte_budget=...)``. Enforced BEFORE
# any cloning/mutation so a tiny view can never silently snapshot an
# enormous backing storage.
DEFAULT_STATE_STORAGE_BYTE_BUDGET = 1 << 32  # 4 GiB

# Devices on which full UntypedStorage clone/restore is exact and
# same-device; anything else (xla, meta, private backends) is rejected
# fail-closed before mutation.
_STATE_SNAPSHOT_DEVICES = frozenset({"cpu", "cuda", "mps"})


class _TensorLeaf:
    """Exact observable metadata of one unique state tensor object."""

    __slots__ = ("dtype", "device", "shape", "strides", "offset",
                 "requires_grad", "slot")

    def __init__(self, t, slot):
        self.dtype = t.dtype
        self.device = t.device
        self.shape = tuple(t.shape)
        self.strides = tuple(t.stride())
        self.offset = t.storage_offset()
        self.requires_grad = bool(t.requires_grad)
        self.slot = slot


class _StorageSlot:
    """One unique backing UntypedStorage within the optimizer-state
    graph, identified collision-safely by device plus storage identity
    (``_cdata``). Cloned exactly once, whole, on its own device."""

    __slots__ = ("key", "device", "nbytes", "cdata", "sample", "clone",
                 "target")

    def __init__(self, key, device, nbytes, cdata, sample):
        self.key = key
        self.device = device
        self.nbytes = nbytes
        self.cdata = cdata
        self.sample = sample
        self.clone = None
        self.target = None


class _StateGraphSnapshot:
    """Structurally exact snapshot of ``optimizer.state`` as a value
    tree over storage slots.

    * ``state_obj`` — the ORIGINAL ``optimizer.state`` mapping object;
      rollback reinstates this identity if a hostile step rebound it.
    * ``root`` — tagged skeleton: ``("dict", items)``/``("list", ...)``/
      ``("tuple", ...)`` containers with original keys/order,
      ``("leaf", _TensorLeaf)`` at every tensor position (the SAME leaf
      record for repeated tensor objects), ``("val", deepcopy)`` for
      non-tensor values.
    * ``slots`` — unique ``_StorageSlot`` records with their whole-
      storage pristine clones.
    """

    __slots__ = ("state_obj", "root", "slots")


def _validate_state_tensor(t):
    """Fail-closed preflight of one candidate state tensor: plain strided
    dense tensors on supported devices whose visible span lies inside an
    identifiably unique backing storage. Anything else must be rejected
    before mutation — there is no silent per-leaf clone fallback."""
    if type(t) is not torch.Tensor:
        raise UnsupportedOptimizerStateError(
            f"unsupported optimizer-state tensor subclass "
            f"{type(t).__name__}; cannot reconstruct exactly")
    if t.layout != torch.strided or t.is_sparse:
        raise UnsupportedOptimizerStateError(
            f"unsupported optimizer-state layout {t.layout} "
            "(sparse/nested/non-strided); cannot reconstruct exactly")
    if t.is_quantized:
        raise UnsupportedOptimizerStateError(
            "quantized optimizer-state tensors are unsupported")
    if t.device.type not in _STATE_SNAPSHOT_DEVICES or \
            t.device.type == "meta":
        raise UnsupportedOptimizerStateError(
            f"unsupported optimizer-state device {t.device}")
    if any(s < 0 for s in t.stride()):
        raise UnsupportedOptimizerStateError(
            "negative-stride optimizer-state view is unsupported")
    try:
        u = t.untyped_storage()
        cdata = u._cdata
    except Exception as e:
        raise UnsupportedOptimizerStateError(
            f"optimizer-state storage is unidentifiable: {e}") from e
    if cdata is None:
        raise UnsupportedOptimizerStateError(
            "optimizer-state storage has no _cdata identity")
    if u.device != t.device:
        raise UnsupportedOptimizerStateError(
            f"tensor device {t.device} differs from backing-storage "
            f"device {u.device}")
    isz = t.element_size()
    off = t.storage_offset()
    if off < 0:
        raise UnsupportedOptimizerStateError(
            "negative optimizer-state storage offset")
    nbytes = u.nbytes()
    if t.numel() > 0:
        span = sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
        if (off + span + 1) * isz > nbytes:
            raise UnsupportedOptimizerStateError(
                "optimizer-state view exceeds its storage bounds "
                f"(end={(off + span + 1) * isz} > nbytes={nbytes})")
    elif off * isz > nbytes:
        raise UnsupportedOptimizerStateError(
            "empty optimizer-state view offset outside storage bounds")
    return u, cdata


def _storage_slot_key(device, cdata):
    return (device.type, device.index if device.index is not None else -1,
            cdata)


def _snapshot_state_tree(value, leaves, slots):
    """Build the tagged skeleton for one state value; validate every
    tensor leaf fail-closed and deduplicate storages by identity."""
    if torch.is_tensor(value):
        prior = leaves.get(id(value))
        if prior is not None:
            return prior
        u, cdata = _validate_state_tensor(value)
        key = _storage_slot_key(value.device, cdata)
        slot = slots.get(key)
        if slot is None:
            slot = slots[key] = _StorageSlot(
                key, value.device, u.nbytes(), cdata, u)
        node = ("leaf", _TensorLeaf(value, slot))
        leaves[id(value)] = node
        return node
    if isinstance(value, dict):
        return ("dict", {k: _snapshot_state_tree(v, leaves, slots)
                         for k, v in value.items()})
    if isinstance(value, list):
        return ("list", [_snapshot_state_tree(v, leaves, slots)
                         for v in value])
    if isinstance(value, tuple):
        return ("tuple", tuple(_snapshot_state_tree(v, leaves, slots)
                               for v in value))
    return ("val", copy.deepcopy(value))


def snapshot_optimizer_state_graph(optimizer, byte_budget=None):
    """Snapshot ``optimizer.state`` exactly as a dense-strided, on-device
    storage graph, BEFORE anything is mutated.

    Traverses the exact state value tree retaining original mapping /
    container keys, rejects unsupported tensor graphs fail-closed
    (raising :class:`UnsupportedOptimizerStateError` or
    :class:`StateStorageBudgetExceededError` while parameters,
    gradients, state and groups are still untouched), then clones each
    unique backing :class:`~torch.UntypedStorage` exactly once on its
    own device/stream under the configured total byte budget
    (``DEFAULT_STATE_STORAGE_BYTE_BUDGET`` when ``byte_budget`` is
    None). Repeated tensor objects, distinct aliasing/overlapping views,
    expanded zero-stride views and cross-dtype views all stay attached
    to their shared storage slot, and hidden backing bytes outside all
    visible leaves are captured by the whole-storage clones."""
    if byte_budget is None:
        budget = float(DEFAULT_STATE_STORAGE_BYTE_BUDGET)
    else:
        budget = float(byte_budget)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(f"invalid state_byte_budget {byte_budget!r}")
    snap = _StateGraphSnapshot()
    snap.state_obj = optimizer.state
    leaves, slots = {}, {}
    if not hasattr(snap.state_obj, "items"):
        raise UnsupportedOptimizerStateError(
            f"optimizer.state mapping of type "
            f"{type(snap.state_obj).__name__} cannot be traversed")
    try:
        snap.root = ("dict", {
            k: _snapshot_state_tree(v, leaves, slots)
            for k, v in snap.state_obj.items()})
        total = sum(slot.nbytes for slot in slots.values())
        if total > budget:
            raise StateStorageBudgetExceededError(
                f"unique optimizer-state storage totals {total} bytes, "
                f"exceeding the snapshot budget {int(budget)}")
    except UnsupportedOptimizerStateError:
        raise
    except StateStorageBudgetExceededError:
        raise
    except Exception as e:
        raise UnsupportedOptimizerStateError(
            f"optimizer-state graph cannot be snapshotted exactly: {e}"
            ) from e
    # Clone AFTER full validation+budget: each unique whole storage once,
    # on its own device/stream.
    try:
        for slot in slots.values():
            cloned = slot.sample.clone()
            if cloned.device != slot.device or \
                    cloned.nbytes() != slot.nbytes:
                raise RuntimeError(
                    f"storage clone mismatch for slot {slot.key}")
            slot.clone = cloned
    except Exception as e:
        raise UnsupportedOptimizerStateError(
            f"optimizer-state storage cloning failed: {e}") from e
    snap.slots = slots
    return snap


def _tree_is_finite(obj) -> bool:
    """Finiteness of a live state tree; anything whose values cannot be
    checked exactly (sparse/nested/quantized layouts, failing kernels)
    counts as NOT finite so the transaction rolls back instead of
    crashing mid-recheck."""
    if torch.is_tensor(obj):
        if obj.layout != torch.strided or obj.is_quantized:
            return False
        try:
            return not obj.is_floating_point() or \
                bool(torch.isfinite(obj).all())
        except Exception:
            return False
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
    """Roll ``optimizer.state`` back to a :class:`_StateGraphSnapshot`
    exactly, coherently per alias group.

    The original ``optimizer.state`` mapping object is reinstated first
    if a hostile step rebinding it. Injected keys are removed and
    removed/replaced entries rebuilt from the skeleton. Each unique
    storage slot is then repaired as ONE group — never torn: the whole
    pristine backing-storage bytes are copied back exactly once (into
    the original storage when it is still intact, preserving its
    identity and every compatible live tensor object; into the pristine
    clone otherwise), which restores hidden bytes outside all visible
    views and keeps aliasing/overlap/cross-dtype relationships. Every
    torn leaf reference is rebound onto that same group storage via
    ``Tensor.set_`` with the exact recorded dtype/shape/stride/offset/
    requires_grad, and a repeated tensor object whose ANY reference was
    torn comes back as ONE shared shell at ALL of its references.
    Zero-stride expanded views ride the same whole-storage path — no
    per-view ``copy_`` anywhere."""
    state = optimizer.state
    if state is not snap.state_obj:
        optimizer.state = snap.state_obj       # undo hostile rebinding
        state = snap.state_obj

    # Group targets first: the ORIGINAL storage while it is still
    # intact (identity-preserving), otherwise the pristine clone.
    for slot in snap.slots.values():
        sample = slot.sample
        slot.target = sample if (
            sample.device == slot.device
            and sample._cdata == slot.cdata
            and sample.nbytes() == slot.nbytes) else slot.clone

    broken_records = set()                     # torn tensor-object records
    all_paths = []                             # (leaf, live, setter)
    shells = {}                                # id(leaf) -> shared shell

    def shell_for(leaf):
        shell = shells.get(id(leaf))
        if shell is None:
            with torch.no_grad():
                shell = torch.empty(0, dtype=leaf.dtype,
                                    device=leaf.device)
                shell.set_(leaf.slot.target, leaf.offset, leaf.shape,
                           leaf.strides)
                shell.requires_grad_(leaf.requires_grad)
            shells[id(leaf)] = shell
        return shell

    def mark_broken(leaf):
        broken_records.add(id(leaf))

    def walk(skel, live, setter):
        kind, payload = skel
        if kind == "leaf":
            leaf = payload
            all_paths.append((leaf, live, setter))
            if type(live) is not torch.Tensor or \
                    leaf_torn(live, leaf):
                mark_broken(leaf)
            return
        if kind == "val":
            v = payload
            if type(live) is type(v) and \
                    isinstance(v, (bool, int, float, complex, str,
                                   bytes, type(None))) and live == v:
                return
            setter(copy.deepcopy(v))
            return
        if kind == "dict":
            if not isinstance(live, dict):
                setter(_materialize(skel))
                return
            for k in list(live.keys()):
                if k not in payload:
                    del live[k]                # hostile injected key
            for k, sub in payload.items():
                walk(sub, live.get(k),
                     lambda nv, k=k: live.__setitem__(k, nv))
            return
        if kind == "list":
            if type(live) is not list:
                setter(_materialize(skel))
                return
            n = len(payload)
            del live[n:]                       # hostile tail entries
            while len(live) < n:
                live.append(None)
            for i, sub in enumerate(payload):
                walk(sub, live[i],
                     lambda nv, i=i: live.__setitem__(i, nv))
            return
        # tuples are immutable: always rebuild exactly
        setter(_materialize(skel))

    def leaf_torn(live, leaf):
        u = live.untyped_storage()
        return not (live.dtype == leaf.dtype
                    and tuple(live.shape) == leaf.shape
                    and tuple(live.stride()) == leaf.strides
                    and live.storage_offset() == leaf.offset
                    and live.device == leaf.device
                    and bool(live.requires_grad) == leaf.requires_grad
                    and u.device == leaf.slot.device
                    and u._cdata == leaf.slot.cdata
                    and u.nbytes() == leaf.slot.nbytes)

    def _materialize(skel):
        """Rebuild a fresh subtree from a skeleton node (fresh containers
        and deep-copied non-tensor values; tensor references come back
        as the shared group shells)."""
        kind, payload = skel
        if kind == "val":
            return copy.deepcopy(payload)
        if kind == "dict":
            return {k: _materialize(sub) for k, sub in payload.items()}
        if kind == "list":
            return [_materialize(sub) for sub in payload]
        if kind == "tuple":
            return tuple(_materialize(sub) for sub in payload)
        leaf = payload                          # rebuilt leaf reference
        mark_broken(leaf)
        return shell_for(leaf)

    root = snap.root[1]
    for k in list(state.keys()):
        if k not in root:
            del state[k]                       # hostile injected param
    for k, sub in root.items():
        walk(sub, state.get(k), lambda nv, k=k: state.__setitem__(k, nv))

    # Slot phase: one whole-storage byte-exact write per group, then
    # rebind every torn reference onto that group storage.
    for slot in snap.slots.values():
        slot.target.copy_(slot.clone)
    for leaf, _live, setter in all_paths:
        if id(leaf) in broken_records:
            setter(shell_for(leaf))
        # else: compatible live object kept; bytes restored above


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


def guarded_optimizer_step(optimizer, loss, params, clip_norm,
                           *, state_byte_budget=None) -> None:
    """Step only when everything is finite; roll back completely otherwise.

    Transactional contract:
      * PRE-STEP (before any mutation): finite loss, positive finite clip,
        finite parameters, finite gradients and a finite total gradient
        norm. Any rejection here performs no step and mutates nothing.
      * The optimizer state is then snapshotted as an exact dense-strided
        on-device storage graph BEFORE ``clip_grad_norm_`` runs: unique
        backing ``UntypedStorage``s are cloned whole (once each, same
        device/stream) under ``state_byte_budget`` total unique bytes
        (:data:`DEFAULT_STATE_STORAGE_BYTE_BUDGET` when None), preserving
        aliasing/overlap/expanded/cross-dtype views and hidden backing
        bytes. Unsupported state graphs — non-strided/sparse/nested/
        quantized/meta layouts, unsupported subclasses/devices, invalid
        bounds or unidentifiable storages — and budget overruns are
        rejected fail-closed (:class:`UnsupportedOptimizerStateError` /
        :class:`StateStorageBudgetExceededError`) while parameters,
        gradients, state and groups remain untouched; clipping never
        starts on a failed preflight and there is no per-leaf clone
        fallback.
      * Gradients are rechecked after clipping; a post-clip rejection also
        restores the pre-clip gradients.
      * Every trainable parameter, every pre-clip gradient (including the
        ``None`` versus tensor distinction and exact bytes), the exact
        optimizer-state storage graph and the complete ``param_groups``
        topology (the original outer list object, group count/order,
        every group field, and each group's live Parameter identities in
        exact order) are snapshotted before ``optimizer.step()``. If the
        step raises, or any post-step parameter or optimizer-state tensor
        is non-finite, all of it is restored exactly: full backing-
        storage bytes (including regions outside all visible leaf views)
        are copied back once per alias group so live tensor objects,
        alias graphs and repeated references survive coherently; torn or
        replaced groups are reconstructed whole via ``set_`` with exact
        recorded metadata; injected fields/keys are removed; a hostile
        rebind of ``optimizer.state`` itself is undone by reinstating the
        original mapping object. Rollback can therefore never raise a
        size/dtype mismatch. The original step exception is then
        re-raised intact (rollback failures chain onto it as __cause__,
        never replacing it). The optimizer remains usable after rollback:
        a retry matches an independent clean control.
      * The caller-supplied ``params`` iterable is never trusted as
        exhaustive: the authoritative transaction/finite-check set is the
        deterministic first-occurrence union of the supplied Parameters
        with every Parameter owned by the optimizer's param_groups, so no
        optimizer-owned Parameter can be updated by ``optimizer.step()``
        without snapshot, post-check and rollback coverage.
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

    # -- transaction snapshots: complete BEFORE clipping mutates grads ------
    state_graph = snapshot_optimizer_state_graph(optimizer, state_byte_budget)
    param_snap = [p.detach().clone() for p in params]
    grad_snap = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in params}
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

    def restore_params() -> None:
        with torch.no_grad():
            for p, saved in zip(params, param_snap):
                p.copy_(saved)

    def rollback() -> None:
        restore_params()
        _restore_optimizer_state(optimizer, state_graph)
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
    for p in params:  # recheck after update
        if not bool(torch.isfinite(p.detach()).all()):
            rollback()
            raise NonFiniteTrainingStateError(
                "parameters became non-finite after optimizer.step")
    if not _tree_is_finite(optimizer.state):
        rollback()
        raise NonFiniteTrainingStateError(
            "optimizer state became non-finite after optimizer.step")
