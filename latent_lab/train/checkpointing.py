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
    parameters, the complete optimizer state (every tensor's values AND
    its exact shape/dtype/device/layout/stride/storage-offset metadata
    PLUS the full reachable storage graph — shared-storage alias
    relationships between leaves, views with non-zero offsets or
    non-dense strides, and repeated tensor objects) and the full
    param-group topology (the original outer list object, group
    count/order, fields, per-group Parameter identities and order) back
    bit-exactly, without ever masking or replacing the original step
    exception. The state snapshot runs a fail-closed preflight BEFORE
    gradient clipping and mutates nothing on rejection: unsupported
    tensors (sparse/quantized/nested/meta, subclasses, unsupported
    devices, conj/neg views), invalid storage bounds or a unique-storage
    byte total over the configurable budget
    (DEFAULT_STATE_STORAGE_BUDGET_BYTES) raise
    OptimizerStateSnapshotError instead of falling back to per-leaf
    clones; rollback also reinstates the original optimizer.state mapping
    object if a hostile step rebound it. The transaction set is derived
    from the optimizer's own groups, so Parameters omitted by the caller
    stay covered.
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


class OptimizerStateSnapshotError(CheckpointError):
    """optimizer.state held a case the exact snapshot refuses (fail-closed).

    Raised BEFORE any mutation (in particular before gradient clipping):
    unsupported tensor kinds (sparse/quantized/nested/meta, conj/neg
    views), unsupported tensor subclasses or devices, storage bounds that
    cannot hold the recorded views, or a unique-storage byte total over
    the configured budget. There is no silent per-leaf clone fallback.
    """


# Fail-closed ceiling on the TOTAL bytes of unique optimizer-state
# storages snapshotted per guarded step (configurable per call via
# ``state_storage_budget_bytes``). Prevents a tiny view from silently
# snapshotting an enormous backing storage.
DEFAULT_STATE_STORAGE_BUDGET_BYTES = 1 << 30

# Devices whose untyped storages the exact snapshot supports; anything
# else (meta and exotic accelerators included) is rejected preflight.
_SUPPORTED_STATE_DEVICE_TYPES = frozenset({"cpu", "cuda"})


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
    """Exact deep copy of a tensor tree (values; metadata of dense leaves).

    Retained for non-transactional callers; the optimizer transaction uses
    the storage-graph-exact snapshot/restore pair below instead, because
    ``detach().clone()`` normalizes storage offsets to 0, re-homes views
    into their own compacted storages and severs shared-storage alias
    relationships between state leaves."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _deep_clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_clone_tree(v) for v in obj)
    return obj


class _StateStorageSlot:
    """One reachable untyped storage behind the optimizer-state tree.

    ``raw`` is a byte-exact clone of the FULL storage (a flat uint8
    tensor owning its own storage), so every view over it — whatever its
    storage offset or strides — can be reconstructed bit-exactly, and
    leaves that shared this storage at snapshot time can be re-homed onto
    one common restored storage afterwards."""

    __slots__ = ("raw", "nbytes", "device")

    def __init__(self, raw, nbytes, device):
        self.raw = raw
        self.nbytes = nbytes
        self.device = device


class _StateTensorLeaf:
    """Snapshot record of one distinct tensor object in the state tree.

    Strided leaves are recorded as (storage slot, dtype, device, size,
    stride, storage offset, requires_grad). Unsupported cases are
    REJECTED before any mutation (fail-closed); there is no fallback
    path. Repeated references to the same object share one leaf node,
    so restore reproduces the reference graph, not just values."""

    __slots__ = ("slot", "dtype", "device", "size", "stride", "offset",
                 "requires_grad")

    def __init__(self, *, slot, dtype, device, size, stride, offset,
                 requires_grad):
        self.slot = slot
        self.dtype = dtype
        self.device = device
        self.size = size
        self.stride = stride
        self.offset = offset
        self.requires_grad = requires_grad


class _OptimizerStateSnapshot:
    """Structurally complete snapshot of ``optimizer.state``: the tree of
    containers/tensor-leaf nodes/original non-tensor objects plus every
    distinct reachable storage slot with its full raw bytes, AND the
    identity of the original ``optimizer.state`` mapping object itself."""

    __slots__ = ("slots", "slot_leaves", "tree", "mapping")

    def __init__(self):
        self.slots = []
        self.slot_leaves = []
        self.tree = None
        self.mapping = None


def _storage_key(ust, empty_token=None) -> tuple:
    """Collision-safe unique-storage key: device plus true storage
    identity (``_cdata`` where supported), never ``(data_ptr, nbytes)``
    alone — distinct empty storages share ``data_ptr 0``/0 bytes but have
    distinct identities and must never collapse into one slot. Without
    ``_cdata``, non-empty storages fall back to their (live-unique)
    ``data_ptr`` and empty ones to the caller-supplied per-snapshot
    token (snapshot side: always-fresh ⇒ never merged; planning side:
    ``None`` ⇒ conservatively collapsible, which can only force the
    coherent rebuild path, never a wrong in-place claim)."""
    dev = (str(ust.device.type), ust.device.index)
    cdata = getattr(ust, "_cdata", None)
    if cdata is not None:
        return (dev, "cdata", int(cdata))
    if ust.nbytes() == 0:
        return (dev, "empty", empty_token)
    return (dev, "ptr", ust.data_ptr())


def _reject_unsupported(reason: str):
    raise OptimizerStateSnapshotError(
        f"optimizer state cannot be snapshotted exactly ({reason}); "
        "refusing before any mutation (fail-closed)")


def _preflight_state_tensor(t) -> None:
    """Fail-closed rejection of every tensor case the exact storage-graph
    snapshot cannot reconstruct bit-exactly. Called during the snapshot,
    which itself runs BEFORE gradient clipping and before any mutation."""
    if type(t) is not torch.Tensor and type(t) is not torch.nn.Parameter:
        _reject_unsupported(f"tensor subclass {type(t).__name__}")
    if t.is_sparse or getattr(t, "is_sparse_csr", False) \
            or t.layout != torch.strided:
        _reject_unsupported(f"non-strided layout {t.layout}")
    if t.is_nested or t.is_quantized:
        _reject_unsupported("nested/quantized tensor")
    if t.device.type == "meta" or \
            t.device.type not in _SUPPORTED_STATE_DEVICE_TYPES:
        _reject_unsupported(f"unsupported device {t.device}")
    if t.is_conj() or t.is_neg():
        _reject_unsupported("conjugate/negative bit view")
    try:
        ust = t.untyped_storage()
        nbytes, udev = ust.nbytes(), ust.device
        shape, strides = tuple(t.shape), tuple(t.stride())
        offset = t.storage_offset()
        esize = t.element_size()
    except Exception as e:  # noqa: BLE001 - exotic tensor: fail closed
        raise OptimizerStateSnapshotError(
            f"optimizer state tensor has no usable untyped storage "
            f"({e}); refusing before any mutation (fail-closed)") from e
    if udev != t.device:
        _reject_unsupported(f"storage device {udev} != tensor {t.device}")
    if any(s < 0 for s in strides) or offset < 0:
        _reject_unsupported("negative stride/storage offset")
    last = offset
    for size, stride in zip(shape, strides):
        if size > 0:
            last += (size - 1) * stride
    needed = last * esize + (esize if t.numel() > 0 else 0)
    if needed > nbytes:
        _reject_unsupported(
            f"view needs {needed} storage bytes beyond {nbytes}")


def _full_storage_bytes(tensor, storage, nbytes):
    """Byte-exact owned copy of the whole storage behind ``tensor``."""
    if nbytes == 0:
        return None
    src = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    src = src.set_(storage, 0, (nbytes,), (1,))
    return src.clone()


def _snapshot_state_tree(state, budget_bytes) -> _OptimizerStateSnapshot:
    """Capture the FULL reachable storage graph of ``optimizer.state``.

    Every distinct tensor object becomes one leaf node carrying its exact
    dtype/device/layout/shape/strides/storage_offset/requires_grad plus
    the index of its underlying storage slot; each slot keeps a byte
    clone of the complete storage. Views (offset/non-dense strides),
    shared-storage aliases between different leaves and repeated
    references to one object are therefore all preserved exactly, which
    ``detach().clone()`` per leaf cannot do.

    Fail-closed: unsupported tensor kinds (sparse/quantized/nested/meta,
    subclasses, unsupported devices, conj/neg views), storage bounds that
    cannot hold a recorded view, or a unique-storage byte total over
    ``budget_bytes`` raise ``OptimizerStateSnapshotError`` — there is no
    silent per-leaf clone fallback. Storages are identified by device +
    ``_cdata`` (never ``(data_ptr, nbytes)``, which collapses empties).
    The original mapping object is retained by identity so rollback can
    reinstate it if a hostile step rebound ``optimizer.state``."""
    snap = _OptimizerStateSnapshot()
    snap.mapping = state
    slot_of_key = {}
    leaf_of_obj = {}
    budget_used = 0
    empty_counter = iter(range(1 << 62))

    def capture_leaf(t, path):
        nonlocal budget_used
        node = leaf_of_obj.get(id(t))
        if node is not None:
            snap.slot_leaves[node.slot].append((path, node))
            return node
        _preflight_state_tensor(t)
        ust = t.untyped_storage()
        key = _storage_key(ust, empty_token=next(empty_counter))
        slot_idx = slot_of_key.get(key)
        if slot_idx is None:
            nbytes = ust.nbytes()
            budget_used += nbytes
            if budget_used > budget_bytes:
                _reject_unsupported(
                    f"unique storage bytes {budget_used} exceed the "
                    f"configured snapshot budget {budget_bytes}")
            slot_idx = len(snap.slots)
            snap.slots.append(_StateStorageSlot(
                _full_storage_bytes(t, ust, nbytes), nbytes, t.device))
            slot_of_key[key] = slot_idx
            snap.slot_leaves.append([])
        node = _StateTensorLeaf(
            slot=slot_idx, dtype=t.dtype, device=t.device,
            size=tuple(t.shape), stride=tuple(t.stride()),
            offset=t.storage_offset(), requires_grad=bool(t.requires_grad))
        snap.slot_leaves[slot_idx].append((path, node))
        leaf_of_obj[id(t)] = node
        return node

    def walk(obj, path):
        if torch.is_tensor(obj):
            return capture_leaf(obj, path)
        if isinstance(obj, dict):
            return {k: walk(v, path + (k,)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, path + (i,)) for i, v in enumerate(obj)]
        if isinstance(obj, tuple):
            return tuple(walk(v, path + (i,))
                         for i, v in enumerate(obj))
        return obj

    snap.tree = walk(state, ())
    return snap


def _lookup_live_path(root, path):
    """Resolve a snapshot path through the LIVE tree; None on any
    structural mismatch (missing key/index, wrong container type)."""
    cur = root
    for seg in path:
        if isinstance(seg, str):
            if not isinstance(cur, dict) or seg not in cur:
                return None
            cur = cur[seg]
        else:
            if not isinstance(cur, (list, tuple)) or seg >= len(cur):
                return None
            cur = cur[seg]
    return cur if torch.is_tensor(cur) else None


def _leaf_matches_live(live, leaf, slot) -> bool:
    """True when restoring THROUGH ``live`` cannot change observable
    metadata and its storage still spans exactly the snapshotted bytes."""
    if live.layout != torch.strided or live.dtype != leaf.dtype \
            or tuple(live.shape) != leaf.size \
            or tuple(live.stride()) != leaf.stride \
            or live.storage_offset() != leaf.offset \
            or live.device != slot.device \
            or bool(live.requires_grad) != leaf.requires_grad:
        return False
    try:
        ust = live.untyped_storage()
    except Exception:  # noqa: BLE001 - exotic live tensor: force rebuild
        return False
    return ust.nbytes() == slot.nbytes and ust.device == slot.device


def _plan_state_slot_restore(state, snap) -> list:
    """Decide, per storage slot, how rollback restores it.

    A slot restores IN PLACE through the live objects only when EVERY
    referencing leaf still exists at its path with bit-identical
    metadata and all of them still share ONE live storage of exactly the
    snapshotted size/device (identified collision-safely by device +
    ``_cdata``), uniquely claimed by this slot. Then the pristine bytes
    are copied back wholesale — live tensor objects and their aliasing
    stay untouched. Any structural divergence (replaced leaf, swapped
    metadata or requires_grad, broken/merged/split sharing, missing
    entry) forces a REBUILD: fresh storage from the raw byte clone,
    every leaf of the group re-homed onto it with its exact metadata —
    never a torn per-view repair of one alias group."""
    plans = [("rebuild", None) for _ in snap.slots]
    claimed = set()
    for idx, slot in enumerate(snap.slots):
        live_map = {}
        ok = True
        for path, leaf in snap.slot_leaves[idx]:
            live = _lookup_live_path(state, path)
            if live is None or not _leaf_matches_live(live, leaf, slot):
                ok = False
                break
            live_map[path] = live
        if not ok:
            continue
        keys = {_storage_key(live.untyped_storage())
                for live in live_map.values()}
        if len(keys) != 1:
            continue
        key = next(iter(keys))
        if key in claimed:
            continue
        plans[idx] = ("inplace", live_map)
        claimed.add(key)
    with torch.no_grad():
        for idx, (kind, _) in enumerate(plans):
            if kind != "inplace":
                continue
            some = next(iter(plans[idx][1].values()))
            dst = torch.empty(0, dtype=torch.uint8, device=snap.slots[idx].device)
            dst = dst.set_(some.untyped_storage(), 0,
                           (snap.slots[idx].nbytes,), (1,))
            if snap.slots[idx].nbytes:
                dst.copy_(snap.slots[idx].raw)
    return plans


class _StateRestoreContext:
    """Lazy per-rollback resources: one freshly materialized storage per
    rebuilt slot and memoized rebuilds so repeated references materialize
    back into ONE object."""

    def __init__(self, snap, plans):
        self.snap = snap
        self.plans = plans
        self.storages = {}
        self.memo = {}

    def fresh_storage(self, idx):
        if idx not in self.storages:
            slot = self.snap.slots[idx]
            holder = torch.empty(slot.nbytes, dtype=torch.uint8,
                                 device=slot.device)
            if slot.nbytes:
                holder.copy_(slot.raw)
            self.storages[idx] = holder.untyped_storage()
        return self.storages[idx]

    def rebuild(self, leaf):
        if id(leaf) not in self.memo:
            t = torch.empty(0, dtype=leaf.dtype,
                            device=self.snap.slots[leaf.slot].device)
            t = t.set_(self.fresh_storage(leaf.slot), leaf.offset,
                       leaf.size, leaf.stride)
            if leaf.requires_grad:
                t.requires_grad_(True)
            self.memo[id(leaf)] = t
        return self.memo[id(leaf)]


def _materialize_state_node(node, ctx, path):
    """Rebuild a snapshot subtree as concrete objects: in-place slots
    yield the (byte-restored) LIVE tensor objects, rebuild slots yield
    storage-faithful reconstructions over one fresh shared storage,
    non-tensor originals are rebound by identity."""
    if isinstance(node, _StateTensorLeaf):
        kind, live_map = ctx.plans[node.slot]
        if kind == "inplace":
            return live_map[path]
        return ctx.rebuild(node)
    if isinstance(node, dict):
        return {k: _materialize_state_node(v, ctx, path + (k,))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_materialize_state_node(v, ctx, path + (i,))
                for i, v in enumerate(node)]
    if isinstance(node, tuple):
        return tuple(_materialize_state_node(v, ctx, path + (i,))
                     for i, v in enumerate(node))
    return node


def _restore_optimizer_state(optimizer, snap) -> None:
    """Roll ``optimizer.state`` back to the snapshot exactly.

    Values AND the full storage graph are restored: every tensor's
    dtype/device/layout/shape/strides/storage_offset/requires_grad,
    shared-storage alias relationships among leaves, repeated tensor
    objects (one object stays one object) and all non-tensor fields by
    original identity. If a hostile step rebound the ``optimizer.state``
    ATTRIBUTE to a different mapping object, the original snapshot-time
    mapping is reinstated by assignment first, so any reference to it
    stays authoritative; hostilely injected keys are then dropped and
    hostile removals/replacements overwritten with the snapshot
    reconstruction. Restoration never routes a mismatched copy through
    ``Tensor.copy_``: incompatible subtrees are rebound wholesale, so
    rollback itself cannot raise a shape/dtype/size error."""
    if optimizer.state is not snap.mapping:
        optimizer.state = snap.mapping    # undo hostile attribute rebind
    state = snap.mapping
    for key in list(state.keys()):
        if key not in snap.tree:
            del state[key]
    plans = _plan_state_slot_restore(state, snap)
    ctx = _StateRestoreContext(snap, plans)
    for key, node in snap.tree.items():
        live_entry = state.get(key) if hasattr(state, "get") else None
        if not isinstance(live_entry, dict) or \
                not isinstance(node, dict):
            state[key] = _materialize_state_node(node, ctx, (key,))
            continue
        for name in list(live_entry.keys()):
            if name not in node:
                del live_entry[name]
        for name, child in node.items():
            live_entry[name] = _materialize_state_node(
                child, ctx, (key, name))


def _tree_is_finite(obj) -> bool:
    if torch.is_tensor(obj):
        return not obj.is_floating_point() or bool(torch.isfinite(obj).all())
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


def guarded_optimizer_step(optimizer, loss, params, clip_norm, *,
                           state_storage_budget_bytes=None) -> None:
    """Step only when everything is finite; roll back completely otherwise.

    Transactional contract:
      * PRE-STEP (before any mutation): finite loss, positive finite clip,
        finite parameters, finite gradients and a finite total gradient
        norm. Any rejection here performs no step and mutates nothing.
      * The optimizer-state snapshot (with its fail-closed preflight) and
        the ``param_groups`` snapshot COMPLETE BEFORE GRADIENT CLIPPING,
        so a snapshot/preflight rejection leaves parameters, gradients,
        state, param_groups and the ``optimizer.state`` mapping identity
        untouched. Preflight rejects, before ANY mutation, every state
        tensor case the exact snapshot cannot reconstruct: sparse/
        quantized/nested/meta tensors, tensor subclasses, unsupported
        devices, conjugate/negative-bit views, invalid storage bounds,
        or a unique-storage byte total over
        ``state_storage_budget_bytes`` (default:
        ``DEFAULT_STATE_STORAGE_BUDGET_BYTES``; the budget bounds the
        TOTAL unique backing-storage bytes cloned per step so a tiny view
        cannot silently snapshot an enormous backing storage). There is
        no silent per-leaf clone fallback.
      * Gradients are rechecked after clipping; a post-clip rejection also
        restores the pre-clip gradients.
      * Every trainable parameter, every pre-clip gradient (including the
        ``None`` versus tensor distinction and exact bytes), a
        storage-graph-exact snapshot of the optimizer state and the
        complete ``param_groups`` topology (the original outer list
        object, group count/order, every group field, and each group's
        live Parameter identities in exact order) are snapshotted before
        ``optimizer.step()``. If the step raises, or any post-step
        parameter or optimizer-state tensor is non-finite, all of it —
        parameters, gradients, complete optimizer state with every
        tensor's exact shape/dtype/device/layout/stride/storage-offset
        metadata, shared-storage alias relationships between state
        leaves, repeated tensor objects, injected fields removed, and the
        original ``optimizer.state`` MAPPING object reinstated if a
        hostile step rebound it — and the full param-group structure down
        to outer-list identity are restored exactly while keeping the
        live Parameter identities; a state leaf whose metadata was swapped
        mid-step is rebound from the pristine snapshot rather than copied
        through, so rollback itself can never raise a size/dtype mismatch.
        The original step exception is then re-raised intact (rollback
        failures chain onto it as __cause__, never replacing it). The
        optimizer remains usable after rollback.
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

    grad_snap = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in params}

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

    # -- transaction snapshot BEFORE clipping: preflight failure here has
    # mutated NOTHING (parameters, gradients, state, groups, identities).
    budget = (DEFAULT_STATE_STORAGE_BUDGET_BYTES
              if state_storage_budget_bytes is None
              else int(state_storage_budget_bytes))
    if budget <= 0:
        raise OptimizerStateSnapshotError(
            f"state storage budget must be positive, got "
            f"{state_storage_budget_bytes!r}")
    state_snap = _snapshot_state_tree(optimizer.state, budget)
    groups_snap = _snapshot_param_groups(optimizer)

    torch.nn.utils.clip_grad_norm_(params, clip)
    for p, _ in pairs:  # recheck after clipping
        if not bool(torch.isfinite(p.grad).all()):
            restore_grads()
            raise NonFiniteTrainingStateError(
                "gradients became non-finite after clipping")

    param_snap = [p.detach().clone() for p in params]

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
    for p in params:  # recheck after update
        if not bool(torch.isfinite(p.detach()).all()):
            rollback()
            raise NonFiniteTrainingStateError(
                "parameters became non-finite after optimizer.step")
    if not _tree_is_finite(optimizer.state):
        rollback()
        raise NonFiniteTrainingStateError(
            "optimizer state became non-finite after optimizer.step")
