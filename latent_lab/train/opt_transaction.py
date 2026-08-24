"""Exact, fail-closed optimizer transaction engine (storage-graph based).

This module is a SECOND, independent implementation of the optimizer
transaction used by :func:`guarded_optimizer_step`.  Where the historical
implementation snapshotted optimizer-state tensor leaves with
``detach().clone()`` — normalizing storage offsets and non-dense strides,
severing distinct-view shared-storage aliasing and repeated-reference
semantics, and losing every hidden backing byte outside a visible leaf —
this engine snapshots the *storage graph*:

Design (dense-strided, on-device fast path; no ``torch.save``/CPU
serialization anywhere):
  1. The exact ``optimizer.state`` value tree is traversed retaining the
     original mapping/container keys; the original ``optimizer.state``
     mapping object itself is captured by identity.
  2. Preflight rejects, BEFORE any mutation, every state-tensor case that
     cannot be reconstructed exactly: non-strided/sparse/nested/quantized/
     meta tensors, conj/neg views, autograd-graph-attached tensors,
     non-plain Tensor subclasses, unsupported devices, invalid storage
     bounds, unsupported container/scalar types, excessive depth.
     There is NO silent per-leaf clone fallback: rejection aborts the
     whole step before gradients are clipped.
  3. Unique storages are identified by ``(device type, device index,
     UntypedStorage._cdata)`` — never by ``(data_ptr, nbytes)`` alone —
     so empty storages on different tensors/devices never collapse
     accidentally while true aliases share exactly one slot.
  4. A documented byte budget over unique storages is enforced before any
     clone is allocated (default via
     ``LATENT_LAB_STATE_SNAPSHOT_BUDGET_BYTES``, else
     ``DEFAULT_STATE_SNAPSHOT_BUDGET_BYTES``; optionally per call), so a
     tiny view can never silently snapshot an enormous backing storage.
  5. Each unique full ``UntypedStorage`` is cloned exactly once, on its
     own device under the caller's current stream context.  Every unique
     tensor object records dtype, device, layout, shape, stride,
     storage_offset, requires_grad and its storage slot.  Repeated tensor
     objects and distinct alias/overlap/cross-dtype views are preserved.

Rollback restores the original ``optimizer.state`` mapping object if a
hostile ``step()`` rebound it, removes injected keys, reinstates removed/
replaced entries, and repairs the bytes of whole storage groups —
including bytes outside all visible leaf views:
  * coherent groups (every recorded object alive with identical metadata
    and still referencing the original storage) are restored by ONE full
    storage ``copy_``, keeping every live tensor object and the complete
    alias graph;
  * any other group is reconstructed coherently: each member object is
    re-pointed (identity-preserving ``Tensor.data`` swap onto a working
    clone of the pristine bytes at its exact recorded metadata), so
    dtype/shape/stride/offset/device drift is undone without ever mixing
    or torn-repairing part of an alias group.
Zero-stride/expanded and overlapping views need no per-view ``copy_``.
The original step exception is always preserved; rollback failures are
only ever chained as its ``__cause__``.  After rollback the optimizer is
usable and a retry matches an independent clean control.

Documented limitations (fail-closed, never silent): state/gradient
tensors attached to an autograd graph (e.g. double-backward accumulations)
are rejected preflight; containers beyond plain dict/list/tuple and
immutable scalars are rejected preflight.
"""

from __future__ import annotations

import copy
import os

try:
    import torch
except ImportError:  # pragma: no cover - import-safety without lab group
    torch = None


DEFAULT_STATE_SNAPSHOT_BUDGET_BYTES = 8 << 30      # 8 GiB unique storage
_BUDGET_ENV_VAR = "LATENT_LAB_STATE_SNAPSHOT_BUDGET_BYTES"
_MAX_TREE_DEPTH = 32
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})
_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes)


class OptimizerTransactionError(RuntimeError):
    """Base class for optimizer-transaction failures."""


class OptimizerStateSnapshotError(OptimizerTransactionError):
    """Fail-closed preflight rejection; nothing has been mutated."""


class OptimizerStateBudgetExceeded(OptimizerStateSnapshotError):
    """Unique-storage byte budget exceeded before any clone/mutation."""


class OptimizerRollbackError(OptimizerTransactionError):
    """A rollback failure; only ever chained as the step error's cause."""


def resolve_state_snapshot_budget(explicit=None) -> int | None:
    """Unique-storage byte budget for a transaction snapshot.

    Precedence: explicit argument > env var > module default.  ``None``
    disables the budget entirely (documented escape hatch).
    """
    if explicit is not None:
        return _checked_budget(explicit)
    raw = os.environ.get(_BUDGET_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_STATE_SNAPSHOT_BUDGET_BYTES
    return _checked_budget(int(raw))


def _checked_budget(value) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise OptimizerStateSnapshotError(
            f"state snapshot budget must be >= 0 or None; got {value!r}")
    return value


# ---------------------------------------------------------------------------
# preflight validation (fail-closed, read-only)
# ---------------------------------------------------------------------------

def _preflight_tensor(t, allowed) -> None:
    if type(t) not in allowed:
        raise OptimizerStateSnapshotError(
            f"unsupported tensor subclass {type(t).__name__} in optimizer "
            "state; refusing non-exact snapshot")
    if t.layout != torch.strided:
        raise OptimizerStateSnapshotError(
            f"unsupported layout {t.layout} in optimizer state")
    if t.is_sparse or t.is_nested or t.is_quantized:
        raise OptimizerStateSnapshotError(
            "sparse/nested/quantized optimizer-state tensor cannot be "
            "reconstructed exactly")
    if t.is_conj() or t.is_neg():
        raise OptimizerStateSnapshotError(
            "conj/neg bit views cannot be reconstructed exactly")
    if t.grad_fn is not None:
        raise OptimizerStateSnapshotError(
            "autograd-graph-attached optimizer-state tensor cannot be "
            "reconstructed exactly")
    if t.device.type not in _SUPPORTED_DEVICE_TYPES:
        raise OptimizerStateSnapshotError(
            f"unsupported device {t.device} for exact state snapshot")
    offset = t.storage_offset()
    if offset < 0:
        raise OptimizerStateSnapshotError("negative storage offset")
    strides = t.stride()
    if any(s < 0 for s in strides):
        raise OptimizerStateSnapshotError("negative stride")
    itemsize = t.element_size()
    nbytes = t.untyped_storage().nbytes()
    span_elems = offset if t.numel() == 0 else offset + 1 + sum(
        (size - 1) * s for size, s in zip(t.shape, strides) if size > 0)
    if span_elems * itemsize > nbytes:
        raise OptimizerStateSnapshotError(
            "tensor view exceeds its backing storage bounds")


# ---------------------------------------------------------------------------
# storage-graph registry
# ---------------------------------------------------------------------------

class _TensorRef:
    """Skeleton marker pointing at a recorded tensor object."""
    __slots__ = ("record",)

    def __init__(self, record):
        self.record = record


class _TensorRecord:
    __slots__ = ("obj", "dtype", "device", "layout", "shape", "stride",
                 "offset", "requires_grad", "slot")

    def __init__(self, obj, slot):
        self.obj = obj                   # strong ref to live tensor object
        self.dtype = obj.dtype
        self.device = obj.device
        self.layout = obj.layout
        self.shape = tuple(obj.shape)
        self.stride = tuple(obj.stride())
        self.offset = obj.storage_offset()
        self.requires_grad = bool(obj.requires_grad)
        self.slot = slot

    def metadata_matches(self) -> bool:
        t = self.obj
        return (tuple(t.shape) == self.shape
                and t.dtype == self.dtype
                and t.device == self.device
                and t.layout == self.layout
                and tuple(t.stride()) == self.stride
                and t.storage_offset() == self.offset)


class _StorageSlot:
    __slots__ = ("key", "original", "cdata", "nbytes", "members",
                 "pristine")

    def __init__(self, key, untyped):
        self.key = key
        self.original = untyped          # strong ref to the live storage
        self.cdata = untyped._cdata
        self.nbytes = untyped.nbytes()
        self.members = []
        self.pristine = None             # cloned exactly once, post-budget


class StorageGraphRegistry:
    """Transaction-wide dedup registry over unique full storages.

    State, gradient and parameter trees share one registry so aliased
    storages across them are cloned once and the byte budget is global.
    """

    def __init__(self, *, budget=DEFAULT_STATE_SNAPSHOT_BUDGET_BYTES,
                 allowed_tensor_types=None):
        self.budget = budget
        self.allowed = (allowed_tensor_types
                        if allowed_tensor_types is not None
                        else (torch.Tensor,))
        self.slots = []
        self.records = []
        self._slots_by_key = {}
        self._records_by_id = {}
        self._active_containers = set()
        self.total_unique_bytes = 0
        self.sealed = False

    # -- capture -------------------------------------------------------------

    def walk(self, root, *, depth=0, allowed=None):
        """Read-only traversal producing an exact skeleton of ``root``.

        ``allowed`` restricts acceptable tensor types per tree: the
        optimizer-state value tree only ever accepts plain Tensors,
        while parameter/gradient trees additionally accept
        ``nn.Parameter``. Raises :class:`OptimizerStateSnapshotError`
        (before ANY allocation or mutation) whenever the graph cannot be
        reconstructed exactly.
        """
        if depth > _MAX_TREE_DEPTH:
            raise OptimizerStateSnapshotError(
                f"optimizer-state tree deeper than {_MAX_TREE_DEPTH}")
        if allowed is None:
            allowed = self.allowed
        if torch.is_tensor(root):
            return _TensorRef(self._register(root, allowed))
        node_type = type(root)
        if node_type is dict or node_type is list or node_type is tuple:
            marker = ("c", id(root))
            if marker in self._active_containers:
                raise OptimizerStateSnapshotError(
                    "cyclic optimizer-state container cannot be "
                    "reconstructed exactly")
            self._active_containers.add(marker)
            try:
                if node_type is dict:
                    return {k: self.walk(v, depth=depth + 1, allowed=allowed)
                            for k, v in root.items()}
                if node_type is list:
                    return [self.walk(v, depth=depth + 1, allowed=allowed)
                            for v in root]
                return tuple(self.walk(v, depth=depth + 1, allowed=allowed)
                             for v in root)
            finally:
                self._active_containers.discard(marker)
        if isinstance(root, _SCALAR_TYPES) or \
                isinstance(root, (torch.dtype, torch.device)):
            return root                   # immutable: pass through verbatim
        raise OptimizerStateSnapshotError(
            f"unsupported optimizer-state node of type "
            f"{type(root).__name__}; refusing non-exact snapshot")

    def _register(self, t, allowed) -> _TensorRecord:
        existing = self._records_by_id.get(id(t))
        if existing is not None:
            return existing               # repeated object -> same record
        _preflight_tensor(t, allowed)
        untyped = t.untyped_storage()
        dev = t.device
        key = (dev.type, dev.index, untyped._cdata)
        slot = self._slots_by_key.get(key)
        if slot is None:
            slot = _StorageSlot(key, untyped)
            self._slots_by_key[key] = slot
            self.slots.append(slot)
        record = _TensorRecord(t, slot)
        slot.members.append(record)
        self.records.append(record)
        self._records_by_id[id(t)] = record
        return record

    def seal(self) -> int:
        """Enforce the documented budget, then clone each unique storage
        exactly once (same device, caller's current stream).  Returns the
        total unique bytes snapshotted."""
        if self.sealed:
            return self.total_unique_bytes
        total = sum(slot.nbytes for slot in self.slots)
        if self.budget is not None and total > int(self.budget):
            raise OptimizerStateBudgetExceeded(
                f"optimizer-state snapshot needs {total} unique storage "
                f"bytes across {len(self.slots)} storages, exceeding the "
                f"configured budget of {int(self.budget)} bytes")
        for slot in self.slots:
            slot.pristine = slot.original.clone()
        self.total_unique_bytes = total
        self.sealed = True
        return total

    # -- restore (phase A) ---------------------------------------------------

    def restore_storage_groups(self) -> None:
        """Repair bytes+metadata of every recorded tensor object.

        Coherent slots are fixed by one full-storage ``copy_``; drifted/
        rebound groups are rebuilt coherently onto a working clone of the
        pristine bytes (identity-preserving ``Tensor.data`` swaps), never
        mixing parts of one alias group.  Hidden bytes outside every
        visible leaf view are restored in both paths."""
        for slot in self.slots:
            if not slot.members:
                continue
            coherent = all(
                r.metadata_matches()
                and r.obj.untyped_storage()._cdata == slot.cdata
                and r.obj.untyped_storage().nbytes() == slot.nbytes
                for r in slot.members)
            if coherent:
                slot.original.copy_(slot.pristine)
                continue
            work = slot.pristine.clone()   # pristine stays untouchable
            for r in slot.members:
                view = torch.empty(0, dtype=r.dtype, device=r.device).set_(
                    work, r.offset, r.shape, r.stride)
                with torch.no_grad():
                    r.obj.data = view       # identity-preserving re-point
                    r.obj.requires_grad_(r.requires_grad)


# ---------------------------------------------------------------------------
# structural reconciliation (phase B)
# ---------------------------------------------------------------------------

def _keys_equal(a, b) -> bool:
    """Mapping-key comparison retaining original key objects: tensor keys
    (Parameters) match by identity only; other keys by type+value."""
    if a is b:
        return True
    if torch.is_tensor(a) or torch.is_tensor(b):
        return False
    if type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except Exception:                 # noqa: BLE001 - hostile __eq__
        return False


def _deref(skel):
    return skel.record.obj if isinstance(skel, _TensorRef) else skel


def _materialize(skel):
    """Build fresh containers from a skeleton (original keys/order)."""
    if isinstance(skel, _TensorRef):
        return skel.record.obj
    if isinstance(skel, dict):
        return {k: _materialize(v) for k, v in skel.items()}
    if isinstance(skel, list):
        return [_materialize(v) for v in skel]
    if isinstance(skel, tuple):
        return tuple(_materialize(v) for v in skel)
    return skel                       # immutable scalar


def _skel_matches(live, skel) -> bool:
    """Identity-fast structural check whether ``live`` still equals the
    skeleton (used to decide in-place repair vs replacement)."""
    if isinstance(skel, _TensorRef):
        return live is skel.record.obj
    if isinstance(skel, dict):
        if type(live) is not dict or len(live) != len(skel):
            return False
        pos = list(live.keys())
        for i, (sk, sv) in enumerate(skel.items()):
            if not _keys_equal(pos[i], sk) or \
                    not _skel_matches(live[pos[i]], sv):
                return False
        return True
    if isinstance(skel, list):
        return (type(live) is list and len(live) == len(skel)
                and all(_skel_matches(a, b) for a, b in zip(live, skel)))
    if isinstance(skel, tuple):
        return (type(live) is tuple and len(live) == len(skel)
                and all(_skel_matches(a, b) for a, b in zip(live, skel)))
    if torch.is_tensor(skel):
        return live is skel
    if type(live) is not type(skel):
        return False
    try:
        return bool(live == skel)
    except Exception:                 # noqa: BLE001 - tensor/hostile __eq__
        return live is skel


def _reconcile_dict(live, skel):
    """Repair a live dict IN PLACE toward the skeleton: remove injected
    keys, reinstate removed/replaced entries, canonicalize key objects
    and insertion order while PRESERVING the dict object itself."""
    for k in list(live.keys()):
        if not any(_keys_equal(k, sk) for sk in skel):
            del live[k]               # hostile injected key
    for sk, sv in skel.items():
        matched = next((lk for lk in live if _keys_equal(lk, sk)), None)
        if matched is None:
            live[sk] = _materialize(sv)      # hostile removal/replacement
        elif matched is not sk:
            live[sk] = live.pop(matched)     # original key object back
        else:
            live[sk] = _reconcile(live[sk], sv)
    if list(live.keys()) != list(skel.keys()):
        ordered = {k: live[k] for k in skel}
        live.clear()
        live.update(ordered)          # same dict object, original order
    return live


def _reconcile(live, skel):
    """Return the repaired live node for ``skel``, mutating containers in
    place wherever their structure still allows exact repair."""
    if isinstance(skel, _TensorRef):
        return skel.record.obj        # phase A already repaired its bytes
    if isinstance(skel, dict):
        if type(live) is not dict:
            return _materialize(skel)
        return _reconcile_dict(live, skel)
    if isinstance(skel, list):
        if type(live) is not list or len(live) != len(skel):
            return _materialize(skel)
        for i, sv in enumerate(skel):
            live[i] = _reconcile(live[i], sv)
        return live
    if isinstance(skel, tuple):
        if type(live) is tuple and len(live) == len(skel) and \
                all(_skel_matches(a, b) for a, b in zip(live, skel)):
            return live
        return _materialize(skel)     # immutable: swap wholesale
    return skel                       # immutable scalar: overwrite verbatim


# ---------------------------------------------------------------------------
# optimizer-level transaction
# ---------------------------------------------------------------------------

class OptimizerTransaction:
    """Everything ``optimizer.step()`` may corrupt, captured exactly."""

    def __init__(self, optimizer, params, *,
                 budget=DEFAULT_STATE_SNAPSHOT_BUDGET_BYTES):
        self._optimizer = optimizer
        self._params = list(params)
        reg = StorageGraphRegistry(budget=budget)
        _plain = (torch.Tensor,)
        _params_ok = (torch.Tensor, torch.nn.Parameter)

        # Original optimizer.state MAPPING OBJECT by identity + its exact
        # value tree (original keys retained; values become skeletons;
        # only plain Tensors are acceptable inside state — subclasses are
        # rejected fail-closed).
        self._state_mapping = optimizer.state
        self._state_skel = reg.walk(dict(optimizer.state.items()),
                                    allowed=_plain)
        self._state_keys = list(optimizer.state.keys())

        # Gradients and parameters through the SAME exact engine.
        self._grad_skels = [(p, None if p.grad is None else
                             reg.walk(p.grad, allowed=_params_ok))
                            for p in self._params]
        self._param_skels = [reg.walk(p, allowed=_params_ok)
                             for p in self._params]

        # Budget check + one clone per unique storage (still pre-mutation:
        # nothing live has been touched; clip has not run yet).
        self._registry = reg
        self._total_unique_bytes = reg.seal()

    @property
    def total_unique_bytes(self) -> int:
        return self._total_unique_bytes

    def rollback(self) -> None:
        """Restore everything exactly.  Raises OptimizerRollbackError on
        failure (callers chain it as the step exception's cause)."""
        try:
            reg = self._registry
            # Phase A: bytes+metadata of every recorded object, every
            # storage group (parameters, gradients, optimizer state).
            reg.restore_storage_groups()

            # Phase B: gradient attributes (assignment validates against
            # the Parameter, so only AFTER phase A restored exact dtypes).
            for p, gskel in self._grad_skels:
                p.grad = None if gskel is None else _deref(gskel)

            # Phase B: optimizer.state mapping identity + contents + order.
            opt = self._optimizer
            if opt.state is not self._state_mapping:
                opt.state = self._state_mapping   # undo hostile rebind
            _reconcile_dict(self._state_mapping, self._state_skel)

            # Parameters need no container work: their objects were never
            # replaced and phase A restored their bytes/metadata; the
            # param_groups topology is reinstated separately by identity.
        except BaseException as e:                # noqa: BLE001
            raise OptimizerRollbackError(
                f"optimizer-state rollback failed: {e}") from e


# ---------------------------------------------------------------------------
# param_groups topology (identity-preserving)
# ---------------------------------------------------------------------------

def snapshot_param_groups(optimizer) -> tuple:
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


def restore_param_groups(optimizer, snap) -> None:
    """Rebuild ``param_groups`` IN PLACE from the snapshot: exact original
    group count/order, the original group dicts with their original field
    values, and each group's original live Parameter objects in their
    exact original order — plus the original OUTER list object: if a
    hostile step rebound ``optimizer.param_groups`` to a different list,
    the snapshot's list is reinstated by assignment before its contents
    are restored in place. Hostile additions are undone by removal,
    hostile removals/reorderings by reinsertion of the original objects;
    no Parameter object is ever replaced."""
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
            if params_ref is None:        # non-list container: restore
                group["params"] = params_order
            else:
                if group.get("params") is not params_ref:
                    group["params"] = params_ref   # hostile replacement
                params_ref[:] = params_order       # identities + order
        elif "params" in group:
            del group["params"]
        rebuilt.append(group)
    live[:] = rebuilt               # drop injected groups, restore order


def optimizer_owned_params(optimizer) -> list:
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
