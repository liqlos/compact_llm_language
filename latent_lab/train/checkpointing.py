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
    its exact shape/dtype/device/layout/stride/storage-offset metadata,
    plus the full reachable STORAGE graph: shared-storage aliasing among
    leaves, repeated Tensor-object references, cross-dtype views, full
    backing-storage extents — captured alias-exactly or the transaction
    fails closed preflight) and the full param-group topology (the
    original outer list object, group count/order, fields, per-group
    Parameter identities and order) back bit-exactly, without ever
    masking or replacing the original step exception; the original
    optimizer.state mapping object (and every per-parameter state
    mapping) is reinstated if a hostile step rebinds it. The transaction
    set is derived from the optimizer's own groups, so Parameters
    omitted by the caller stay covered.
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


class UnsupportedStorageGraphError(CheckpointError):
    """A reachable tensor/storage graph cannot be captured exactly; the
    transaction fails closed before any mutation instead of snapshotting
    lossily."""


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

class _AliasCloneCtx:
    """Dedup context shared by one transaction's snapshot pass: repeated
    references to the SAME Tensor object map to one copied object, and
    distinct leaves viewing ONE live storage map into ONE cloned storage
    (so relative offsets, strides and cross-dtype views survive exactly).
    Independent zero-capacity storages are never merged — empty tensors
    can all report data_ptr()==0 without being aliases."""

    def __init__(self) -> None:
        self.by_obj = {}
        self.by_storage = {}


def _clone_tensor_exact(t, ctx):
    """Rebuild ``t`` bit-exactly as a view over a copy of its FULL
    backing storage (values, dtype, device, layout, shape, strides and
    storage offset; sharing per ``_AliasCloneCtx``). Unsupported graphs
    raise UnsupportedStorageGraphError so callers fail closed."""
    got = ctx.by_obj.get(id(t))
    if got is not None:
        return got
    if t.layout != torch.strided:
        raise UnsupportedStorageGraphError(
            f"cannot capture {t.layout} layout tensor exactly")
    try:
        storage = t.untyped_storage()
        new_storage = None
        if storage.nbytes() > 0:
            key = (storage.data_ptr(), storage.nbytes())
            new_storage = ctx.by_storage.get(key)
            if new_storage is None:
                new_storage = storage.clone()
                ctx.by_storage[key] = new_storage
        else:
            # zero-capacity storages carry no bytes and must not be
            # merged by a coincidental data_ptr()==0: keep them separate
            new_storage = storage.clone()
        out = torch.empty(0, dtype=t.dtype, device=t.device)
        out.set_(new_storage, t.storage_offset(),
                 tuple(t.shape), tuple(t.stride()))
    except UnsupportedStorageGraphError:
        raise
    except Exception as e:  # noqa: BLE001 - exotic graph: fail closed
        raise UnsupportedStorageGraphError(
            f"cannot capture tensor storage graph exactly: {e!r}") from e
    out.requires_grad_(False)
    ctx.by_obj[id(t)] = out
    return out


def _deep_clone_tree(obj, ctx=None):
    """Exact deep copy of a tensor tree that preserves the FULL reachable
    STORAGE graph: every leaf's values/dtype/device/layout/shape/strides/
    storage-offset are reproduced bit-exactly, distinct leaves sharing one
    backing storage stay shared in the copy (each storage is cloned once,
    at full extent, even when only an offset view is reachable), repeated
    references to one Tensor object map to one copied object, and
    cross-dtype views over one storage remain cross-dtype views over one
    storage. Graphs that cannot be captured this way raise
    UnsupportedStorageGraphError instead of silently normalizing."""
    if ctx is None:
        ctx = _AliasCloneCtx()
    if torch.is_tensor(obj):
        return _clone_tensor_exact(obj, ctx)
    if isinstance(obj, dict):
        return {k: _deep_clone_tree(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_clone_tree(v, ctx) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_clone_tree(v, ctx) for v in obj)
    return obj


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


def _restore_optimizer_state(optimizer, snap) -> None:
    """Roll optimizer.state back to the snapshot bit-exactly, in place.

    The snapshot records the original ``optimizer.state`` MAPPING object
    and every per-parameter state mapping object (both by reference); any
    hostile rebinding is undone by reinstatement before contents are
    restored, and hostile injected keys are deleted. Two modes:
      * IN-PLACE — when every live leaf still matches its pristine
        counterpart structurally AND the live alias topology equals the
        snapshot's (the same sharing pattern over full backing storages,
        with independent empty storages never merged), values are copied
        through the live objects via ``copy_``, preserving every external
        tensor reference and all live sharing;
      * otherwise the WHOLE state tree is rebuilt in one alias-exact pass
        from the pristine snapshot (fresh storages cloned once per
        sharing group at full extent, views recreated with exact
        shape/stride/offset/dtype — including cross-dtype views — and
        repeated references unified) and installed into the original
        containers.
    A leaf is only ever copied when structurally compatible, so rollback
    itself can never raise a size/dtype mismatch."""
    state_ref, entry_refs, snap_tree = snap
    if optimizer.state is not state_ref:
        optimizer.state = state_ref          # undo hostile mapping rebind
    state = optimizer.state
    for key in list(state.keys()):
        if key not in snap_tree:
            del state[key]                   # hostile injection removed
    for key, ref in entry_refs.items():
        if isinstance(ref, dict) and _get_live(state, key) is not ref:
            state[key] = ref                 # undo hostile entry rebind

    plan = _plan_inplace_state_restore(state, entry_refs, snap_tree)
    if plan is not None:
        for live_leaf, snap_leaf in plan:
            with torch.no_grad():
                live_leaf.copy_(snap_leaf)
        return
    # one shared context: aliases ACROSS entries survive the rebuild too
    ctx = _AliasCloneCtx()
    rebuilt = {key: _deep_clone_tree(entry, ctx)
               for key, entry in snap_tree.items()}
    for key, entry in rebuilt.items():
        ref = entry_refs.get(key)
        if isinstance(ref, dict) and isinstance(entry, dict):
            ref.clear()                      # refill original container
            ref.update(entry)
        else:
            state[key] = entry


def _storage_tag(t):
    """Identity of a leaf's backing storage for topology comparison;
    zero-capacity storages are tagged separately so independent empty
    tensors are never grouped by a coincidental data_ptr()==0."""
    try:
        us = t.untyped_storage()
    except Exception:  # noqa: BLE001 - opaque storages compare by position
        return ("opaque", id(t))
    if us.nbytes() == 0:
        return ("empty", id(t))
    return ("storage", us.data_ptr(), us.nbytes())


def _same_storage_partition(live_tags, snap_tags) -> bool:
    """True iff the two tag sequences induce the SAME sharing pattern:
    a consistent bijection between snapshot-side and live-side storage
    groups along identical traversal positions."""
    match, taken = {}, set()
    for live_tag, snap_tag in zip(live_tags, snap_tags):
        if snap_tag in match:
            if match[snap_tag] != live_tag:
                return False
        elif live_tag in taken:
            return False
        else:
            match[snap_tag] = live_tag
            taken.add(live_tag)
    return True


def _walk_inplace_plan(live, snap_entry, copies, live_tags, snap_tags):
    """Depth-first structural compatibility walk over one state entry:
    records (live, snapshot) copy pairs for tensor leaves that match in
    shape, dtype, device, layout, strides AND storage offset, and collects
    both sides' storage tags in lockstep. Any structural mismatch —
    replaced objects, swapped metadata, missing/injected fields, changed
    non-tensor values — aborts with False, forcing a full rebuild."""
    if torch.is_tensor(snap_entry):
        if not torch.is_tensor(live) or \
                not _inplace_exact_restore_possible(live, snap_entry):
            return False
        copies.append((live, snap_entry))
        live_tags.append(_storage_tag(live))
        snap_tags.append(_storage_tag(snap_entry))
        return True
    if isinstance(snap_entry, dict):
        if not isinstance(live, dict) or \
                set(live.keys()) != set(snap_entry.keys()):
            return False
        return all(_walk_inplace_plan(live[k], v, copies, live_tags,
                                      snap_tags)
                   for k, v in snap_entry.items())
    if isinstance(snap_entry, (list, tuple)):
        if not isinstance(live, type(snap_entry)) or \
                len(live) != len(snap_entry):
            return False
        return all(_walk_inplace_plan(a, b, copies, live_tags, snap_tags)
                   for a, b in zip(live, snap_entry))
    try:
        return (not torch.is_tensor(live)
                and type(live) is type(snap_entry)
                and bool(live == snap_entry))
    except Exception:  # noqa: BLE001 - hostile __eq__: treat as mismatch
        return False


def _plan_inplace_state_restore(state, entry_refs, snap_tree):
    """Whole-state in-place restoration plan, or None when ANY entry has
    drifted structurally or the live alias topology differs from the
    snapshot's (decided once globally so cross-entry sharing can never be
    half-restored)."""
    copies, live_tags, snap_tags = [], [], []
    for key, snap_entry in snap_tree.items():
        live_entry = _get_live(state, key)
        if live_entry is not entry_refs.get(key):
            return None               # identity drift: rebuild everything
        if not _walk_inplace_plan(live_entry, snap_entry, copies,
                                  live_tags, snap_tags):
            return None
    if not _same_storage_partition(live_tags, snap_tags):
        return None
    return copies


def _get_live(mapping, key):
    return mapping.get(key) if hasattr(mapping, "get") else None


def _snapshot_optimizer_state(optimizer, ctx=None) -> tuple:
    """Capture optimizer.state exactly: the original MAPPING object and
    every per-parameter state mapping object (by reference, so hostile
    rebinds can be undone) plus a deep alias-exact value-tree copy."""
    if ctx is None:
        ctx = _AliasCloneCtx()
    state_ref = optimizer.state
    entry_refs = ({k: v for k, v in state_ref.items()}
                  if hasattr(state_ref, "items") else {})
    snap_tree = _deep_clone_tree(state_ref, ctx)
    return state_ref, entry_refs, snap_tree


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


def _snapshot_param_groups(optimizer, ctx=None) -> tuple:
    """Structural snapshot of ``param_groups`` sufficient to rebuild the
    exact pre-step topology, INCLUDING the identity of the original outer
    ``param_groups`` list object itself. Per group it records the live
    group dict (by reference), exact copies of every non-``params`` field
    (tensor fields alias-exact via the shared context), whether a
    ``params`` entry existed, the live params list object (by reference)
    and a copy of its Parameter order. Parameter objects themselves are
    never copied or replaced."""
    groups_obj = optimizer.param_groups
    outer_ref = groups_obj if isinstance(groups_obj, list) else None
    snap = []
    for group in groups_obj:
        fields = {
            k: (_deep_clone_tree(v, ctx) if torch.is_tensor(v)
                else copy.deepcopy(v))
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
            # snapshot tensor leaves are installed as-is (alias-exact);
            # the pristine snapshot is never reused after this rollback
            group[key] = (val if torch.is_tensor(val)
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


def guarded_optimizer_step(optimizer, loss, params, clip_norm) -> None:
    """Step only when everything is finite; roll back completely otherwise.

    Transactional contract:
      * PRE-STEP (before any mutation): finite loss, positive finite clip,
        finite parameters, finite gradients and a finite total gradient
        norm. Any rejection here performs no step and mutates nothing.
      * Gradients are rechecked after clipping; a post-clip rejection also
        restores the pre-clip gradients.
      * Every trainable parameter, every pre-clip gradient (including the
        ``None`` versus tensor distinction and exact bytes), a deep exact
        copy of the optimizer state and the complete ``param_groups``
        topology (the original outer list object, group count/order,
        every group field, and each group's live Parameter identities in
        exact order) are snapshotted BEFORE any mutation — before
        clipping — via an alias-exact capture that preserves the full
        reachable STORAGE graph: every tensor's values plus its exact
        shape/dtype/device/layout/stride/storage-offset metadata,
        shared-storage aliasing among leaves (including across nested
        entries and across the state/group/parameter boundaries),
        repeated Tensor-object references, cross-dtype views over one
        storage, and full backing-storage extents even when only an
        offset view is reachable; independent empty tensors are never
        merged by a coincidental data_ptr()==0. A graph that cannot be
        captured this way raises UnsupportedStorageGraphError here,
        fail-closed, before clipping or stepping. If the step raises, or
        any post-step parameter or optimizer-state tensor is non-finite,
        all of it — parameters, gradients, complete optimizer state
        including injected fields with every leaf's exact metadata AND
        aliasing, the original ``optimizer.state`` mapping object and
        per-parameter mappings, and the full param-group structure down
        to outer-list identity — is restored exactly while keeping the
        live Parameter identities; a drifted subtree is rebuilt
        alias-exactly from the pristine snapshot rather than copied
        through, so rollback itself can never raise a size/dtype
        mismatch. The original step exception is then re-raised intact
        (rollback failures chain onto it as __cause__, never replacing
        it). The optimizer remains usable after rollback.
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

    # -- transaction: snapshot everything the step may corrupt BEFORE any
    # mutation (clipping included); unsupported storage graphs fail closed
    # here, before clipping or optimizer.step can touch anything -----------
    ctx = _AliasCloneCtx()
    param_snap = [_deep_clone_tree(p, ctx) for p in params]
    grad_snap = {id(p): (None if p.grad is None
                         else _deep_clone_tree(p.grad, ctx))
                 for p in params}
    state_snap = _snapshot_optimizer_state(optimizer, ctx)
    groups_snap = _snapshot_param_groups(optimizer, ctx)

    def restore_params() -> None:
        with torch.no_grad():
            for p, saved in zip(params, param_snap):
                p.copy_(saved)

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
