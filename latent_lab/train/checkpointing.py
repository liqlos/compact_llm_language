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
    parameters, the complete optimizer STATE GRAPH and the full
    param-group topology (the original outer list object, group
    count/order, fields, per-group Parameter identities and order) back
    bit-exactly, without ever masking or replacing the original step
    exception. The optimizer-state transaction snapshot is ONE atomic
    serialization roundtrip of the whole ordered state-value graph taken
    before ANY mutation (including gradient clipping): torch's memo and
    storage tables preserve repeated tensor objects, distinct views
    sharing one full backing storage (values, size/stride/
    storage_offset/dtype/device/layout), nested containers and
    sparse/quantized layouts. Graphs that serialization cannot represent
    faithfully — notably cross-dtype aliases of one untyped storage —
    are rejected fail-closed before anything is mutated. Rollback
    reinstates the original ``state`` mapping object and wholesale-binds
    its original ordered key objects to a pristine rehydration of that
    graph; it never mixes recursive per-leaf copies into the swap. The
    transaction set is derived from the optimizer's own groups, so
    Parameters omitted by the caller stay covered.
"""

from __future__ import annotations

import copy
import io
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


class OptimizerStateGraphError(CheckpointError):
    """The optimizer-state VALUE GRAPH cannot be transacted without
    silent degradation — e.g. it aliases one untyped storage through
    different dtypes (which torch serialization cannot represent), or a
    snapshot roundtrip failed its fidelity verification."""


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


def _walk_graph_nodes(obj, _seen=None):
    """Yield every node of a nested dict/list/tuple/tensor value graph
    exactly once (cycle-safe)."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))
    yield obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_graph_nodes(k, _seen)
            yield from _walk_graph_nodes(v, _seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_graph_nodes(v, _seen)


def _reject_unrepresentable_aliases(state) -> None:
    """Fail closed BEFORE any mutation on graphs torch serialization
    cannot represent faithfully: two tensors viewing ONE untyped storage
    under DIFFERENT dtypes would be silently rewritten (or rejected
    mid-save) by ``torch.save``, so the transaction refuses them with a
    precise domain error instead of degrading them."""
    owners = {}  # untyped-storage data_ptr -> first tensor seen with it
    for node in _walk_graph_nodes(state):
        if not torch.is_tensor(node) or node.layout != torch.strided:
            continue
        ptr = node.untyped_storage().data_ptr()
        if ptr == 0:
            continue
        rep = owners.get(ptr)
        if rep is None:
            owners[ptr] = node
        elif rep.dtype != node.dtype:
            raise OptimizerStateGraphError(
                "optimizer state aliases one untyped storage with "
                f"different dtypes ({rep.dtype} and {node.dtype}); torch "
                "serialization cannot represent this graph, so refusing "
                "to transact rather than degrade it silently")


def _roundtrip_via_local_bytes(obj):
    """ONE ``torch.save`` into an in-memory buffer + ONE ``torch.load``
    with explicit arguments.

    The bytes are generated locally from the live trusted object graph —
    never an external source — so loading with ``weights_only=False`` is
    safe; no ``map_location`` is passed, preserving every device.
    Serialization memo/storage tables preserve repeated objects, distinct
    views sharing one full backing storage (values, size/stride/
    storage_offset/dtype/device/layout), nested containers and
    sparse/quantized layouts."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def _verify_roundtrip_fidelity(source, loaded) -> None:
    """Lockstep structural verification that a rehydrated graph equals
    the live one: identical container structure, per-tensor metadata
    (shape/dtype/device/layout/strides/storage offset/full backing
    extent) and values, preserved repeated-object identities and
    preserved storage-sharing relations. Any deviation fails closed."""

    def fail(msg: str):
        raise OptimizerStateGraphError(
            "optimizer-state serialization roundtrip broke the value "
            f"graph: {msg}")

    memo = {}        # id(source node) -> loaded node (tensors/containers)
    storage_map = {} # source storage ptr -> loaded storage ptr
    reverse_map = {} # loaded storage ptr -> source storage ptr

    def walk(src, ldd):
        if torch.is_tensor(src):
            if not torch.is_tensor(ldd):
                fail("a tensor was replaced by a non-tensor")
            prior = memo.get(id(src))
            if prior is not None:
                if prior is not ldd:
                    fail("repeated tensor object lost its identity")
                return
            memo[id(src)] = ldd
            if (tuple(src.shape) != tuple(ldd.shape)
                    or src.dtype != ldd.dtype
                    or src.device != ldd.device
                    or src.layout != ldd.layout
                    or src.requires_grad != ldd.requires_grad):
                fail(f"tensor metadata changed ({tuple(src.shape)}, "
                     f"{src.dtype})")
            if src.layout == torch.strided:
                if (tuple(src.stride()) != tuple(ldd.stride())
                        or src.storage_offset() != ldd.storage_offset()):
                    fail("view geometry (stride/storage offset) changed")
                sptr = src.untyped_storage().data_ptr()
                lptr = ldd.untyped_storage().data_ptr()
                if src.untyped_storage().nbytes() != \
                        ldd.untyped_storage().nbytes():
                    fail("full backing-storage extent changed")
                if storage_map.setdefault(sptr, lptr) != lptr:
                    fail("storage-sharing relations were rewritten")
                if reverse_map.setdefault(lptr, sptr) != sptr:
                    fail("distinct storages collapsed into one")
            try:
                same_values = bool(torch.equal(src, ldd))
            except (TypeError, RuntimeError):
                same_values = True   # exotic layout; metadata checks stand
            if not same_values:
                fail("tensor values changed")
            return
        if isinstance(src, (dict, list, tuple)):
            if type(ldd) is not type(src) or len(ldd) != len(src):
                fail(f"container {type(src).__name__} changed shape/type")
            prior = memo.get(id(src))
            if prior is not None:
                if prior is not ldd:
                    fail("shared container object split into copies")
                return
            memo[id(src)] = ldd
            if isinstance(src, dict):
                for sk, lk, sv, lv in zip(src.keys(), ldd.keys(),
                                          src.values(), ldd.values(),
                                          strict=True):
                    walk(sk, lk)
                    walk(sv, lv)
            else:
                for sv, lv in zip(src, ldd, strict=True):
                    walk(sv, lv)
            return
        if type(ldd) is not type(src) or ldd != src:
            fail("scalar/non-tensor leaf changed")

    walk(source, loaded)


def _snapshot_optimizer_state(optimizer) -> tuple:
    """Capture the ENTIRE ordered state-value graph as one pristine,
    rehydrated in-memory copy — taken BEFORE ANY mutation (including
    gradient clipping).

    Returns ``(original mapping object, ordered original key objects,
    pristine ordered values)``. Unsupported graphs raise
    ``OptimizerStateGraphError`` here, before anything can be mutated."""
    state = optimizer.state
    keys = list(state.keys())
    _reject_unrepresentable_aliases(state)
    try:
        pristine = _roundtrip_via_local_bytes(state)
    except OptimizerStateGraphError:
        raise
    except Exception as e:  # noqa: BLE001 - unrepresentable => fail closed
        raise OptimizerStateGraphError(
            f"optimizer-state graph cannot be snapshotted: {e}") from e
    if type(pristine) is not type(state) or len(pristine) != len(keys):
        raise OptimizerStateGraphError(
            "optimizer-state mapping did not survive the snapshot "
            "roundtrip intact")
    _verify_roundtrip_fidelity(state, pristine)
    return state, keys, list(pristine.values())


def _restore_optimizer_state(optimizer, snap) -> None:
    """Roll ``optimizer.state`` back to the pristine snapshot WHOLESALE.

    Reinstates the original state mapping object if a hostile step
    rebound it, discards ALL injected/replaced content, and binds the
    original ordered key objects to a fresh rehydration of the retained
    pristine value graph. This is an atomic graph swap: it never routes
    through per-leaf ``copy_``/recursive cloning, so shared storages,
    view geometries and repeated-object identities survive untouched,
    and the retained pristine graph itself is never consumed (rollback
    stays repeatable)."""
    mapping_ref, keys, pristine_values = snap
    live = optimizer.state
    if live is not mapping_ref:
        optimizer.state = mapping_ref     # undo hostile mapping rebind
        live = mapping_ref
    fresh = _roundtrip_via_local_bytes(list(pristine_values))
    if not isinstance(fresh, list) or len(fresh) != len(keys):
        raise OptimizerStateGraphError(
            "pristine state graph rehydration changed arity")
    live.clear()
    for key, value in zip(keys, fresh):
        live[key] = value


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


def guarded_optimizer_step(optimizer, loss, params, clip_norm) -> None:
    """Step only when everything is finite; roll back completely otherwise.

    Transactional contract:
      * PRE-STEP (before any mutation): finite loss, positive finite clip,
        finite parameters, finite gradients and a finite total gradient
        norm. Any rejection here performs no step and mutates nothing.
      * Gradients are rechecked after clipping; a post-clip rejection also
        restores the pre-clip gradients.
      * Every trainable parameter, every pre-clip gradient (including the
        ``None`` versus tensor distinction and exact bytes), a pristine
        graph snapshot of the whole optimizer state and the complete
        ``param_groups`` topology (the original outer list object, group
        count/order, every group field, and each group's live Parameter
        identities in exact order) are snapshotted before
        ``optimizer.step()``. The state snapshot is ONE serialization
        roundtrip of the ENTIRE ordered value graph taken BEFORE ANY
        mutation (including gradient clipping): it preserves repeated
        objects, distinct views sharing one full backing storage,
        per-tensor shape/dtype/device/layout/stride/offset metadata and
        nested containers; graphs that cannot be represented — notably
        cross-dtype aliases of one untyped storage — are rejected with
        ``OptimizerStateGraphError`` before anything is mutated. If the
        step raises, or any post-step parameter or optimizer-state tensor
        is non-finite, all of it is restored exactly: parameters and
        gradients bit-exactly, the original ``state`` mapping object and
        ordered key objects wholesale-bound to the pristine graph (a
        hostile mapping rebind or injected keys cannot survive), and the
        full param-group structure down to outer-list identity — while
        keeping the live Parameter identities. The original step
        exception is then re-raised intact (rollback failures chain onto
        it as __cause__, never replacing it). The optimizer remains
        usable after rollback.
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

    # -- transaction: snapshot BEFORE ANY mutation (incl. clipping) ----------
    state_snap = _snapshot_optimizer_state(optimizer)

    torch.nn.utils.clip_grad_norm_(params, clip)
    for p, _ in pairs:  # recheck after clipping
        if not bool(torch.isfinite(p.grad).all()):
            restore_grads()
            raise NonFiniteTrainingStateError(
                "gradients became non-finite after clipping")

    # -- transaction: snapshot everything else the step may corrupt ----------
    param_snap = [p.detach().clone() for p in params]
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
