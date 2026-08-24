"""Adversarial proof for the exact storage-graph optimizer transaction.

Every behavioral test here exercises a defect class that the historical
``detach().clone()`` leaf-snapshot implementation provably mishandled on
pristine 60a48d5 (severed alias graphs, lost hidden backing bytes,
normalized strides/offsets, broken repeated-reference semantics, absent
preflight/budget, lost ``optimizer.state`` mapping identity) and that the
storage-graph engine in ``latent_lab.train.opt_transaction`` must handle
exactly.  Tests intentionally import nothing new: on the pristine commit
they RUN and FAIL on their assertions.
"""

import time
import traceback

import pytest

import torch

from latent_lab.train.checkpointing import guarded_optimizer_step


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _BombSGD(torch.optim.SGD):
    """Momentum SGD performing a REAL update, then arbitrary corruption,
    then raising its own distinctive exception."""

    def __init__(self, params, lr, corrupt=None, msg="armed transaction bomb"):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False
        self._corrupt = corrupt
        self._msg = msg

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
            if self._corrupt is not None:
                self._corrupt(self)
            raise RuntimeError(self._msg)


def _fresh_param(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter(torch.ones(n))
    grad = torch.randn(n, generator=g) * 0.01
    return w, grad


def _sgd_with_state(w, lr=0.05):
    """Optimizer with one clean guarded step already taken (momentum)."""
    opt = torch.optim.SGD([w], lr=lr, momentum=0.9)
    with torch.no_grad():
        w.grad = torch.randn_like(w) * 0.01
    guarded_optimizer_step(opt, torch.tensor(0.5), [w], 1.0)
    assert "momentum_buffer" in opt.state[w]
    return opt


def _alias_zoo():
    """One backing storage, several crafted views: nonzero offset,
    overlapping, non-dense strides, plus a hidden region OUTSIDE every
    visible leaf."""
    backing = (torch.arange(32, dtype=torch.float32) + 1.0) * 0.25
    v_offset = backing[5:13]                       # offset 5, dense
    v_overlap = torch.as_strided(backing, (8,), (1,), 9)      # 9..16
    v_sparse = torch.as_strided(backing, (3, 2), (1, 7), 2)   # non-dense
    # covered visible element indices: {2,3,4, 5..12(v_offset),
    # 9..16(v_overlap), 9,10,11, 16,17} -> hidden window 18..31
    return backing, v_offset, v_overlap, v_sparse


def _snapshot_backing(t):
    return t.detach().clone()


def _assert_same_values(a, b):
    assert bool(torch.equal(a, b))


# ---------------------------------------------------------------------------
# aliases, offsets, overlap, non-dense strides, hidden bytes
# ---------------------------------------------------------------------------

def test_alias_views_full_extent_and_hidden_bytes_restored():
    w, grad = _fresh_param()
    backing, v_offset, v_overlap, v_sparse = _alias_zoo()
    opt = _sgd_with_state(w)
    st = opt.state[w]
    st["mb"] = v_offset                 # replaces momentum slot reference
    st["overlap"] = v_overlap
    st["nondense"] = v_sparse

    def corrupt(o):
        s = o.state[o.param_groups[0]["params"][0]]
        with torch.no_grad():
            s["mb"].add_(100.0)         # visible-view corruption
            s["overlap"].mul_(-2.0)
            # stomp memory OUTSIDE every visible leaf (hidden bytes)
            backing[18:32] = 777.0

    bomb = _BombSGD([w], lr=0.05, corrupt=corrupt)
    bomb.state[w] = st                  # share the crafted state
    pre_backing = _snapshot_backing(backing)

    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)

    # full backing extent restored bit-exactly, hidden bytes included
    _assert_same_values(backing, pre_backing)
    # live view objects kept, exact metadata intact
    assert opt.state[w]["mb"] is v_offset
    assert opt.state[w]["overlap"] is v_overlap
    assert opt.state[w]["nondense"] is v_sparse
    assert tuple(v_offset.stride()) == (1,) and v_offset.storage_offset() == 5
    assert tuple(v_sparse.stride()) == (1, 7) and \
        v_sparse.storage_offset() == 2


def test_repeated_object_vs_distinct_views_preserved():
    w, grad = _fresh_param()
    backing, v_offset, _, _ = _alias_zoo()
    opt = _sgd_with_state(w)
    st = opt.state[w]
    twin = backing[1:4]                 # distinct view object
    st["rep1"] = v_offset               # SAME object under two keys
    st["rep2"] = v_offset
    st["twin"] = twin

    def corrupt(o):
        s = o.state[o.param_groups[0]["params"][0]]
        s["rep2"] = torch.full((8,), -5.0)     # sever one repeated ref
        del s["twin"]                          # hostile removal
        with torch.no_grad():
            s["rep1"].add_(50.0)

    bomb = _BombSGD([w], lr=0.05, corrupt=corrupt)
    bomb.state[w] = st
    pre_twin = twin.detach().clone()

    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)

    s = opt.state[w]
    # repeated-reference semantics: the SAME object back under both keys
    assert s["rep1"] is v_offset and s["rep2"] is v_offset
    assert s["twin"] is twin
    _assert_same_values(twin, pre_twin)
    _assert_same_values(backing[1:4], pre_twin)


def test_zero_stride_expand_view_survives_exactly():
    w, grad = _fresh_param()
    base = torch.randn(6) * 0.5 + 3.0
    expanded = base.expand(4, 6)        # zero stride, shares storage
    opt = _sgd_with_state(w)
    st = opt.state[w]
    st["flat"] = base.view(-1)          # dense view onto same storage
    st["expanded"] = expanded

    def corrupt(o):
        s = o.state[o.param_groups[0]["params"][0]]
        s["expanded"] = torch.zeros(4, 6)      # hostile rebind...
        with torch.no_grad():
            s["flat"].add_(9.0)                # ...plus byte corruption

    bomb = _BombSGD([w], lr=0.05, corrupt=corrupt)
    bomb.state[w] = st
    pre_base = base.detach().clone()

    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb") as ei:
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)
    assert ei.value.__cause__ is None, \
        "clean rollback must not chain a cause onto the step exception"

    s = opt.state[w]
    assert s["expanded"] is expanded, "expand view object was replaced"
    assert tuple(expanded.stride()) == (0, 1), "zero stride normalized"
    _assert_same_values(base, pre_base)
    _assert_same_values(expanded, pre_base.expand(4, 6))


def test_scalar_and_empty_storages_not_collapsed():
    w, grad = _fresh_param()
    opt = _sgd_with_state(w)
    st = opt.state[w]

    owner = torch.empty(4)              # shared-storage empty views
    e_shared_a = owner[:0]
    e_shared_b = owner[2:2]
    e_distinct_a = torch.empty(0)       # genuinely separate storages
    e_distinct_b = torch.empty(0)
    scalar = torch.tensor(2.5)          # 0-dim tensor leaf
    st["esha"] = e_shared_a
    st["eshb"] = e_shared_b
    st["edisa"] = e_distinct_a
    st["edisb"] = e_distinct_b
    st["scalar"] = scalar

    bomb = _BombSGD([w], lr=0.05, corrupt=lambda o: (
        o.state[o.param_groups[0]["params"][0]].update(
            {"injected": "junk", "scalar": "not-a-tensor"})))
    bomb.state[w] = st
    pre_owner = owner.detach().clone()

    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)

    s = opt.state[w]
    # aliasing between the two shared-storage empties survives...
    assert s["esha"].untyped_storage()._cdata == \
        s["eshb"].untyped_storage()._cdata
    # ...while truly distinct empty storages are never merged
    assert s["edisa"].untyped_storage()._cdata != \
        s["edisb"].untyped_storage()._cdata
    assert isinstance(s["scalar"], torch.Tensor) and \
        s["scalar"].dtype == torch.float32 and \
        float(s["scalar"]) == 2.5
    _assert_same_values(owner, pre_owner)
    assert "injected" not in s


def test_cross_dtype_views_share_one_storage_including_hidden_tail():
    w, grad = _fresh_param()
    raw = (torch.arange(24, dtype=torch.uint8) * 3)   # one byte storage
    f32 = raw[:16].view(torch.float32)     # float view of first 16 bytes
    u8 = raw                                # whole-storage uint8 view
    i16 = raw[:16].view(torch.int16)       # int16 reinterpretation
    opt = _sgd_with_state(w)
    st = opt.state[w]
    st["f"], st["u"], st["i"] = f32, u8, i16
    st["u_again"] = u8                     # SAME object under a second key

    def corrupt(o):
        s = o.state[o.param_groups[0]["params"][0]]
        s["u_again"] = torch.full((24,), 7, dtype=torch.uint8)
        with torch.no_grad():
            s["i"].add_(1)                  # mutate THROUGH the int16 view
            s["f"].add_(1.0)                # and through the float view
            raw[20:24] = 255                # tail beyond f32/i16 leaves

    bomb = _BombSGD([w], lr=0.05, corrupt=corrupt)
    bomb.state[w] = st
    pre_raw = raw.detach().clone()
    pre_f = f32.detach().clone()

    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)

    s = opt.state[w]
    assert s["f"] is f32 and s["u"] is u8 and s["i"] is i16
    assert s["u_again"] is u8, \
        "repeated cross-dtype view reference was not preserved"
    assert s["f"].untyped_storage()._cdata == \
        s["u"].untyped_storage()._cdata == \
        s["i"].untyped_storage()._cdata
    _assert_same_values(raw, pre_raw)       # bytes, tail included
    _assert_same_values(f32, pre_f)
    # cross-dtype interpretations remain coherent on the shared storage
    _assert_same_values(f32.view(torch.uint8), raw[:16])
    _assert_same_values(i16.view(torch.uint8), raw[:16])


def test_hidden_bytes_outside_all_visible_leaves_restored():
    w, grad = _fresh_param()
    backing = torch.linspace(0.0, 1.0, 40)
    visible = backing[10:20]               # only slice exposed to state
    sentinel = backing[30:40].clone()      # far outside the visible leaf
    opt = _sgd_with_state(w)
    opt.state[w]["only_view"] = visible

    def corrupt(o):
        with torch.no_grad():
            o.state[w]["only_view"].mul_(3.0)
            backing[30:40] = float("nan")  # corrupt beyond ANY leaf

    bomb = _BombSGD([w], lr=0.05, corrupt=corrupt)
    bomb.state[w] = opt.state[w]
    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)

    _assert_same_values(backing[30:40], sentinel)
    _assert_same_values(visible, (torch.linspace(0.0, 1.0, 40))[10:20])
    assert opt.state[w]["only_view"] is visible


# ---------------------------------------------------------------------------
# hostile optimizer.state mapping rebinding
# ---------------------------------------------------------------------------

def test_hostile_state_mapping_rebind_identity_restored():
    w, grad = _fresh_param()
    opt = _sgd_with_state(w)
    original_mapping = opt.state
    pre_keys = list(original_mapping.keys())
    pre_mb = opt.state[w]["momentum_buffer"].detach().clone()

    def corrupt(o):
        hostile = {}
        hostile[w] = {"momentum_buffer": torch.full((8,), -1.0),
                      "evil": True}
        o.state = hostile           # hostile attribute REBIND

    bomb = _BombSGD([w], lr=0.05, corrupt=corrupt)
    bomb.state = opt.state
    with torch.no_grad():
        w.grad = grad.clone()
    bomb.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)

    assert opt.state is original_mapping, \
        "original optimizer.state mapping object identity was not restored"
    assert list(opt.state.keys()) == pre_keys
    assert "evil" not in opt.state[w]
    _assert_same_values(opt.state[w]["momentum_buffer"], pre_mb)


# ---------------------------------------------------------------------------
# fail-closed preflight: unsupported tensors, budget
# ---------------------------------------------------------------------------

def _big_grad(w):
    g = torch.randn_like(w) * 10.0
    with torch.no_grad():
        w.grad = g.clone()
    return g


@pytest.mark.parametrize("kind", ["sparse", "meta", "quantized",
                                  "subclass"])
def test_unsupported_state_tensors_rejected_before_clip_or_step(kind):
    w, _ = _fresh_param()
    opt = _sgd_with_state(w)
    if kind == "sparse":
        evil = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 2.0]),
            (2, 2))
    elif kind == "meta":
        evil = torch.empty(2, 2, device="meta")
    elif kind == "quantized":
        evil = torch.quantize_per_tensor(torch.randn(4), scale=1.0,
                                         zero_point=0, dtype=torch.qint8)
    else:
        evil = torch.nn.Parameter(torch.zeros(3))   # subclass in STATE
    opt.state[w]["evil"] = evil

    pre_params = w.detach().clone()
    pre_state = {k: dict(entry) for k, entry in opt.state.items()}
    grad = _big_grad(w)                     # norm >> clip: clipping WOULD
    tiny_clip = 0.01                        # visibly rescale these grads

    with pytest.raises(RuntimeError,
                       match="refusing|cannot be reconstructed|unsupported"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [w], tiny_clip)

    # NOTHING happened: gradients unscaled (clip never ran), no step,
    # state untouched, parameter untouched.
    assert w.grad is not None and bool(torch.equal(w.grad, grad)), \
        "gradients were clipped despite fail-closed preflight rejection"
    _assert_same_values(w.detach(), pre_params)
    assert set(opt.state.keys()) == set(pre_state.keys())
    for k, entry in pre_state.items():
        for n, v in entry.items():
            lv = opt.state[k][n]
            assert lv is v, f"{n} object was replaced"
            if torch.is_tensor(v):
                try:
                    assert bool(torch.equal(lv, v)), f"{n} value changed"
                except NotImplementedError:   # sparse/meta backends
                    pass


def test_storage_byte_budget_rejected_before_mutation(monkeypatch):
    w, _ = _fresh_param()
    opt = _sgd_with_state(w)
    monkeypatch.setenv("LATENT_LAB_STATE_SNAPSHOT_BUDGET_BYTES", "64")

    pre_params = w.detach().clone()
    pre_mb = opt.state[w]["momentum_buffer"].detach().clone()
    grad = _big_grad(w)

    with pytest.raises(RuntimeError, match="budget"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [w], 0.01)

    assert w.grad is not None and bool(torch.equal(w.grad, grad)), \
        "budget rejection happened after clipping"
    _assert_same_values(w.detach(), pre_params)
    _assert_same_values(opt.state[w]["momentum_buffer"], pre_mb)

    # disabling the budget (explicitly) lets the identical step proceed
    monkeypatch.setenv("LATENT_LAB_STATE_SNAPSHOT_BUDGET_BYTES", "")
    guarded_optimizer_step(opt, torch.tensor(0.5), [w], 1.0)
    assert not bool(torch.equal(w.detach(), pre_params)), \
        "budget-less configuration refused a legitimate step"


# ---------------------------------------------------------------------------
# exception identity and rollback-failure chaining
# ---------------------------------------------------------------------------

class _DistinctStepFailure(RuntimeError):
    pass


def test_original_exception_identity_preserved_on_clean_rollback():
    class _CustomBomb(torch.optim.SGD):
        def __init__(self, params, lr):
            super().__init__(params, lr=lr, momentum=0.9)

        def step(self, closure=None):
            super().step(closure)
            raise _DistinctStepFailure("the-exact-original-message")

    w, grad = _fresh_param()
    opt = _sgd_with_state(w)
    custom = _CustomBomb([w], lr=0.05)
    custom.state[w] = opt.state[w]
    with torch.no_grad():
        w.grad = grad.clone()
    with pytest.raises(_DistinctStepFailure) as ei:
        guarded_optimizer_step(custom, torch.tensor(0.5), [w], 1.0)
    assert type(ei.value) is _DistinctStepFailure
    assert str(ei.value) == "the-exact-original-message"
    assert ei.value.args == ("the-exact-original-message",)
    assert ei.value.__cause__ is None, \
        "clean rollback must not attach a cause"
    tb_text = "".join(traceback.format_exception(ei.value))
    assert "in step\n" in tb_text or "in step' \n" in tb_text or \
        "step (closure" in tb_text, \
        "original traceback relationship to optimizer.step was lost"


def test_rollback_failure_chained_as_cause_only():
    """A hostile optimizer whose ``state`` attribute REFUSES restoration
    makes genuine rollback fail; the original step exception must survive
    with the rollback failure chained purely as its ``__cause__``."""

    class _LockedStateSGD(torch.optim.SGD):
        def __init__(self, params, lr):
            super().__init__(params, lr=lr, momentum=0.9)

        @property
        def state(self):
            return self.__dict__["state"]

        @state.setter
        def state(self, value):
            if getattr(self, "lock_state", False):
                raise RuntimeError("rollback exploded")
            self.__dict__["state"] = value

        def step(self, closure=None):
            super().step(closure)
            s = self.__dict__["state"][
                self.param_groups[0]["params"][0]]
            s["momentum_buffer"].add_(99.0)     # corrupt (coherent group)
            self.lock_state = True
            self.__dict__["state"] = {"hostile": {}}   # rebind + lock
            raise RuntimeError("step-failed-first")

    w, grad = _fresh_param()
    opt = _sgd_with_state(w)
    bomb = _LockedStateSGD([w], lr=0.05)
    bomb.__dict__["state"] = opt.state           # share crafted state
    with torch.no_grad():
        w.grad = grad.clone()
    with pytest.raises(RuntimeError, match="step-failed-first") as ei:
        guarded_optimizer_step(bomb, torch.tensor(0.5), [w], 1.0)
    assert "step-failed-first" in str(ei.value), \
        "original step exception was replaced by the rollback error"
    assert isinstance(ei.value.__cause__, RuntimeError) and \
        "rollback exploded" in str(ei.value.__cause__), \
        "rollback failure must appear solely as the chained cause"


# ---------------------------------------------------------------------------
# retry after rollback matches independent clean control
# ---------------------------------------------------------------------------

def test_retry_after_full_adversarial_rollback_matches_clean_control():
    backing, v_offset, v_overlap, v_sparse = _alias_zoo()

    def build():
        g = torch.Generator().manual_seed(42)
        torch.manual_seed(42)
        m = torch.nn.Linear(8, 4)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        return m

    m_adv, m_ctl = build(), build()
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(7))
    opt_adv = _BombSGD(list(m_adv.parameters()), lr=0.03)
    opt_ctl = torch.optim.SGD(list(m_ctl.parameters()), lr=0.03,
                              momentum=0.9)     # match _BombSGD exactly

    def fresh_loss(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    def ctl_step():
        loss = fresh_loss(m_ctl)
        torch.nn.utils.clip_grad_norm_(list(m_ctl.parameters()), 1.0)
        opt_ctl.step()

    # one clean step on both sides -> identical non-empty momentum state
    guarded_optimizer_step(opt_adv, fresh_loss(m_adv).detach(),
                           list(m_adv.parameters()), 1.0)
    ctl_step()
    for a, b in zip(m_adv.parameters(), m_ctl.parameters()):
        _assert_same_values(a.detach(), b.detach())

    # craft aliased views inside the adversarial optimizer's state
    st = opt_adv.state[m_adv.weight]
    st["mb"], st["ov"], st["nd"] = v_offset, v_overlap, v_sparse
    pre_backing = backing.clone()
    pre_st_vals = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                   for k, v in st.items()}
    pre_lrs = [g["lr"] for g in opt_adv.param_groups]
    original_mapping = opt_adv.state
    pre_param_snap = [p.detach().clone() for p in m_adv.parameters()]

    def corrupt(o):
        s = o.state[o.param_groups[0]["params"][0]]
        s["ov"] = "junk"                       # alias replaced by scalar
        del s["nd"]                            # hostile removal
        s["injected_field"] = 123
        with torch.no_grad():
            s["mb"].add_(64.0)                 # visible-view corruption
            backing[18:32] = -999.0            # hidden-byte stomp
            for g_ in o.param_groups:
                g_["lr"] = 999.0
        o.state = {"hostile": {}}              # hostile mapping REBIND

    opt_adv._corrupt = corrupt
    opt_adv.armed = True
    with pytest.raises(RuntimeError, match="armed transaction bomb"):
        guarded_optimizer_step(opt_adv,
                               fresh_loss(m_adv).detach(),
                               list(m_adv.parameters()), 1.0)

    # exact restoration of everything the bomb touched
    _assert_same_values(backing, pre_backing)
    assert opt_adv.state is original_mapping, "mapping identity not restored"
    st2 = opt_adv.state[m_adv.weight]
    for k, v in pre_st_vals.items():
        lv = st2[k]
        if torch.is_tensor(v):
            assert torch.is_tensor(lv) and bool(torch.equal(lv, v)), k
        else:
            assert lv is v, k
    assert st2["mb"] is v_offset and st2["nd"] is v_sparse
    assert "injected_field" not in st2
    assert "hostile" not in dict(opt_adv.state)
    assert [g["lr"] for g in opt_adv.param_groups] == pre_lrs
    for p, pre in zip(m_adv.parameters(), pre_param_snap):
        _assert_same_values(p.detach(), pre)

    # disarmed retry of the SAME update vs the independent control
    opt_adv.armed = False
    guarded_optimizer_step(opt_adv,
                           fresh_loss(m_adv).detach(),
                           list(m_adv.parameters()), 1.0)
    ctl_step()

    for a, b in zip(m_adv.parameters(), m_ctl.parameters()):
        _assert_same_values(a.detach(), b.detach())
    for pa, pc in zip(m_adv.parameters(), m_ctl.parameters()):
        _assert_same_values(opt_adv.state[pa]["momentum_buffer"],
                            opt_ctl.state[pc]["momentum_buffer"])
    assert [g["lr"] for g in opt_adv.param_groups] == pre_lrs


# ---------------------------------------------------------------------------
# fast-path efficiency: one clone per UNIQUE storage, bounded overhead
# ---------------------------------------------------------------------------

def test_snapshot_clones_each_unique_storage_exactly_once(monkeypatch):
    w, grad = _fresh_param(16)
    opt = _sgd_with_state(w)
    backing, v_offset, v_overlap, _ = _alias_zoo()
    st = opt.state[w]
    st["mb"] = v_offset
    st["ov"] = v_overlap                    # same storage as mb: 1 slot

    calls = []
    real_clone = torch.UntypedStorage.clone

    def counting_clone(self):
        calls.append((self.device.type, self.nbytes()))
        return real_clone(self)

    monkeypatch.setattr(torch.UntypedStorage, "clone", counting_clone)
    try:
        with torch.no_grad():
            w.grad = grad.clone()
        guarded_optimizer_step(opt, torch.tensor(0.5), [w], 1.0)
    finally:
        monkeypatch.undo()

    involved = [opt.state[w][k] for k in opt.state[w]
                if torch.is_tensor(opt.state[w][k])]
    involved.append(w)
    involved.append(w.grad)
    unique_cdatas = {t.untyped_storage()._cdata for t in involved}
    assert len(calls) == len(unique_cdatas), (
        f"expected exactly one clone per unique storage "
        f"({len(unique_cdatas)}), got {len(calls)}")


def test_guarded_step_overhead_is_bounded():
    torch.manual_seed(3)
    m = torch.nn.Linear(64, 64)
    ps = list(m.parameters())
    opt = torch.optim.SGD(ps, lr=0.01, momentum=0.9)
    x = torch.randn(16, 64)
    n_iters = 40
    t0 = time.perf_counter()
    for _ in range(n_iters):
        loss = (m(x) ** 2).mean()
        guarded_optimizer_step(opt, loss.detach(), ps, 1.0)
    per_step_ms = (time.perf_counter() - t0) / n_iters * 1000.0
    assert per_step_ms < 25.0, \
        f"guarded fast path regressed: {per_step_ms:.1f} ms/step"
