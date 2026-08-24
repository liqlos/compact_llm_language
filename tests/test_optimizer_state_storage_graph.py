"""Adversarial proof for the exact on-device optimizer-state storage-graph
transaction in ``guarded_optimizer_step``.

Every test here targets the defect where per-leaf ``detach().clone()``
snapshots normalized storage offsets/strides, severed shared-storage
alias graphs and repeated-reference semantics, and lost hidden backing
bytes — making rollback/retry diverge. The required fast path snapshots
unique ``UntypedStorage``s whole (fail-closed, budgeted, on-device) and
restores each alias group coherently.

Tests are tiny, deterministic and CPU-only.
"""

import pytest

import torch

from latent_lab.train.checkpointing import guarded_optimizer_step


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _u8(t):
    """uint8 tensor spanning t's ENTIRE backing storage (hidden bytes too)."""
    u = t.detach().untyped_storage()
    b = torch.empty(0, dtype=torch.uint8, device=t.device)
    b.set_(u, 0, (u.nbytes(),), (1,))
    return b


def _storage_snapshot(t):
    return _u8(t).clone()


def _leaf_snapshot(t):
    """Metadata- AND value-preserving snapshot of a possibly non-dense
    tensor (``Tensor.clone`` would normalize strides/offset)."""
    u = t.detach().untyped_storage().clone()
    out = torch.empty(0, dtype=t.dtype, device=t.device)
    with torch.no_grad():
        out.set_(u, t.storage_offset(), tuple(t.shape), tuple(t.stride()))
        out.requires_grad_(t.requires_grad)
    return out


def _cdata(t):
    return t.detach().untyped_storage()._cdata


def _leaf_exact(live, expect) -> bool:
    """Value AND metadata exactness against an expected tensor."""
    return (torch.is_tensor(live)
            and tuple(live.shape) == tuple(expect.shape)
            and live.dtype == expect.dtype
            and live.device == expect.device
            and live.layout == expect.layout
            and tuple(live.stride()) == tuple(expect.stride())
            and live.storage_offset() == expect.storage_offset()
            and bool(torch.equal(live, expect)))


class _ArmedPoisonSGD(torch.optim.SGD):
    """Real momentum update, then a pluggable corruption, then raise."""

    def __init__(self, params, *, lr, corrupt, error=None):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False
        self.calls = 0
        self._corrupt = corrupt
        self._error = error

    def step(self, closure=None):
        super().step(closure)
        self.calls += 1
        if self.armed:
            self._corrupt(self)
            raise self._error or RuntimeError(
                "adversarial corruption mid-step")


def _fresh_grad(p, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(p.shape, generator=g)


def _planted_sgd(p, momentum):
    """SGD whose visible momentum buffer is caller-planted."""
    opt = torch.optim.SGD([p], lr=0.05, momentum=0.9)
    opt.state[p] = {"momentum_buffer": momentum}
    return opt


# ---------------------------------------------------------------------------
# 1+6: distinct base/view aliases, nonzero offset, non-dense/overlapping
# strides, and hidden backing bytes outside every visible leaf range
# ---------------------------------------------------------------------------

def test_alias_group_full_backing_extent_and_hidden_bytes_restored():
    backing = torch.arange(16, dtype=torch.float32) * 1.5
    momentum = backing[4:10].view(3, 2)                     # elems 4..10
    overlap = torch.as_strided(backing, (3, 3), (1, 2), 1)  # elems 1..8
    tail = backing[13:16]                                   # elems 13..16
    p = torch.nn.Parameter(torch.randn(3, 2))

    def corrupt(o):
        st = o.opt_state()
        o.hidden_low.fill_(-777.0)          # elem 0: outside ALL leaves
        o.hidden_mid.fill_(888.0)           # elems 10..13: ditto
        st["momentum_buffer"].add_(100.0)
        st["overlap"].mul_(-2.0)

    opt = _ArmedPoisonSGD([p], lr=0.05, corrupt=corrupt)
    opt.hidden_low = backing[0:1]
    opt.hidden_mid = backing[10:13]
    opt.opt_state = lambda: opt.state[p]
    opt.state[p] = {"momentum_buffer": momentum, "overlap": overlap,
                    "tail": tail}

    pre_bytes = _storage_snapshot(backing)
    pre_momentum, pre_overlap, pre_tail = (_leaf_snapshot(momentum),
        _leaf_snapshot(overlap), _leaf_snapshot(tail))
    p.grad = _fresh_grad(p, 1)

    opt.armed = True
    with pytest.raises(RuntimeError, match="adversarial corruption"):
        guarded_optimizer_step(opt, torch.tensor(0.75), [p], 1.0)

    assert torch.equal(_u8(opt.state[p]["momentum_buffer"]), pre_bytes), \
        "full backing extent (incl. hidden bytes) was not restored"
    assert _cdata(opt.state[p]["momentum_buffer"]) == _cdata(backing), \
        "alias graph was severed by rollback"
    assert _leaf_exact(opt.state[p]["momentum_buffer"], pre_momentum)
    assert _leaf_exact(opt.state[p]["overlap"], pre_overlap), \
        "overlapping-stride view metadata/values not restored exactly"
    assert _leaf_exact(opt.state[p]["tail"], pre_tail)
    assert _cdata(opt.state[p]["overlap"]) == _cdata(backing) == \
        _cdata(opt.state[p]["tail"]), "distinct views no longer alias"

    # one clean retry still works on the repaired optimizer
    opt.armed = False
    p.grad = _fresh_grad(p, 2)
    guarded_optimizer_step(opt, torch.tensor(0.75), [p], 1.0)
    assert opt.calls == 2


# ---------------------------------------------------------------------------
# 2: repeated same tensor object versus distinct views
# ---------------------------------------------------------------------------

def test_repeated_object_identity_and_distinct_view_aliases_preserved():
    p = torch.nn.Parameter(torch.randn(2, 2))
    q = torch.nn.Parameter(torch.randn(4))
    same = torch.randn(2, 2)                    # planted twice (same object)
    d0 = torch.randn(4)
    dv = d0[1:]                                 # distinct view of d0

    def corrupt(o):
        st_p, st_q = o.state[p], o.state[q]
        st_p["mirror"] = torch.full((2, 2), -9.0)   # sever repeated object
        st_p["momentum_buffer"].mul_(5.0)
        st_q["alias_tail"] = torch.randn(3)         # sever view alias
        st_q["momentum_buffer"].fill_(42.0)

    opt = _ArmedPoisonSGD([p, q], lr=0.05, corrupt=corrupt)
    opt.state[p] = {"momentum_buffer": same, "mirror": same}
    opt.state[q] = {"momentum_buffer": d0, "alias_tail": dv}
    pre_same, pre_d0, pre_dv = (same.clone(), d0.clone(),
                                _leaf_snapshot(dv))

    opt.armed = True
    with pytest.raises(RuntimeError, match="adversarial"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p, q], 1.0)

    st_p, st_q = opt.state[p], opt.state[q]
    assert st_p["mirror"] is st_p["momentum_buffer"], \
        "repeated tensor object came back as two independent objects"
    assert _leaf_exact(st_p["mirror"], pre_same)
    assert _cdata(st_q["alias_tail"]) == _cdata(st_q["momentum_buffer"]) \
        and st_q["alias_tail"].storage_offset() == \
        pre_dv.storage_offset(), \
        "distinct-view alias relationship was not reconstructed"
    assert _leaf_exact(st_q["momentum_buffer"], pre_d0) and \
        _leaf_exact(st_q["alias_tail"], pre_dv)


# ---------------------------------------------------------------------------
# 3: zero-stride expand view
# ---------------------------------------------------------------------------

def test_zero_stride_expand_view_restored_without_per_view_copy_failure():
    base = torch.randn(3, 4)
    expanded = base[:, 0:1].expand(3, 2)        # stride (4, 0), offset 0
    p = torch.nn.Parameter(torch.randn(3, 2))

    def corrupt(o):
        st = o.state[p]
        o.base_ref[:, 0].fill_(-7.0)            # seen THROUGH expand
        o.base_ref.fill_(-123.25)               # incl. hidden columns

    opt = _ArmedPoisonSGD([p], lr=0.05, corrupt=corrupt)
    opt.base_ref = base
    opt.state[p] = {"momentum_buffer": expanded}
    pre_bytes, pre_exp = _storage_snapshot(base), _leaf_snapshot(expanded)

    opt.armed = True
    with pytest.raises(RuntimeError, match="adversarial"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0)

    live = opt.state[p]["momentum_buffer"]
    assert torch.equal(_u8(live), pre_bytes), \
        "hidden expand columns (outside the visible stride pattern) differ"
    assert tuple(live.stride()) == (4, 0) and \
        live.storage_offset() == 0, \
        "zero-stride expand metadata was normalized instead of restored"
    assert _leaf_exact(live, pre_exp)


# ---------------------------------------------------------------------------
# 4: scalar and empty tensors; distinct empty storages not merged
# ---------------------------------------------------------------------------

def test_scalars_and_empties_exact_with_distinct_empty_storages():
    backing = torch.arange(16, dtype=torch.float32)
    empty_off = backing[9:9]                    # empty WITH storage offset
    ea, eb = torch.empty(0), torch.empty(0)     # DISTINCT empty storages
    ea_cd, eb_cd = _cdata(ea), _cdata(eb)
    p = torch.nn.Parameter(torch.randn(4))
    scalar = torch.tensor(7.0)

    def corrupt(o):
        st = o.state[p]
        st["scalar"].fill_(999.0)
        st["empty_off"] = torch.empty(0)        # metadata swap
        st["eb"] = st["ea"]                     # merge distinct empties
        st["momentum_buffer"].zero_()

    opt = _ArmedPoisonSGD([p], lr=0.05, corrupt=corrupt)
    opt.state[p] = {"momentum_buffer": torch.randn(4), "scalar": scalar,
                    "empty_off": empty_off, "ea": ea, "eb": eb}
    pre_momentum = opt.state[p]["momentum_buffer"].clone()

    opt.armed = True
    with pytest.raises(RuntimeError, match="adversarial"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0)

    st = opt.state[p]
    assert _leaf_exact(st["scalar"], scalar), "scalar leaf not exact"
    assert float(st["scalar"]) == 7.0
    assert tuple(st["empty_off"].shape) == (0,) and \
        st["empty_off"].storage_offset() == 9 and \
        tuple(st["empty_off"].stride()) == (1,), \
        "empty-with-offset view was normalized to a fresh dense empty"
    assert st["ea"] is not st["eb"], "distinct empties were merged"
    assert _cdata(st["ea"]) == ea_cd and _cdata(st["eb"]) == eb_cd and \
        ea_cd != eb_cd, \
        "original distinct empty storages were not preserved exactly"
    assert _leaf_exact(st["momentum_buffer"], pre_momentum)


# ---------------------------------------------------------------------------
# 5: cross-dtype views sharing one untyped storage
# ---------------------------------------------------------------------------

def test_cross_dtype_views_share_one_untyped_storage_after_rollback():
    raw = torch.arange(8, dtype=torch.float32) * 3.0
    momentum = raw.view(2, 4)
    byte_view = raw.view(torch.int32)[1:3]      # cross-dtype alias
    p = torch.nn.Parameter(torch.randn(2, 4))

    def corrupt(o):
        st = o.state[p]
        st["byte_view"].fill_(1 << 30)          # corrupt f32 bytes
        o.raw_ref.fill_(-0.5)                   # rewrite whole storage
        st["byte_view"] = torch.zeros(2, dtype=torch.int32)  # rebind away

    opt = _ArmedPoisonSGD([p], lr=0.05, corrupt=corrupt)
    opt.raw_ref = raw
    opt.state[p] = {"momentum_buffer": momentum, "byte_view": byte_view}
    pre_bytes, pre_f, pre_i = (_storage_snapshot(raw),
                               momentum.clone(), _leaf_snapshot(byte_view))

    opt.armed = True
    with pytest.raises(RuntimeError, match="adversarial"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0)

    st = opt.state[p]
    assert _cdata(st["byte_view"]) == _cdata(st["momentum_buffer"]), \
        "cross-dtype views no longer share one untyped storage"
    assert st["byte_view"].dtype == torch.int32 and \
        st["byte_view"].storage_offset() == pre_i.storage_offset()
    assert torch.equal(_u8(st["momentum_buffer"]), pre_bytes), \
        "shared untyped-storage bytes were not restored exactly"
    assert _leaf_exact(st["momentum_buffer"], pre_f)
    assert _leaf_exact(st["byte_view"], pre_i)


# ---------------------------------------------------------------------------
# 7: hostile replacement/rebinding of optimizer.state itself
# ---------------------------------------------------------------------------

def test_hostile_optimizer_state_rebinding_restores_original_mapping():
    p = torch.nn.Parameter(torch.randn(3))
    buf = torch.randn(3)

    def corrupt(o):
        o.state[p]["momentum_buffer"].fill_(66.0)
        o.state = {"evil": "replacement"}       # hostile rebinding

    opt = _ArmedPoisonSGD([p], lr=0.05, corrupt=corrupt)
    opt.state[p] = {"momentum_buffer": buf}
    original = opt.state
    pre_buf = buf.clone()

    opt.armed = True
    with pytest.raises(RuntimeError, match="adversarial"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0)

    assert opt.state is original, \
        "hostilely-rebound optimizer.state was not reinstated by identity"
    assert "evil" not in opt.state, "injected top-level key survived"
    assert _leaf_exact(opt.state[p]["momentum_buffer"], pre_buf)


# ---------------------------------------------------------------------------
# 8: unsupported sparse/meta/quantized/subclass graphs rejected preflight
# ---------------------------------------------------------------------------

def _unsupported_cases():
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), (2,))
    meta = torch.empty(2, device="meta")
    quantized = torch.quantize_per_tensor(
        torch.tensor([1.0]), 1.0, 0, torch.qint8)

    class _StateSubclass(torch.Tensor):
        pass

    subclass = _StateSubclass(torch.ones(2))
    return [("sparse", sparse), ("meta", meta),
            ("quantized", quantized), ("subclass", subclass)]


@pytest.mark.parametrize("name", [n for n, _ in _unsupported_cases()])
def test_unsupported_state_graph_rejected_before_clipping(name):
    from latent_lab.train.checkpointing import UnsupportedOptimizerStateError
    case = dict(_unsupported_cases())[name]
    p = torch.nn.Parameter(torch.randn(3, 2))
    opt = torch.optim.SGD([p], lr=0.05, momentum=0.9)
    opt.state[p] = {"momentum_buffer": torch.randn(3, 2), "junk": case}
    pre_params = p.detach().clone()
    pre_grad = _fresh_grad(p, 3)
    p.grad = pre_grad.clone()
    assert float(pre_grad.norm()) > 1.5, \
        "grad norm must exceed the clip to expose any scaling"

    with pytest.raises(UnsupportedOptimizerStateError):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0)

    assert torch.equal(p.grad, pre_grad), \
        "clipping ran (or grads moved) despite failed state preflight"
    assert torch.equal(p.detach(), pre_params), \
        "parameters were disturbed by the preflight rejection"
    assert set(opt.state[p].keys()) == {"momentum_buffer", "junk"}, \
        "optimizer state was disturbed by the preflight rejection"


# ---------------------------------------------------------------------------
# 9: unique-storage byte budget enforced before mutation
# ---------------------------------------------------------------------------

def test_storage_byte_budget_rejected_before_any_mutation():
    from latent_lab.train.checkpointing import (
        StateStorageBudgetExceededError)
    p = torch.nn.Parameter(torch.randn(3, 2))
    buf = torch.randn(3, 2)
    opt = _planted_sgd(p, buf)
    pre_params, pre_buf = p.detach().clone(), buf.clone()
    pre_grad = _fresh_grad(p, 4)
    p.grad = pre_grad.clone()

    with pytest.raises(StateStorageBudgetExceededError):
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0,
                               state_byte_budget=8)  # < 24 unique bytes

    assert torch.equal(p.grad, pre_grad), \
        "clipping ran despite budget rejection"
    assert torch.equal(p.detach(), pre_params)
    assert _leaf_exact(opt.state[p]["momentum_buffer"], pre_buf)

    # a budget exactly covering the unique storage must succeed
    nbytes = buf.untyped_storage().nbytes()
    p.grad = pre_grad.clone()
    guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0,
                           state_byte_budget=nbytes)
    assert not torch.equal(p.detach(), pre_params), "step did not run"


# ---------------------------------------------------------------------------
# 10: original exception identity/message/trace relationship preserved
# ---------------------------------------------------------------------------

class _BoomMapping(dict):
    """Mapping whose traversal explodes — a pristine per-dict rollback
    would crash here and chain a cause onto the original exception."""

    def keys(self):
        raise RuntimeError("boom-mapping traversal")

    items = values = keys


def test_original_step_exception_identity_and_cause_relationship():
    class _DistinctiveError(RuntimeError):
        pass

    p = torch.nn.Parameter(torch.randn(3))
    buf = torch.randn(3)

    def corrupt(o):
        o.state[p]["momentum_buffer"].fill_(9.0)
        o.state = _BoomMapping({})              # sabotage state traversal

    opt = _ArmedPoisonSGD([p], lr=0.05, corrupt=corrupt,
                          error=_DistinctiveError("kaboom-msg"))
    opt.state[p] = {"momentum_buffer": buf}
    pre_buf = buf.clone()

    opt.armed = True
    with pytest.raises(_DistinctiveError,
                       match=r"\bkaboom-msg\b") as e:
        guarded_optimizer_step(opt, torch.tensor(0.5), [p], 1.0)

    assert type(e.value) is _DistinctiveError, \
        "original step exception type was replaced"
    assert e.value.__cause__ is None, \
        "rollback failure leaked as __cause__ though rollback succeeded"
    assert e.value.__suppress_context__ is False
    assert e.value.__traceback__ is not None
    assert _leaf_exact(opt.state[p]["momentum_buffer"], pre_buf), \
        "state not restored across hostile mapping sabotage"


# ---------------------------------------------------------------------------
# 11: one successful retry exactly matching an independent clean control
# ---------------------------------------------------------------------------

def _aliased_pair(seed):
    """Candidate + control with IDENTICAL aliased momentum topology:
    momentum and twin are overlapping views of one backing storage."""
    torch.manual_seed(seed)
    pc = torch.nn.Parameter(torch.randn(3, 2))
    pk = torch.nn.Parameter(torch.zeros(3, 2))
    pk.data.copy_(pc.data)
    bc, bk = torch.randn(12), None
    bk = bc.clone()
    mom_c, mom_k = bc[2:8].view(3, 2), bk[2:8].view(3, 2)
    twin_c, twin_k = bc[6:12].view(3, 2), bk[6:12].view(3, 2)
    init = torch.randn(3, 2, generator=torch.Generator().manual_seed(9))
    mom_c.copy_(init)
    mom_k.copy_(init)
    twin_c.fill_(0.25)
    twin_k.copy_(twin_c)

    def corrupt(o):
        o.state[pc]["momentum_buffer"].add_(100.0)
        o.state[pc]["twin"].fill_(-50.0)
        o.backing.fill_(-1.0)

    oc = _ArmedPoisonSGD([pc], lr=0.05, corrupt=corrupt)
    oc.backing = bc
    oc.state[pc] = {"momentum_buffer": mom_c, "twin": twin_c}
    ok = torch.optim.SGD([pk], lr=0.05, momentum=0.9)
    ok.state[pk] = {"momentum_buffer": mom_k, "twin": twin_k}
    return pc, pk, oc, ok


def test_retry_after_alias_corruption_matches_clean_control_exactly():
    pc, pk, oc, ok = _aliased_pair(31)
    grad_seed = 77
    pre_mom = _leaf_snapshot(oc.state[pc]["momentum_buffer"])
    pre_twin = _leaf_snapshot(oc.state[pc]["twin"])
    pre_pc = pc.detach().clone()
    calls_before = oc.calls

    oc.armed = True
    pc.grad = _fresh_grad(pc, grad_seed)
    with pytest.raises(RuntimeError, match="adversarial corruption"):
        guarded_optimizer_step(oc, torch.tensor(0.5), [pc], 1.0)
    assert oc.calls == calls_before + 1

    # rolled back bit-exactly with aliases intact...
    assert _leaf_exact(oc.state[pc]["momentum_buffer"], pre_mom)
    assert _leaf_exact(oc.state[pc]["twin"], pre_twin)
    assert _cdata(oc.state[pc]["twin"]) == \
        _cdata(oc.state[pc]["momentum_buffer"])
    assert torch.equal(pc.detach(), pre_pc)

    # ...exactly ONE disarmed retry, mirroring the clean control step
    oc.armed = False
    pc.grad = _fresh_grad(pc, grad_seed)
    guarded_optimizer_step(oc, torch.tensor(0.5), [pc], 1.0)
    assert oc.calls == calls_before + 2
    pk.grad = _fresh_grad(pk, grad_seed)
    torch.nn.utils.clip_grad_norm_([pk], 1.0)
    ok.step()

    assert torch.equal(pc.detach(), pk.detach()), \
        "post-retry parameter trajectory diverged from the clean control"
    assert torch.equal(oc.state[pc]["momentum_buffer"],
                       ok.state[pk]["momentum_buffer"]), \
        "post-retry momentum diverged from the clean control"
    assert torch.equal(oc.state[pc]["twin"], ok.state[pk]["twin"]), \
        "overlapping alias view diverged from the clean control " \
        "(severed alias group)"
