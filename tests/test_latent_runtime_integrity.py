"""Runtime-integrity guarantees: checkpointing, fail-closed stepping, bundles.

All tests are tiny, deterministic, CPU-only and never touch the network or a
retained checkpoint (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are forced below;
the hybrid model is built locally from a config with a fixed seed).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from latent_lab.backends.localized import (
    LoRALinear,
    LocalizedRecurrence,
    make_step_clock,
)
from latent_lab.train.checkpointing import (
    AdapterBundleError,
    AdapterBundleIdentityError,
    AdapterBundleSchemaError,
    BestCheckpointTracker,
    EmptyCheckpointError,
    NonFiniteMetricError,
    NonFiniteStateError,
    NonFiniteTrainingStateError,
    OptimizerParamCoverageError,
    guarded_optimizer_step,
    load_adapter_bundle,
    require_pinned_revision,
    save_adapter_bundle,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Pinned immutable commit-style revisions (40 hex chars). Every bundle
# identity flow must use one of these; mutable refs such as "main",
# "latest", branch names or tags must be rejected fail-closed.
REV_A = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
REV_B = "0f9e8d7c6b5a49382715040302010ffedcba9876"

def _state(value: float, key: str = "w") -> dict:
    return {key: torch.full((3, 3), value)}


def _snapshot(params) -> dict:
    return {id(p): p.detach().clone() for p in params}


def _unchanged(params, snap) -> bool:
    return all(bool(torch.equal(p.detach(), snap[id(p)])) for p in params)


def _deep_opt_state(opt) -> dict:
    return {p: {k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in st.items()} for p, st in opt.state.items()}


def _opt_state_unchanged(opt, snap) -> bool:
    if set(opt.state.keys()) != set(snap.keys()):
        return False
    for p, entry in snap.items():
        live = opt.state[p]
        if set(live.keys()) != set(entry.keys()):
            return False
        for k, v in entry.items():
            lv = live[k]
            if torch.is_tensor(v) != torch.is_tensor(lv):
                return False
            if torch.is_tensor(v) and not bool(torch.equal(lv, v)):
                return False
            if not torch.is_tensor(v) and lv is not v:
                return False
    return True


def _param_groups_snapshot(opt) -> list:
    return [{k: (v.detach().clone() if torch.is_tensor(v)
                 else _copy_value(v))
             for k, v in g.items() if k != "params"}
            for g in opt.param_groups]


def _copy_value(v):
    import copy as _copy
    return _copy.deepcopy(v)


def _param_groups_unchanged(opt, snap) -> bool:
    if len(opt.param_groups) != len(snap):
        return False
    for g, sg in zip(opt.param_groups, snap):
        if set(g.keys()) - {"params"} != set(sg.keys()):
            return False
        for k, v in sg.items():
            lv = g[k]
            if torch.is_tensor(v) != torch.is_tensor(lv):
                return False
            if torch.is_tensor(v):
                if not bool(torch.equal(lv, v)):
                    return False
            elif lv != v or type(lv) is not type(v):
                return False
    return True


def _param_group_param_ids(opt) -> list:
    return [[id(p) for p in g["params"]] for g in opt.param_groups]


class _FakeLora:
    def __init__(self, r, fin, fout):
        self.lora_A = torch.nn.Parameter(torch.zeros(r, fin))
        self.lora_B = torch.nn.Parameter(torch.zeros(fout, r))


def _fake_rec(use_clock: bool = True):
    rec = SimpleNamespace(
        injected=[_FakeLora(2, 4, 4)],
        use_clock=use_clock,
        clock=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(3, 4)))
        if use_clock else None,
    )
    rec._expected_adapter_targets = (
        lambda: LocalizedRecurrence._expected_adapter_targets(rec))
    return rec


def _tiny_qwen35(seed: int = 11, layers: int = 6):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig)

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, head_dim=16, num_key_value_heads=2,
        linear_num_value_heads=8, linear_num_key_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
        vocab_size=512, max_position_embeddings=256,
        tie_word_embeddings=True)
    return AutoModelForCausalLM.from_config(cfg).eval()


def _nonzero_adapted_rec(model, seed: int = 5, interval=None):
    if interval is None:
        interval = (0, model.config.num_hidden_layers)
    rec = LocalizedRecurrence(
        model, None, interval=interval,
        max_k=4, lora_r=4, lora_alpha=8, grad_checkpoint=False)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for l in rec.injected:
            l.lora_B.copy_(
                torch.randn(l.lora_B.shape, generator=g) * 0.05)
    return rec


_IDS = torch.tensor([[5, 9, 13, 7, 200, 3, 42]])
_ANS = [100, 101]
_CANDS = [[100, 101], [102]]


def _cached_matches_full(rec) -> tuple:
    """Cached candidate ranking must match a full recomputed pass exactly."""
    order, scores, _rep = rec.rank_candidates(_IDS, _CANDS, 2)
    with torch.no_grad():
        ce = rec.loss_on_example(_IDS, torch.tensor([_ANS]), 2)
    full_sum_logp = -float(ce) * len(_ANS)
    cached_gold = scores[_CANDS.index(_ANS)]
    assert abs(cached_gold - full_sum_logp) < 1e-5, (
        f"cached {cached_gold} != full {full_sum_logp}")
    return order, scores


# ---------------------------------------------------------------------------
# best-checkpoint tracking
# ---------------------------------------------------------------------------

def test_best_checkpoint_selection_reload_before_report_and_save(tmp_path):
    tr = BestCheckpointTracker()
    s1, s2, s3 = _state(1.0), _state(2.0), _state(3.0)
    assert tr.update(0.5, s1, step=1) is True
    assert tr.update(0.9, s2, step=2) is True
    assert tr.update(0.7, s3, step=3) is False          # worse: ignored
    assert tr.update(0.9, s3, step=4) is False          # tie: keep earliest
    assert tr.best_score == 0.9 and tr.best_step == 2.0

    s2["w"].fill_(999.0)                                # live pollution
    applied = []
    tr.apply_best(applied.append)                       # reload selected best
    assert len(applied) == 1
    assert bool(torch.equal(applied[0]["w"], torch.full((3, 3), 2.0)))

    path = tmp_path / "best_params.pt"
    tr.save(path, model_id="m", revision=REV_A)
    loaded = load_adapter_bundle(path, model_id="m", revision=REV_A)
    assert bool(torch.equal(loaded["w"], torch.full((3, 3), 2.0)))

    empty = BestCheckpointTracker()
    with pytest.raises(EmptyCheckpointError):
        empty.best_state()
    with pytest.raises(EmptyCheckpointError):
        empty.apply_best(lambda st: st)
    with pytest.raises(EmptyCheckpointError):
        empty.save(tmp_path / "never.pt", model_id="m", revision=REV_A)
    assert not (tmp_path / "never.pt").exists()


def test_nan_metric_and_nan_state_rejected():
    tr = BestCheckpointTracker()
    with pytest.raises(NonFiniteMetricError):
        tr.update(float("nan"), _state(1.0), step=1)
    with pytest.raises(NonFiniteMetricError):
        tr.update(float("inf"), _state(1.0), step=1)
    with pytest.raises(NonFiniteMetricError):
        tr.update("not-a-number", _state(1.0), step=1)
    assert not tr.has_best()

    assert tr.update(0.25, _state(1.0), step=1) is True
    with pytest.raises(NonFiniteMetricError):
        tr.update(float("inf"), _state(2.0), step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {"w": torch.tensor([0.1, float("nan")])}, step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {"w": torch.tensor([float("inf")])}, step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {"w": "not-a-tensor"}, step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {}, step=2)
    assert tr.has_best() and tr.best_score == 0.25 and tr.best_step == 1.0
    assert bool(torch.equal(tr.best_state()["w"], torch.ones(3, 3)))


# ---------------------------------------------------------------------------
# fail-closed optimizer stepping
# ---------------------------------------------------------------------------

def test_optimizer_step_fail_closed():
    class _PoisonedSGD(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(float("inf"))

    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    params = list(m.parameters())

    def loss_fn():
        return (m(x) ** 2).mean()

    def fresh_grads():
        m.zero_grad(set_to_none=True)
        loss_fn().backward()

    snap = _snapshot(params)
    fresh_grads()
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt := torch.optim.SGD(params, lr=0.1),
                               torch.tensor(float("nan")), params, 1.0)
    assert _unchanged(params, snap), "stepped on non-finite loss"

    fresh_grads()
    with torch.no_grad():
        m.weight.grad[0, 0] = float("nan")
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, loss_fn(), params, 1.0)
    assert _unchanged(params, snap), "stepped on non-finite gradient"

    fresh_grads()
    with torch.no_grad():
        m.weight.grad.fill_(float("inf"))
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, loss_fn(), params, 1.0)
    assert bool(torch.isinf(m.weight.grad).all()), \
        "clip silently scaled non-finite grads instead of failing closed"
    assert _unchanged(params, snap), "stepped on non-finite gradient norm"

    fresh_grads()
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, loss_fn(), params, float("nan"))
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, loss_fn(), params, 0.0)
    assert _unchanged(params, snap)

    poisoned = _PoisonedSGD(params, lr=0.1)
    fresh_grads()
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(poisoned, loss_fn(), params, 1.0)
    assert _unchanged(params, snap), \
        "post-update poisoning was not rolled back bit-exactly"

    m2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    before = m2.weight.detach().clone()
    (m2(x) ** 2).mean().backward()
    guarded_optimizer_step(opt2, (m2(x) ** 2).mean(),
                           list(m2.parameters()), 1.0)
    assert not bool(torch.equal(m2.weight.detach(), before)), \
        "happy-path step did not run"


class _PoisonParamAdam(torch.optim.Adam):
    """Adam that corrupts a parameter after its own update once armed;
    it also REORDERS the group's params list so the rollback must undo
    per-group Parameter topology, not merely values."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
            self.param_groups[0]["params"].reverse()
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(float("inf"))


class _PoisonStateAdam(torch.optim.Adam):
    """Adam that corrupts its own state after a real update once armed;
    it also SWAPS the first and last group Parameters so the rollback
    must restore the exact per-group Parameter identity order."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
            plist = self.param_groups[0]["params"]
            plist[0], plist[-1] = plist[-1], plist[0]
            with torch.no_grad():
                self.state[self.param_groups[0]["params"][0]][
                    "exp_avg"].fill_(float("nan"))


def _adam_with_nonempty_state():
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    params = list(m.parameters())
    opt = torch.optim.Adam(params, lr=0.1)
    (m(x) ** 2).mean().backward()
    guarded_optimizer_step(opt, (m(x) ** 2).mean(), params, 1.0)
    assert opt.state, "optimizer state must be non-empty for this test"
    return m, x, params, opt


def test_prepoisoned_parameter_rejected_without_mutation():
    m, x, params, opt = _adam_with_nonempty_state()

    with torch.no_grad():
        m.bias.fill_(float("inf"))          # poison the parameter itself
    m.zero_grad(set_to_none=True)
    with torch.no_grad():                   # grads stay finite on purpose
        for p in params:
            p.grad = torch.randn_like(p) * 0.01
    param_snap = _snapshot(params)          # baseline = the poisoned state
    state_snap = _deep_opt_state(opt)
    finite_loss = torch.tensor(0.5)         # reported loss stays finite
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, finite_loss, params, 1.0)
    assert _unchanged(params, param_snap), "poisoned parameter was stepped"
    assert _opt_state_unchanged(opt, state_snap), \
        "rejected step mutated nested optimizer state"
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in params), "gradients were disturbed by rejection"

    # recovery: after unpoisoning, the same optimizer steps cleanly
    with torch.no_grad():
        m.bias.zero_()
        for p in params:
            p.grad.fill_(0.01)
    guarded_optimizer_step(opt, finite_loss, params, 1.0)
    assert not _unchanged(params, param_snap), "recovery step did not run"


@pytest.mark.parametrize("bad_clip", [float("nan"), float("inf"),
                                      0.0, -1.0])
def test_nan_and_zero_clip_rejected_without_step_or_mutation(bad_clip):
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    params = list(m.parameters())
    opt = torch.optim.SGD(params, lr=0.1)
    (m(x) ** 2).mean().backward()
    param_snap = _snapshot(params)
    state_snap = _deep_opt_state(opt)
    grads_before = [p.grad.clone() for p in params]

    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, (m(x) ** 2).mean(), params, bad_clip)
    assert _unchanged(params, param_snap), "stepped on invalid clip config"
    assert _opt_state_unchanged(opt, state_snap)
    for p, g0 in zip(params, grads_before):
        assert torch.equal(p.grad, g0), "clipping ran before clip validation"


def test_poststep_poisoned_parameter_rolls_back_params_and_state():
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    params = list(m.parameters())
    opt = _PoisonParamAdam(params, lr=0.1)

    (m(x) ** 2).mean().backward()           # disarmed: builds real state
    guarded_optimizer_step(opt, (m(x) ** 2).mean(), params, 1.0)
    assert opt.state, "optimizer state must be non-empty before arming"

    param_snap = _snapshot(params)
    state_snap = _deep_opt_state(opt)
    opt.armed = True
    m.zero_grad(set_to_none=True)
    (m(x) ** 2).mean().backward()
    grad_snap = [p.grad.detach().clone() for p in params]
    groups_snap = _param_groups_snapshot(opt)
    ids_before = _param_group_param_ids(opt)
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, (m(x) ** 2).mean(), params, 1.0)
    assert _unchanged(params, param_snap), \
        "post-step poisoned parameter survived the rollback"
    assert _opt_state_unchanged(opt, state_snap), \
        "nested optimizer state differs after rollback"
    for p, g0 in zip(params, grad_snap):
        assert p.grad is not None and bool(torch.equal(p.grad, g0)), \
            "pre-clip gradients were not restored bit-exactly"
    assert _param_groups_unchanged(opt, groups_snap), \
        "param_groups fields differ after rollback"
    assert _param_group_param_ids(opt) == ids_before, \
        "rollback replaced the live Parameter objects in param_groups"


def test_poststep_poisoned_optimizer_state_rolls_back_exactly():
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    params = list(m.parameters())
    poisoned = _PoisonStateAdam(params, lr=0.1)

    (m(x) ** 2).mean().backward()           # first: create non-empty state
    guarded_optimizer_step(poisoned, (m(x) ** 2).mean(), params, 1.0)
    state_snap = _deep_opt_state(poisoned)
    assert any("exp_avg" in st for st in state_snap.values())
    param_snap = _snapshot(params)

    poisoned.armed = True                   # now the step poisons its state
    m.zero_grad(set_to_none=True)
    (m(x) ** 2).mean().backward()
    grad_snap = [p.grad.detach().clone() for p in params]
    groups_snap = _param_groups_snapshot(poisoned)
    ids_before = _param_group_param_ids(poisoned)
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(poisoned, (m(x) ** 2).mean(), params, 1.0)
    assert _unchanged(params, param_snap)
    assert _opt_state_unchanged(poisoned, state_snap), \
        "nested optimizer state is not byte-exact after rollback"
    for p, g0 in zip(params, grad_snap):
        assert p.grad is not None and bool(torch.equal(p.grad, g0)), \
            "pre-clip gradients were not restored bit-exactly"
    assert _param_groups_unchanged(poisoned, groups_snap), \
        "param_groups fields differ after rollback"
    assert _param_group_param_ids(poisoned) == ids_before, \
        "rollback replaced the live Parameter objects in param_groups"


class _AdversarialSGD(torch.optim.SGD):
    """Performs a REAL update, then corrupts parameters, tensor and
    non-tensor optimizer state, ``param_groups`` (LR + injected
    metadata) and the per-group ``params`` order, and finally raises —
    the worst-case mid-step failure."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        with torch.no_grad():
            for group in self.param_groups:
                group["lr"] = 123.0
                group["injected_meta"] = {"evil": [1, 2, 3]}
            for state in self.state.values():
                if "momentum_buffer" in state:
                    state["momentum_buffer"].add_(11.0)
                    state["poison_flag"] = "corrupted"
            self.param_groups[0]["params"].reverse()
            self.param_groups[0]["params"][0].add_(7.0)
        raise RuntimeError("adversarial failure after full corruption")


def _adversarial_pair(seed):
    """Two identically-initialized model/optimizer pairs: candidate under
    test and an independent control that receives the same initial state."""
    torch.manual_seed(seed)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    x = torch.randn(8, 4)

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    opt_adv = _AdversarialSGD(ps_adv, lr=0.05)
    opt_ctl = torch.optim.SGD(ps_ctl, lr=0.05, momentum=0.9)
    # one clean step on both sides -> identical non-empty momentum state
    loss_a = fresh_grads(m_adv)
    guarded_optimizer_step(opt_adv, loss_a.detach(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    opt_ctl.step()
    assert bool(torch.equal(m_adv.weight.detach(), m_ctl.weight.detach()))
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip(ps_adv, ps_ctl)), \
        "pairs did not start from identical optimizer state"
    return m_adv, m_ctl, x, ps_adv, ps_ctl, opt_adv, opt_ctl, fresh_grads


def test_adversarial_step_rolls_back_everything_then_matches_control():
    m_adv, m_ctl, x, ps_adv, ps_ctl, opt_adv, opt_ctl, fresh_grads = \
        _adversarial_pair(21)

    # -- armed failure after a real, corrupting update ----------------------
    opt_adv.armed = True
    fresh_grads(m_adv)
    with torch.no_grad():
        m_adv.bias.grad = None              # None-vs-tensor grad pattern
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in ps_adv}
    pre_state = _deep_opt_state(opt_adv)
    pre_groups = _param_groups_snapshot(opt_adv)
    pre_ids = _param_group_param_ids(opt_adv)

    with pytest.raises(RuntimeError, match="adversarial failure"):
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)

    # full rollback: bit-exact everything, injected fields gone
    assert _unchanged(ps_adv, pre_params), "param corruption survived"
    for p in ps_adv:
        saved = pre_grads[id(p)]
        if saved is None:
            assert p.grad is None, "None gradient became a tensor"
        else:
            assert p.grad is not None and bool(torch.equal(p.grad, saved))
    assert _opt_state_unchanged(opt_adv, pre_state), \
        "tensor/non-tensor optimizer state not restored bit-exactly"
    assert all("poison_flag" not in st for st in opt_adv.state.values()), \
        "injected optimizer-state field survived rollback"
    assert _param_groups_unchanged(opt_adv, pre_groups), \
        "param_groups metadata not equality-identical after rollback"
    assert all("injected_meta" not in g for g in opt_adv.param_groups), \
        "injected param-group field survived rollback"
    assert any(g["lr"] == 0.05 for g in opt_adv.param_groups), \
        "mutated LR was not rolled back"
    assert _param_group_param_ids(opt_adv) == pre_ids, \
        "live Parameter identity/order broken by rollback"

    # -- disarmed retry of the SAME update vs independent control ----------
    opt_adv.armed = False
    fresh_grads(m_adv)
    with torch.no_grad():
        m_adv.bias.grad = None              # control mirrors this pattern
    guarded_optimizer_step(opt_adv, (m_adv(x) ** 2).mean(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    with torch.no_grad():
        m_ctl.bias.grad = None
    opt_ctl.step()

    assert bool(torch.equal(m_adv.weight.detach(), m_ctl.weight.detach())), \
        "post-retry trajectory diverges from the clean control"
    assert bool(torch.equal(m_adv.bias.detach(), m_ctl.bias.detach()))
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip(ps_adv, ps_ctl)), \
        "optimizer state diverges from the clean control after retry"
    assert all(g["lr"] == 0.05 and "injected_meta" not in g
               for g in opt_adv.param_groups)


class _TopologyAdversarySGD(torch.optim.SGD):
    """Two-group SGD that performs a REAL update, then — once armed —
    rewrites the param-group TOPOLOGY (Parameters swapped between groups,
    reordered within groups, foreign additions and removals, an extra
    group appended or a group removed), mutates hyperparameters and
    optimizer state, and either raises (mode="raise") or silently poisons
    a parameter so only the post-step check can react (mode="poison")."""

    def __init__(self, groups, lrs):
        super().__init__([{"params": list(g), "lr": lr}
                          for g, lr in zip(groups, lrs)], momentum=0.9)
        self.armed = False
        self.mode = "raise"

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        if self.mode == "raise":
            g0, g1 = self.param_groups[0], self.param_groups[1]
            g0["params"], g1["params"] = g1["params"], g0["params"]
            g1["params"].reverse()                       # reorder
            g0["params"].append(                         # foreign addition
                torch.nn.Parameter(torch.zeros(1)))
            del g1["params"][0]                          # removal
            self.param_groups.append({"params": [], "lr": 9.9})
            for group in self.param_groups:
                group["lr"] = 123.0
                group["injected_meta"] = {"evil": [1, 2, 3]}
            for st in self.state.values():
                if "momentum_buffer" in st:
                    st["momentum_buffer"].add_(11.0)
                    st["poison_flag"] = "corrupted"
            raise RuntimeError("topology adversarial failure")
        # poison mode: remove group 1, merge + reverse into group 0 via a
        # brand-new list object, then poison WITHOUT raising
        g0 = self.param_groups[0]
        merged = list(g0["params"]) + list(self.param_groups[1]["params"])
        del self.param_groups[1]
        g0["params"] = list(reversed(merged))
        for group in self.param_groups:
            group["lr"] = 123.0
            group["injected_meta"] = {"evil": [1]}
        with torch.no_grad():
            g0["params"][0].add_(float("inf"))


def _two_group_pair(seed):
    """Identically-initialized model/optimizer pairs with TWO param groups
    at different learning rates ([p0, p1] @ lr 0.05; [p2, p3] @ 0.2)."""
    torch.manual_seed(seed)
    m_adv = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    m_ctl = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    m_ctl.load_state_dict(m_adv.state_dict())
    x = torch.randn(8, 4)

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    opt_adv = _TopologyAdversarySGD([ps_adv[:2], ps_adv[2:]], [0.05, 0.2])
    opt_ctl = torch.optim.SGD([
        {"params": ps_ctl[:2], "lr": 0.05},
        {"params": ps_ctl[2:], "lr": 0.2}], momentum=0.9)
    # one clean step on both sides -> identical non-empty momentum state
    loss_a = fresh_grads(m_adv)
    guarded_optimizer_step(opt_adv, loss_a.detach(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    opt_ctl.step()
    assert all(torch.equal(a.detach(), b.detach())
               for a, b in zip(ps_adv, ps_ctl)), \
        "pairs did not start from identical parameters"
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip(ps_adv, ps_ctl)), \
        "pairs did not start from identical optimizer state"
    return m_adv, m_ctl, x, ps_adv, ps_ctl, opt_adv, opt_ctl, fresh_grads


def test_adversarial_topology_rollback_restores_full_group_structure():
    m_adv, m_ctl, x, ps_adv, ps_ctl, opt_adv, opt_ctl, fresh_grads = \
        _two_group_pair(33)

    # -- armed raise-mode failure after a real, topology-corrupting update --
    opt_adv.armed = True
    opt_adv.mode = "raise"
    fresh_grads(m_adv)
    with torch.no_grad():
        m_adv[0].bias.grad = None           # None-vs-tensor grad pattern
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in ps_adv}
    pre_state = _deep_opt_state(opt_adv)
    pre_meta = _param_groups_snapshot(opt_adv)
    pre_group_objs = list(opt_adv.param_groups)
    pre_list_objs = [g["params"] for g in opt_adv.param_groups]
    pre_ids = _param_group_param_ids(opt_adv)

    with pytest.raises(RuntimeError, match="topology adversarial failure"):
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)

    # exact original group count/order as the ORIGINAL dict objects ...
    assert len(opt_adv.param_groups) == len(pre_group_objs)
    assert all(g is orig for g, orig
               in zip(opt_adv.param_groups, pre_group_objs)), \
        "rollback did not restore the original group objects in order"
    # ... with the ORIGINAL params list objects ...
    assert all(gl is lo for gl, lo
               in zip((g["params"] for g in opt_adv.param_groups),
                      pre_list_objs)), \
        "rollback did not restore the original params list objects"
    # ... holding the original Parameter identities in the original order
    assert _param_group_param_ids(opt_adv) == pre_ids, \
        "per-group Parameter identity/order was not restored"
    assert _param_groups_unchanged(opt_adv, pre_meta), \
        "non-params group fields differ after topology rollback"
    assert all("injected_meta" not in g for g in opt_adv.param_groups), \
        "injected group field survived rollback"
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05, 0.2], \
        "per-group hyperparameters were not restored"
    assert _unchanged(ps_adv, pre_params), \
        "parameter corruption survived the topology rollback"
    for p in ps_adv:
        saved = pre_grads[id(p)]
        if saved is None:
            assert p.grad is None, "None gradient became a tensor"
        else:
            assert p.grad is not None and bool(torch.equal(p.grad, saved))
    assert _opt_state_unchanged(opt_adv, pre_state), \
        "optimizer state not restored bit-exactly after topology attack"
    assert all("poison_flag" not in st for st in opt_adv.state.values()), \
        "injected optimizer-state field survived rollback"

    # -- disarmed retry of the SAME update vs independent control ----------
    opt_adv.armed = False
    fresh_grads(m_adv)
    with torch.no_grad():
        m_adv[0].bias.grad = None           # control mirrors this pattern
    guarded_optimizer_step(opt_adv, (m_adv(x) ** 2).mean(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    with torch.no_grad():
        m_ctl[0].bias.grad = None
    opt_ctl.step()

    assert all(torch.equal(a.detach(), b.detach())
               for a, b in zip(ps_adv, ps_ctl)), \
        "post-retry trajectory diverges from the clean control"
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip(ps_adv, ps_ctl)), \
        "optimizer state diverges from the clean control after retry"
    assert [[id(p) for p in g["params"]]
            for g in opt_adv.param_groups] == pre_ids, \
        "topology diverged from the clean control after retry"
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05, 0.2]

    # -- armed poison-mode: group REMOVAL + merge, silent inf caught only
    #    by the post-step finite check ---------------------------------------
    opt_adv.armed = True
    opt_adv.mode = "poison"
    fresh_grads(m_adv)
    pre_params = _snapshot(ps_adv)
    pre_state = _deep_opt_state(opt_adv)
    pre_meta = _param_groups_snapshot(opt_adv)
    pre_group_objs = list(opt_adv.param_groups)
    pre_ids = _param_group_param_ids(opt_adv)
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)
    assert len(opt_adv.param_groups) == 2, "removed group was not restored"
    assert all(g is orig for g, orig
               in zip(opt_adv.param_groups, pre_group_objs))
    assert _param_group_param_ids(opt_adv) == pre_ids
    assert _param_groups_unchanged(opt_adv, pre_meta)
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05, 0.2]
    assert _unchanged(ps_adv, pre_params)
    assert _opt_state_unchanged(opt_adv, pre_state)

    # the optimizer remains fully usable after this second rollback kind
    opt_adv.armed = False
    fresh_grads(m_adv)
    with torch.no_grad():
        m_adv[0].bias.grad = None
    guarded_optimizer_step(opt_adv, (m_adv(x) ** 2).mean(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    with torch.no_grad():
        m_ctl[0].bias.grad = None
    opt_ctl.step()
    assert all(torch.equal(a.detach(), b.detach())
               for a, b in zip(ps_adv, ps_ctl)), \
        "retry after group-removal rollback diverges from clean control"


def test_omitted_optimizer_parameter_rejected_before_any_mutation():
    """The caller supplies one of two optimizer-owned Parameters: the guard
    must fail closed BEFORE optimizer.step() runs — no mutation anywhere,
    full snapshot/rollback coverage never bypassed."""

    class _CountingSGD(torch.optim.SGD):
        def __init__(self, groups, lrs):
            super().__init__([{"params": list(g), "lr": lr}
                              for g, lr in zip(groups, lrs)], momentum=0.9)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            super().step(closure)

    torch.manual_seed(44)
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    w, b = list(m.parameters())
    opt = _CountingSGD([[w], [b]], [0.05, 0.2])

    m.zero_grad(set_to_none=True)
    (m(x) ** 2).mean().backward()

    pre_params = _snapshot([w, b])
    pre_grads = {id(p): p.grad.detach().clone() for p in (w, b)}
    pre_state = _deep_opt_state(opt)
    pre_ids = _param_group_param_ids(opt)
    pre_group_objs = list(opt.param_groups)

    with pytest.raises(OptimizerParamCoverageError):
        guarded_optimizer_step(opt, torch.tensor(0.5), [w], 1.0)
    assert opt.step_calls == 0, "step ran despite incomplete coverage"
    assert _unchanged([w, b], pre_params), \
        "rejected call mutated a parameter"
    for p in (w, b):
        assert bool(torch.equal(p.grad, pre_grads[id(p)])), \
            "rejected call disturbed gradients"
    assert _opt_state_unchanged(opt, pre_state), \
        "rejected call mutated optimizer state"
    assert _param_group_param_ids(opt) == pre_ids and \
        all(g is g0 for g, g0 in zip(opt.param_groups, pre_group_objs)), \
        "rejected call disturbed param-group topology"

    stranger = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(OptimizerParamCoverageError):
        guarded_optimizer_step(opt, torch.tensor(0.5), [w, b, stranger], 1.0)
    assert opt.step_calls == 0
    assert _unchanged([w, b], pre_params)


def test_duplicate_supplied_parameters_are_deterministic_and_covered():
    """Duplicates and caller-side reordering change nothing: the
    authoritative transaction set derives from the pre-step groups, so
    [b, w, b] behaves exactly like [w, b] against a clean control."""
    torch.manual_seed(45)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    x = torch.randn(6, 4)

    def grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    pw, pb = list(m_adv.parameters())
    cw, cb = list(m_ctl.parameters())
    opt_adv = torch.optim.SGD([{"params": [pw], "lr": 0.05},
                               {"params": [pb], "lr": 0.2}], momentum=0.9)
    opt_ctl = torch.optim.SGD([{"params": [cw], "lr": 0.05},
                               {"params": [cb], "lr": 0.2}], momentum=0.9)

    la = grads(m_adv)
    guarded_optimizer_step(opt_adv, la.detach(), [pb, pw, pb], 1.0)
    grads(m_ctl)
    opt_ctl.step()

    assert bool(torch.equal(pw.detach(), cw.detach())) and \
        bool(torch.equal(pb.detach(), cb.detach())), \
        "duplicate-supplied step diverged from the clean control"
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip((pw, pb), (cw, cb)))
    assert [[id(p) for p in g["params"]]
            for g in opt_adv.param_groups] == [[id(pw)], [id(pb)]]
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05, 0.2]


# ---------------------------------------------------------------------------
# fp32 trainables over a lower-precision backbone
# ---------------------------------------------------------------------------

def test_lora_and_clock_trainables_are_fp32_over_bf16_backbone():
    base = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    lora = LoRALinear(base, r=4, alpha=8.0)
    assert base.weight.dtype == torch.bfloat16
    assert lora.base.weight.dtype == torch.bfloat16
    assert lora.lora_A.dtype == torch.float32
    assert lora.lora_B.dtype == torch.float32
    x = torch.randn(2, 5, 8).to(torch.bfloat16)
    y = lora(x)
    assert y.dtype == torch.bfloat16
    assert bool(torch.isfinite(y).all())
    y.float().sum().backward()
    assert lora.lora_A.grad is not None
    assert lora.lora_A.grad.dtype == torch.float32
    assert bool(torch.isfinite(lora.lora_A.grad).all())

    clock = make_step_clock(8, 4)
    assert clock.weight.dtype == torch.float32
    assert float(clock.weight.abs().sum().detach()) == 0.0


def test_recurrence_clock_is_fp32_regardless_of_default_dtype():
    model = _tiny_qwen35(13, 4)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rec = LocalizedRecurrence(model, None, interval=(0, 4), max_k=4,
                                  lora_r=4, lora_alpha=8,
                                  grad_checkpoint=False)
        assert rec.clock.weight.dtype == torch.float32, \
            "clock dtype followed the global default"
        assert rec.clock.weight.requires_grad
    finally:
        torch.set_default_dtype(old)     # restore safely even on failure
    assert torch.get_default_dtype() == old
    assert all(p.dtype == torch.float32 for p in rec.trainable_parameters())


# ---------------------------------------------------------------------------
# identity-bound adapter bundles
# ---------------------------------------------------------------------------

def test_bundle_identity_mismatch_rejected(tmp_path):
    sd = _state(1.0, "lora.0.A")
    path = tmp_path / "best_params.pt"
    save_adapter_bundle(path, sd, model_id="model-a", revision=REV_A)

    loaded = load_adapter_bundle(path, model_id="model-a", revision=REV_A)
    assert bool(torch.equal(loaded["lora.0.A"], sd["lora.0.A"]))
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="model-a", revision=REV_B)
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="model-b", revision=REV_A)
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="model-a", revision="main")


def test_pinned_revision_acceptance_normalization_and_rejection(tmp_path):
    sd = _state(1.0, "lora.0.A")

    # accepted: pinned commits, normalized consistently (trim + lowercase)
    variants = (REV_A, REV_A.upper(), f"  {REV_A.upper()} ")
    for i, variant in enumerate(variants):
        assert require_pinned_revision(variant) == REV_A
        path = tmp_path / f"pin_{i}.pt"
        bundle = save_adapter_bundle(path, sd, model_id="m",
                                     revision=variant)
        assert bundle["revision"] == REV_A  # stored exactly as pinned
        loaded = load_adapter_bundle(path, model_id="m", revision=variant)
        assert bool(torch.equal(loaded["lora.0.A"], sd["lora.0.A"]))

    mutable = [
        "main", "latest", "HEAD", "develop", "main@{yesterday}",
        "release-1.0", "v1.2.3",                       # tags/branches
        "abcdef0",                                     # short sha
        "f" * 39, "f" * 41,                            # wrong length
        "g" * 40, REV_A[:-1] + "g",                    # non-hex
        "", "   ", None, 12345, ("a" * 40,),           # junk types
    ]
    for i, value in enumerate(mutable):
        p = tmp_path / f"reject_{i}.pt"
        with pytest.raises(AdapterBundleIdentityError):
            save_adapter_bundle(p, sd, model_id="m", revision=value)
        assert not p.exists(), "mutable revision was persisted"
        # fail closed BEFORE any file read on load, too
        with pytest.raises(AdapterBundleIdentityError):
            load_adapter_bundle(p, model_id="m", revision=value)

    tr = BestCheckpointTracker()
    tr.update(0.5, sd, step=1)
    with pytest.raises(AdapterBundleIdentityError):
        tr.save(tmp_path / "best_params.pt", model_id="m", revision="main")
    assert not (tmp_path / "best_params.pt").exists()


def test_exact_save_load_roundtrip_and_persisted_metrics(tmp_path):
    torch.manual_seed(7)
    sd = {
        "lora.0.A": torch.randn(3, 5),
        "lora.0.B": torch.randn(7, 3) * 0.01,
        "clock.weight": torch.randn(5, 5),
    }
    path = tmp_path / "best_params.pt"
    bundle = save_adapter_bundle(path, sd, model_id="M", revision=REV_A,
                                 metrics={"acc": 0.5, "loss": -1.25})
    assert not (tmp_path / "best_params.pt.tmp").exists(), "tmp file leaked"
    assert bundle["model_id"] == "M" and bundle["revision"] == REV_A
    assert bundle["metrics"] == {"acc": 0.5, "loss": -1.25}
    assert bundle["tensors"]["lora.0.A"]["shape"] == [3, 5]
    assert bundle["tensors"]["lora.0.A"]["dtype"] == "torch.float32"

    loaded = load_adapter_bundle(path, model_id="M", revision=REV_A)
    assert set(loaded) == set(sd)
    for k, v in sd.items():
        assert torch.equal(loaded[k], v), f"{k} not bit-exact"

    with pytest.raises(NonFiniteMetricError):
        save_adapter_bundle(tmp_path / "b2.pt", sd, model_id="M",
                            revision=REV_A, metrics={"acc": float("nan")})

    tampered = {**bundle, "metrics": {"acc": float("inf")}}
    p2 = tmp_path / "tampered_metric.pt"
    torch.save(tampered, p2)
    with pytest.raises(NonFiniteMetricError):
        load_adapter_bundle(p2, model_id="M", revision=REV_A)

    tensors = dict(bundle["tensors"])
    tensors["clock.weight"] = {
        **tensors["clock.weight"],
        "data": torch.full((5, 5), float("nan")),
    }
    p3 = tmp_path / "tampered_tensor.pt"
    torch.save({**bundle, "tensors": tensors}, p3)
    with pytest.raises(NonFiniteStateError):
        load_adapter_bundle(p3, model_id="M", revision=REV_A)


def test_failed_load_is_atomic():
    rec = _fake_rec()
    torch.manual_seed(9)
    good = {
        "lora.0.A": torch.randn(2, 4) * 0.1,
        "lora.0.B": torch.randn(4, 2) * 0.1,
        "clock.weight": torch.randn(3, 4) * 0.1,
    }
    targets = lambda r: [r.injected[0].lora_A, r.injected[0].lora_B,
                         r.clock.weight]  # noqa: E731

    LocalizedRecurrence.load_adapter_state(rec, {k: v.clone()
                                                 for k, v in good.items()})
    baseline = _snapshot(targets(rec))

    bad_states = [
        ({k: v for k, v in good.items() if k != "clock.weight"},
         AdapterBundleSchemaError),                     # missing key
        ({**good, "extra.key": torch.zeros(1)},
         AdapterBundleSchemaError),                     # unexpected key
        ({**good, "lora.0.A": torch.zeros(5, 4)},
         AdapterBundleSchemaError),                     # wrong shape
        ({**good, "lora.0.B": good["lora.0.B"].to(torch.float64)},
         AdapterBundleSchemaError),                     # wrong dtype
        ({**good, "clock.weight": "not-a-tensor"},
         AdapterBundleSchemaError),                     # not a Tensor
        ({**good, "clock.weight": torch.full((3, 4), float("nan"))},
         NonFiniteStateError),                          # non-finite values
    ]
    for sd, exc in bad_states:
        with pytest.raises(exc):
            LocalizedRecurrence.load_adapter_state(rec, sd)
        assert _unchanged(targets(rec), baseline), \
            f"failed load mutated live state ({exc.__name__})"

    with pytest.raises(AdapterBundleError):
        LocalizedRecurrence.load_adapter_state(rec, "not-a-dict")
    assert _unchanged(targets(rec), baseline)

    corrupt = Path(__import__("tempfile").gettempdir()) / "corrupt_bundle.pt"
    corrupt.write_bytes(b"this is not a checkpoint")
    with pytest.raises(AdapterBundleError):
        load_adapter_bundle(corrupt, model_id="m", revision=REV_A)


def test_load_adapter_state_rolls_back_bit_exact_on_late_copy_failure(
        monkeypatch):
    """A copy that raises on the SECOND staged target — after an earlier
    target was already mutated — must still leave every target bit-exactly
    unchanged. The failing copy implementation stays active THROUGHOUT the
    rollback (which must therefore not route through Tensor.copy_), live
    Parameter identities survive, a later normal load succeeds, and the
    raised AdapterBundleError preserves the original cause."""
    rec = _fake_rec()
    torch.manual_seed(9)
    good = {
        "lora.0.A": torch.randn(2, 4) * 0.1,
        "lora.0.B": torch.randn(4, 2) * 0.1,
        "clock.weight": torch.randn(3, 4) * 0.1,
    }
    targets = lambda r: [r.injected[0].lora_A, r.injected[0].lora_B,
                         r.clock.weight]  # noqa: E731
    target_ids = [id(t) for t in targets(rec)]

    LocalizedRecurrence.load_adapter_state(rec, {k: v.clone()
                                                 for k, v in good.items()})
    baseline = _snapshot(targets(rec))

    incoming = {k: v + 1.234 for k, v in good.items()}  # all differ
    # inject the failure on the SECOND staged target ("lora.0.B") so that
    # "lora.0.A" has already been copied (mutated) when it fires
    late_src = incoming["lora.0.B"]
    real_copy = torch.Tensor.copy_
    mutated_evidence = {}

    def failing_copy(self, src, *a, **kw):
        if src is late_src:
            raise RuntimeError("injected late-copy failure")
        result = real_copy(self, src, *a, **kw)
        mutated_evidence[id(self)] = self.detach().clone()
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", failing_copy)
    with pytest.raises(AdapterBundleError) as excinfo:
        LocalizedRecurrence.load_adapter_state(rec, incoming)
    assert isinstance(excinfo.value.__cause__, RuntimeError), \
        "original cause was not preserved"
    assert "injected late-copy failure" in str(excinfo.value.__cause__)

    # the failure was STILL ACTIVE here — no monkeypatch.undo() yet — so the
    # rollback provably never used the failing copy path; and the recorded
    # post-copy value proves lora.0.A really was mutated before the failure
    id_a = id(rec.injected[0].lora_A)
    assert id_a in mutated_evidence, "no earlier target was copied first"
    assert not bool(torch.equal(mutated_evidence[id_a],
                                baseline[id_a])), \
        "earlier target was not actually mutated before the failure"
    monkeypatch.undo()

    assert _unchanged(targets(rec), baseline), (
        "late copy failure left partially-copied adapter state; "
        "rollback did not restore every target bit-exactly")
    assert [id(t) for t in targets(rec)] == target_ids, \
        "rollback replaced live Parameter objects"

    # and a subsequent NORMAL adapter load succeeds end-to-end
    LocalizedRecurrence.load_adapter_state(rec, {k: v.clone()
                                                 for k, v in good.items()})
    for t, v in zip(targets(rec), good.values()):
        assert bool(torch.equal(t, v)), "normal reload after rollback failed"


def _assert_cache_and_positions_consumed(rec, ids, k):
    """Prove the latent loop genuinely consumes the DynamicCache and the
    positions: storage grows exactly where it must, positions advance, and
    sabotaging either changes the result (test fails if cache arguments
    were ignored)."""
    t = ids.shape[1]
    with torch.no_grad():
        cache_a, z0_a = rec._encode(ids)
        cache_b, z0_b = rec._encode(ids)
        cache_c, _ = rec._encode(ids)
        assert torch.equal(z0_a, z0_b)
        for lyr in cache_c.layers:       # corrupt every stored state; if
            for attr in ("conv_states", "recurrent_states",  # the loop read
                         "keys", "values"):  # the cache, output must move
                v = getattr(lyr, attr, None)
                if torch.is_tensor(v):
                    v.mul_(100.0)
        z_a, pos_a = rec.latent_steps(z0_a, cache_a, t, k)
        z_b, pos_b = rec.latent_steps(z0_b, cache_b, t + 3, k)  # shifted pos
        z_c, pos_c = rec.latent_steps(z0_a, cache_c, t, k)
    assert pos_a == t + k and pos_b == t + 3 + k and pos_c == t + k
    assert not torch.equal(z_a, z_b), "loop ignored position arguments"
    assert not torch.equal(z_a, z_c), "loop ignored cache contents"
    lo, hi = rec.interval
    for i in range(rec.n_layers):        # kv layout: [b, kv_heads, seq, dim]
        ks = getattr(cache_a.layers[i], "keys", None)
        if ks is None:
            continue
        expected = t + (k if lo <= i < hi else 0)
        assert ks.shape[2] == expected, \
            f"layer {i} kv length {ks.shape[2]} != expected {expected}"


# ---------------------------------------------------------------------------
# cached localized/full equivalence across an adapter roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("interval", [(0, 6), (2, 4)])   # full + localized
def test_cached_localized_full_equivalence_across_fp32_roundtrip(
        tmp_path, interval):
    pytest.importorskip("transformers")
    layers = 6

    rec1 = _nonzero_adapted_rec(_tiny_qwen35(11, layers), interval=interval)
    trainables = rec1.trainable_parameters()
    assert all(p.dtype == torch.float32 for p in trainables), "fp32 contract"
    assert all(bool(torch.isfinite(p).all()) for p in trainables)
    assert sum(float(p.abs().sum()) for p in trainables) > 0.0, \
        "adapter must be nonzero for this test"

    _assert_cache_and_positions_consumed(rec1, _IDS, 2)

    order1, scores1 = _cached_matches_full(rec1)
    order1b, scores1b = _cached_matches_full(rec1)
    assert order1 == order1b and torch.equal(torch.tensor(scores1),
                                             torch.tensor(scores1b)), \
        "cached scoring is not deterministic"

    path = tmp_path / f"best_params_{interval[0]}_{interval[1]}.pt"
    rec1.export_adapter_bundle(path, model_id="tiny-qwen35",
                               revision=REV_A, metrics={"val_acc": 0.75})
    assert not (tmp_path / (path.name + ".tmp")).exists(), "tmp file leaked"

    rec2 = LocalizedRecurrence(_tiny_qwen35(11, layers), None,
                               interval=interval, max_k=4, lora_r=4,
                               lora_alpha=8, grad_checkpoint=False)
    with pytest.raises(AdapterBundleIdentityError):
        rec2.load_adapter_bundle(path, model_id="tiny-qwen35",
                                 revision=REV_B)
    rec2.load_adapter_bundle(path, model_id="tiny-qwen35", revision=REV_A)
    assert all(p.dtype == torch.float32
               for p in rec2.trainable_parameters())

    _assert_cache_and_positions_consumed(rec2, _IDS, 2)
    order2, scores2 = _cached_matches_full(rec2)
    assert order1 == order2
    assert torch.equal(torch.tensor(scores1), torch.tensor(scores2)), \
        "scores changed across the bundle roundtrip"


# ---------------------------------------------------------------------------
# fail-closed ordering: falsey revisions + eval-time bundle identity first
# ---------------------------------------------------------------------------

def test_load_model_falsey_revision_rejected_before_any_hf_loader(monkeypatch):
    """Only ``revision=None`` selects the pinned default; falsey values
    ("", False, 0) and mutable refs must reach require_pinned_revision and
    be rejected BEFORE AutoTokenizer/AutoModel are ever contacted."""
    pytest.importorskip("transformers")
    import transformers

    from latent_lab.bench.latent_run import DEFAULT_MODEL_ID, load_model

    calls = []

    def _forbid(kind):
        def _boom(*a, **kw):
            calls.append(kind)
            raise AssertionError(f"{kind} loader was called")
        return _boom

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        _forbid("tokenizer"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        _forbid("model"))

    for bad in ["", "   ", False, 0, "main", "latest", "abcdef0"]:
        with pytest.raises(AdapterBundleIdentityError):
            load_model("cpu", DEFAULT_MODEL_ID, bad)
    assert calls == [], \
        "transformers loaders were reached despite an invalid revision"


def test_cmd_eval_identity_mismatch_aborts_before_load_model(
        tmp_path, monkeypatch):
    """A tampered train_report.json with a VALID 40-hex revision must be
    rejected via the on-disk bundle identity check BEFORE load_model can
    fetch anything."""
    from latent_lab.bench import latent_run

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_adapter_bundle(adapter_dir / "best_params.pt",
                        {"lora.0.A": torch.randn(2, 4) * 0.1},
                        model_id="M", revision=REV_B)

    def write_report(rev):
        (adapter_dir / "train_report.json").write_text(json.dumps({"config": {
            "mode": "E-localized", "model": "M", "revision": rev,
            "interval": [0, 1], "k": 1, "max_k": 2, "lora_r": 2}}))

    def _forbid_load_model(*a, **kw):
        raise AssertionError("load_model ran before bundle identity check")

    monkeypatch.setattr(latent_run, "load_model", _forbid_load_model)
    args = SimpleNamespace(adapter=str(adapter_dir), split="test_id",
                           k=None, ablate=None, seed=0, limit=None,
                           device="cpu", out=None)

    # valid-but-mismatched pinned revision in the report: bundle identity
    # fails first, load_model never runs
    write_report(REV_A)
    with pytest.raises(AdapterBundleIdentityError):
        latent_run.cmd_eval(args)

    # ordering sanity: with MATCHING identities load_model is reached
    # (aborted immediately after by this stub) instead of the bundle error
    write_report(REV_B)

    def _recording_load_model(device, mid, rev):
        reached.append((mid, rev))
        raise RuntimeError("stop-right-after-load-model")

    reached = []
    monkeypatch.setattr(latent_run, "load_model", _recording_load_model)
    with pytest.raises(RuntimeError, match="stop-right-after-load-model"):
        latent_run.cmd_eval(args)
    assert reached == [("M", REV_B)]
