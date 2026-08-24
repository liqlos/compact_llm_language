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
    OptimizerStateGraphError,
    EmptyCheckpointError,
    NonFiniteMetricError,
    NonFiniteStateError,
    NonFiniteTrainingStateError,
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
    """Adam that corrupts a parameter AND rewrites its per-group params
    topology after its own update once armed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(float("inf"))
            self.param_groups[0]["params"].reverse()


class _PoisonStateAdam(torch.optim.Adam):
    """Adam that corrupts its own state AND injects a foreign Parameter at
    the front of its group after a real update once armed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
            with torch.no_grad():
                self.state[self.param_groups[0]["params"][0]][
                    "exp_avg"].fill_(float("nan"))
            self.param_groups[0]["params"].insert(
                0, torch.nn.Parameter(torch.zeros(1)))


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
    non-tensor optimizer state, ``param_groups`` (LR + injected metadata +
    per-group params topology), and finally raises — the worst-case
    mid-step failure."""

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
            self.param_groups[0]["params"][0].add_(7.0)
        # hostile params-topology rewrite: reorder + foreign addition
        self.param_groups[0]["params"].reverse()
        self.param_groups[0]["params"].append(
            torch.nn.Parameter(torch.zeros(1)))
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


class _OmittedParamPoisonSGD(torch.optim.SGD):
    """Performs a real update over BOTH owned Parameters, then corrupts
    the caller-OMITTED one (finitely) and raises — coverage must not
    depend on the caller's iterable."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False
        self.calls = 0
        self.omitted_after = None

    def step(self, closure=None):
        super().step(closure)
        self.calls += 1
        if self.armed:
            victim = self.param_groups[0]["params"][1]
            with torch.no_grad():
                victim.add_(5.0)
                self.omitted_after = victim.detach().clone()
            raise RuntimeError("omitted-param adversarial failure")


def test_caller_omitted_owned_parameter_is_fully_covered():
    torch.manual_seed(11)
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    ps = list(m.parameters())
    assert len(ps) >= 2
    opt = _OmittedParamPoisonSGD(ps, lr=0.1)

    def fresh_grads():
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    # clean build-up step: identical supplied/owned sets
    guarded_optimizer_step(opt, fresh_grads().detach(), ps, 1.0)
    assert opt.calls == 1
    omitted = ps[1]

    opt.armed = True
    fresh_grads()
    pre_params = _snapshot(ps)
    pre_grads = {id(p): p.grad.detach().clone() for p in ps}
    pre_state = _deep_opt_state(opt)
    pre_groups = _param_groups_snapshot(opt)
    pre_ids = _param_group_param_ids(opt)

    with pytest.raises(RuntimeError, match="omitted-param"):
        guarded_optimizer_step(opt, torch.tensor(0.5), [ps[0]], 1.0)

    assert opt.calls == 2, "adversarial step never ran; coverage unproven"
    assert opt.omitted_after is not None and \
        not torch.equal(opt.omitted_after, pre_params[id(omitted)]), \
        "the caller-omitted Parameter was never actually mutated mid-step"
    assert _unchanged(ps, pre_params), \
        "supplied Parameter corruption survived the rollback"
    assert bool(torch.equal(omitted.detach(), pre_params[id(omitted)])), \
        "optimizer-owned Parameter omitted by the caller was not rolled back"
    for p in ps:
        assert p.grad is not None and bool(torch.equal(p.grad,
                                                       pre_grads[id(p)])), \
            "pre-clip gradients were not restored bit-exactly"
    assert _opt_state_unchanged(opt, pre_state), \
        "optimizer state differs after rollback"
    assert _param_groups_unchanged(opt, pre_groups), \
        "param-group fields differ after rollback"
    assert _param_group_param_ids(opt) == pre_ids, \
        "params topology not restored by rollback"

    # disarmed retry with the SAME partial iterable still covers everything
    opt.armed = False
    guarded_optimizer_step(opt, fresh_grads().detach(), [ps[0]], 1.0)
    assert not _unchanged(ps, pre_params), \
        "retry after disarm did not update every owned parameter"


class _TopologyPoisonSGD(torch.optim.SGD):
    """Two param groups with different hyperparameters: performs a REAL
    update, then rewrites group AND per-group params topology (cross-group
    move, reorder, hostile addition, hostile removal, full group-list
    reversal), corrupts values/state/metadata, and finally raises."""

    def __init__(self, groups, lr):
        super().__init__(groups, lr=lr)
        self.armed = False
        self.foreign = torch.nn.Parameter(torch.zeros(2))

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        g0, g1 = self.param_groups[0], self.param_groups[1]
        with torch.no_grad():
            for group in self.param_groups:
                group["lr"] = 123.0
                group["injected_meta"] = {"evil": [1]}
            for state in self.state.values():
                if "momentum_buffer" in state:
                    state["momentum_buffer"].add_(11.0)
                    state["poison_flag"] = "corrupted"
            g0["params"][0].add_(7.0)          # value corruption
        moved = g1["params"][0]                # hostile cross-group move...
        del g1["params"][0]                    # ...and per-group removal
        g0["params"].insert(0, moved)          # ...plus in-group reorder
        g0["params"].append(self.foreign)      # hostile param addition
        self.param_groups.reverse()            # hostile group reorder
        raise RuntimeError("topology-adversarial failure")


def test_adversarial_two_group_topology_rollback_then_matches_control():
    def _build(seed):
        torch.manual_seed(seed)
        m = torch.nn.Linear(4, 4)
        return m, list(m.parameters())

    torch.manual_seed(21)
    x = torch.randn(8, 4)
    m_adv, ps_adv = _build(31)
    m_ctl, ps_ctl = _build(31)
    w_a, b_a = ps_adv
    w_c, b_c = ps_ctl

    groups_adv = [{"params": [w_a], "lr": 0.05, "momentum": 0.9},
                  {"params": [b_a], "lr": 0.20, "momentum": 0.5}]
    groups_ctl = [{"params": [w_c], "lr": 0.05, "momentum": 0.9},
                  {"params": [b_c], "lr": 0.20, "momentum": 0.5}]
    opt_adv = _TopologyPoisonSGD(groups_adv, lr=0.05)
    opt_ctl = torch.optim.SGD(groups_ctl, lr=0.05)
    foreign = opt_adv.foreign

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    def ctl_step(drop_bias_grad=False):
        fresh_grads(m_ctl)
        if drop_bias_grad:                  # mirror None-vs-tensor pattern
            with torch.no_grad():
                b_c.grad = None
        torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
        opt_ctl.step()

    # one clean step on both sides -> identical non-empty momentum state
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert bool(torch.equal(w_a.detach(), w_c.detach())) and \
        bool(torch.equal(b_a.detach(), b_c.detach()))
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip(ps_adv, ps_ctl)), \
        "pairs did not start from identical optimizer state"

    # -- armed failure after a real, topology-rewriting update --------------
    opt_adv.armed = True
    fresh_grads(m_adv)
    with torch.no_grad():
        b_a.grad = None                     # None-vs-tensor grad pattern
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for p in ps_adv}
    pre_state = _deep_opt_state(opt_adv)
    pre_groups = _param_groups_snapshot(opt_adv)
    pre_ids = _param_group_param_ids(opt_adv)
    assert pre_ids == [[id(w_a)], [id(b_a)]]

    with pytest.raises(RuntimeError, match="topology-adversarial"):
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)

    # exact original group count/order and mappings
    assert len(opt_adv.param_groups) == 2, "group count not restored"
    assert _param_group_param_ids(opt_adv) == pre_ids, \
        "per-group Parameter identity/order not restored"
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05, 0.20], \
        "group order / hyperparameter mapping not restored"
    assert [g["momentum"] for g in opt_adv.param_groups] == [0.9, 0.5]
    assert all("injected_meta" not in g for g in opt_adv.param_groups), \
        "injected group field survived rollback"
    assert all(id(foreign) not in {id(p) for p in g["params"]}
               for g in opt_adv.param_groups), \
        "hostilely-added foreign Parameter survived rollback"
    # bit-exact parameters, gradients, optimizer state, non-params fields
    assert _unchanged(ps_adv, pre_params), "value corruption survived"
    for p in ps_adv:
        saved = pre_grads[id(p)]
        if saved is None:
            assert p.grad is None, "None gradient became a tensor"
        else:
            assert p.grad is not None and bool(torch.equal(p.grad, saved))
    assert _opt_state_unchanged(opt_adv, pre_state), \
        "optimizer state not restored bit-exactly"
    assert all("poison_flag" not in st for st in opt_adv.state.values()), \
        "injected state field survived rollback"
    assert _param_groups_unchanged(opt_adv, pre_groups), \
        "non-params group fields not restored bit-exactly"

    # -- disarmed retry of the SAME update vs independent control ----------
    opt_adv.armed = False
    fresh_grads(m_adv)
    with torch.no_grad():
        b_a.grad = None                     # control mirrors this pattern
    guarded_optimizer_step(opt_adv, (m_adv(x) ** 2).mean(), ps_adv, 1.0)
    ctl_step(drop_bias_grad=True)

    assert bool(torch.equal(w_a.detach(), w_c.detach())), \
        "post-retry weight trajectory diverges from the clean control"
    assert bool(torch.equal(b_a.detach(), b_c.detach())), \
        "post-retry bias trajectory diverges from the clean control"
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[b]["momentum_buffer"])
        for a, b in zip(ps_adv, ps_ctl)), \
        "optimizer state diverges from the clean control after retry"
    assert _param_group_param_ids(opt_adv) == [[id(w_a)], [id(b_a)]], \
        "retry topology diverges from the clean control mapping"
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05, 0.20]
    assert [g["momentum"] for g in opt_adv.param_groups] == [0.9, 0.5]
    assert all("injected_meta" not in g for g in opt_adv.param_groups)


# ---------------------------------------------------------------------------
# structurally exact optimizer-state rollback (adversarial metadata swaps)
# ---------------------------------------------------------------------------

def _state_leaf_exact(live, snap) -> bool:
    """Bit-and-metadata exactness: every tensor leaf must equal the
    snapshot in values AND shape, dtype, device, layout, strides and
    storage offset; containers must match structurally; scalars must be
    value- and type-identical."""
    if torch.is_tensor(snap):
        return (torch.is_tensor(live)
                and tuple(live.shape) == tuple(snap.shape)
                and live.dtype == snap.dtype
                and live.device == snap.device
                and live.layout == snap.layout
                and tuple(live.stride()) == tuple(snap.stride())
                and live.storage_offset() == snap.storage_offset()
                and bool(torch.equal(live, snap)))
    if isinstance(snap, dict):
        return (isinstance(live, dict)
                and set(live.keys()) == set(snap.keys())
                and all(_state_leaf_exact(live[k], v)
                        for k, v in snap.items()))
    if isinstance(snap, (list, tuple)):
        return (isinstance(live, type(snap)) and len(live) == len(snap)
                and all(_state_leaf_exact(a, b)
                        for a, b in zip(live, snap)))
    return (not torch.is_tensor(live) and type(live) is type(snap)
            and live == snap)


def _opt_state_structurally_exact(opt, snap_tree) -> bool:
    """Whole-optimizer-state exactness against a ``_deep_opt_state`` tree:
    identical key sets and per-entry structural/metadata/value equality
    (hostile injected fields cannot survive)."""
    if set(opt.state.keys()) != set(snap_tree.keys()):
        return False
    return all(_state_leaf_exact(opt.state[p], entry)
               for p, entry in snap_tree.items())


class _ShapeDtypePoisonAdam(torch.optim.Adam):
    """Performs a REAL update, then replaces one momentum tensor with a
    DIFFERENT-SHAPE, DIFFERENT-DTYPE tensor, swaps another to float16,
    turns a scalar state field into junk, corrupts parameters, group
    metadata/topology and gradients, and finally raises a distinctive
    ValueError."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.armed = False
        self.foreign = torch.nn.Parameter(torch.zeros(1))

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        w = self.param_groups[0]["params"][0]
        b = self.param_groups[0]["params"][1]
        self.state[w]["exp_avg"] = torch.zeros(
            1, dtype=torch.float16)               # wrong shape AND dtype
        self.state[b]["exp_avg"] = \
            self.state[b]["exp_avg"].to(torch.float16)  # same shape, fp16
        self.state[b]["step"] = "junk-non-tensor"
        with torch.no_grad():
            w.add_(7.0)
            for group in self.param_groups:
                group["lr"] = 123.0
                group["injected_meta"] = {"evil": [1]}
            if b.grad is not None:
                b.grad.fill_(99.0)                 # hostile gradient rewrite
        g = self.param_groups[0]                   # hostile topology rewrite
        g["params"].reverse()
        g["params"].append(self.foreign)
        raise ValueError("shape-shifting state tensor corruption")


def test_adversarial_shape_dtype_state_swap_rolls_back_exactly():
    torch.manual_seed(5)
    x = torch.randn(8, 4)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    w_a, b_a = ps_adv
    opt_adv = _ShapeDtypePoisonAdam(ps_adv, lr=0.05)
    opt_ctl = torch.optim.Adam(ps_ctl, lr=0.05)
    foreign = opt_adv.foreign

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    def ctl_step():
        fresh_grads(m_ctl)
        torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
        opt_ctl.step()

    # one clean step on both sides -> identical non-empty Adam state
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach()))
    assert all(
        torch.equal(opt_adv.state[a]["exp_avg"], opt_ctl.state[c]["exp_avg"])
        and torch.equal(opt_adv.state[a]["exp_avg_sq"],
                        opt_ctl.state[c]["exp_avg_sq"])
        for a, c in zip(ps_adv, ps_ctl)), \
        "pairs did not start from identical optimizer state"

    # -- armed failure after a real, structure-breaking update --------------
    opt_adv.armed = True
    fresh_grads(m_adv)
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): p.grad.detach().clone() for p in ps_adv}
    pre_state = _deep_opt_state(opt_adv)
    pre_ids = _param_group_param_ids(opt_adv)

    with pytest.raises(ValueError,
                       match="shape-shifting state tensor corruption") as e:
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)
    assert type(e.value) is ValueError, \
        "original step exception was masked by a rollback failure"

    # exact state: structure + metadata + values (swapped shape/dtype/junk
    # leaf must ALL be gone; the pristine snapshot must be back verbatim)
    assert _opt_state_structurally_exact(opt_adv, pre_state), \
        "optimizer state was not restored exactly after metadata swap"
    assert _unchanged(ps_adv, pre_params), \
        "parameter corruption survived the rollback"
    for p in ps_adv:
        assert p.grad is not None and \
            bool(torch.equal(p.grad, pre_grads[id(p)])), \
            "pre-step gradients were not restored bit-exactly"
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05], \
        "mutated LR was not rolled back"
    assert all("injected_meta" not in g for g in opt_adv.param_groups), \
        "injected group field survived the rollback"
    assert _param_group_param_ids(opt_adv) == pre_ids, \
        "per-group Parameter identity/order not restored"
    assert all(id(foreign) != id(p)
               for g in opt_adv.param_groups for p in g["params"]), \
        "hostilely-added foreign Parameter survived the rollback"

    # -- disarmed retry of the SAME update vs independent control -----------
    opt_adv.armed = False
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach())) and \
        bool(torch.equal(b_a.detach(), ps_ctl[1].detach())), \
        "post-retry trajectory diverges from the clean control"
    assert all(
        torch.equal(opt_adv.state[a]["exp_avg"], opt_ctl.state[c]["exp_avg"])
        and torch.equal(opt_adv.state[a]["exp_avg_sq"],
                        opt_ctl.state[c]["exp_avg_sq"])
        for a, c in zip(ps_adv, ps_ctl)), \
        "optimizer state diverges from the clean control after retry"


class _DtypeOnlySwapSGD(torch.optim.SGD):
    """Performs a REAL momentum update, then replaces a same-shape
    float32 ``momentum_buffer`` with a float16 tensor of DIFFERENT VALUES
    and raises. Restoration through ``Tensor.copy_`` would silently keep
    the corrupt live dtype (and cast-rounded values)."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        st = self.state[self.param_groups[0]["params"][0]]
        assert st["momentum_buffer"].dtype == torch.float32
        st["momentum_buffer"] = (st["momentum_buffer"].to(
            torch.float16) * 1.5)                  # same shape, fp16, moved
        raise ValueError("dtype-only state swap")


def test_adversarial_same_shape_dtype_downgrade_restores_exactly():
    torch.manual_seed(6)
    x = torch.randn(8, 4)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    w_a, b_a = ps_adv
    opt_adv = _DtypeOnlySwapSGD(ps_adv, lr=0.05)
    opt_ctl = torch.optim.SGD(ps_ctl, lr=0.05, momentum=0.9)

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    def ctl_step():
        fresh_grads(m_ctl)
        torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
        opt_ctl.step()

    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    buf = opt_adv.state[w_a]["momentum_buffer"]
    assert buf.dtype == torch.float32 and buf.shape == (4, 4)

    opt_adv.armed = True
    fresh_grads(m_adv)
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): p.grad.detach().clone() for p in ps_adv}
    pre_state = _deep_opt_state(opt_adv)

    with pytest.raises(ValueError, match="dtype-only state swap") as e:
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)
    assert type(e.value) is ValueError, \
        "original step exception was masked by a rollback failure"

    assert _opt_state_structurally_exact(opt_adv, pre_state), \
        "float16-swapped momentum buffer was not restored to exact fp32"
    assert opt_adv.state[w_a]["momentum_buffer"].dtype == torch.float32, \
        "rollback preserved the corrupt live dtype"
    assert _unchanged(ps_adv, pre_params)
    for p in ps_adv:
        assert p.grad is not None and \
            bool(torch.equal(p.grad, pre_grads[id(p)]))
    assert _param_group_param_ids(opt_adv) == [[id(w_a), id(b_a)]], \
        "per-group Parameter identity/order not restored"

    opt_adv.armed = False
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach())) and \
        bool(torch.equal(b_a.detach(), ps_ctl[1].detach())), \
        "post-retry trajectory diverges from the clean control"
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[c]["momentum_buffer"])
        for a, c in zip(ps_adv, ps_ctl)), \
        "optimizer state diverges from the clean control after retry"


class _OuterListRebindSGD(torch.optim.SGD):
    """Performs a REAL update while mutating the ORIGINAL outer list's
    groups, then REBINDS ``self.param_groups`` to a fresh hostile outer
    list (the original list object survives only via this cache), and
    raises. Refuses any retry unless the ACTIVE outer list IS the
    originally cached one."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False
        self.original_outer = self.param_groups      # cache outer identity
        self.foreign = torch.nn.Parameter(torch.zeros(1))

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        with torch.no_grad():
            next(iter(self.state.values()))["momentum_buffer"].add_(5.0)
            self.param_groups[0]["params"][0].add_(3.0)
            for g in self.param_groups:
                g["lr"] = 42.0
                g["injected_meta"] = True
        self.param_groups = [                        # hostile OUTER rebinding
            {"params": [self.foreign], "lr": 9.9, "hostile": True}]
        raise ValueError("outer param_groups list rebinding")

    def refuse_if_outer_list_lost(self):
        if self.param_groups is not self.original_outer:
            raise AssertionError(
                "active param_groups is NOT the original outer list object")


def test_adversarial_outer_param_groups_list_identity_restored():
    torch.manual_seed(7)
    x = torch.randn(8, 4)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    w_a, b_a = ps_adv
    opt_adv = _OuterListRebindSGD(ps_adv, lr=0.05)
    opt_ctl = torch.optim.SGD(ps_ctl, lr=0.05, momentum=0.9)
    foreign = opt_adv.foreign
    original_outer = opt_adv.original_outer
    assert opt_adv.param_groups is original_outer

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    def ctl_step():
        fresh_grads(m_ctl)
        torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
        opt_ctl.step()

    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert opt_adv.param_groups is original_outer, \
        "clean step replaced the outer param_groups list"

    opt_adv.armed = True
    fresh_grads(m_adv)
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): p.grad.detach().clone() for p in ps_adv}
    pre_state = _deep_opt_state(opt_adv)
    pre_groups = _param_groups_snapshot(opt_adv)
    pre_ids = _param_group_param_ids(opt_adv)

    with pytest.raises(ValueError,
                       match="outer param_groups list rebinding") as e:
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)
    assert type(e.value) is ValueError, \
        "original step exception was masked by a rollback failure"

    # the ORIGINAL outer list object must be back — not an equal copy
    assert opt_adv.param_groups is original_outer, \
        "rollback did not restore the original outer param_groups identity"
    opt_adv.refuse_if_outer_list_lost()      # refuses retry on lost identity
    assert len(opt_adv.param_groups) == 1, "hostile group count leaked"
    assert _opt_state_structurally_exact(opt_adv, pre_state), \
        "optimizer state not restored exactly across outer-list rebind"
    assert _unchanged(ps_adv, pre_params)
    for p in ps_adv:
        assert p.grad is not None and \
            bool(torch.equal(p.grad, pre_grads[id(p)])), \
            "pre-step gradients were not restored bit-exactly"
    assert _param_groups_unchanged(opt_adv, pre_groups), \
        "group fields differ after outer-list rollback"
    assert [g["lr"] for g in opt_adv.param_groups] == [0.05], \
        "mutated LR was not rolled back"
    assert all("injected_meta" not in g and "hostile" not in g
               for g in opt_adv.param_groups), \
        "injected group field survived the rollback"
    assert _param_group_param_ids(opt_adv) == pre_ids, \
        "per-group Parameter identity/order not restored"
    assert id(foreign) not in {id(p) for p in
                               opt_adv.param_groups[0]["params"]}, \
        "hostilely-added foreign Parameter survived the rollback"

    # -- disarmed retry of the SAME update vs independent control -----------
    opt_adv.armed = False
    opt_adv.refuse_if_outer_list_lost()   # still refuses before retrying
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    opt_adv.refuse_if_outer_list_lost()   # ... and after it
    ctl_step()
    assert opt_adv.param_groups is original_outer
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach())) and \
        bool(torch.equal(b_a.detach(), ps_ctl[1].detach())), \
        "post-retry trajectory diverges from the clean control"
    assert all(
        torch.equal(opt_adv.state[a]["momentum_buffer"],
                    opt_ctl.state[c]["momentum_buffer"])
        for a, c in zip(ps_adv, ps_ctl)), \
        "optimizer state diverges from the clean control after retry"


# ---------------------------------------------------------------------------
# storage-graph-exact optimizer-state rollback (adversarial aliases/views)
# ---------------------------------------------------------------------------

def _full_extent_view(t) -> torch.Tensor:
    """A 1-D tensor spanning a tensor's ENTIRE backing storage."""
    st = t.untyped_storage()
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(
        st, 0, (st.nbytes() // t.element_size(),), (1,))


def _tensor_meta(t) -> tuple:
    return (tuple(t.shape), tuple(t.stride()), t.storage_offset(),
            t.dtype, t.device, t.layout)


class _AliasPoisonSGD(torch.optim.SGD):
    """Performs a REAL momentum update, then corrupts state THROUGH an
    offset/strided view, scribbles the WHOLE backing storage outside the
    visible region, swaps dtype/stride metadata, severs alias topology
    (shared storages + repeated references), poisons parameters, group
    topology/metadata and gradients, and finally raises."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False
        self.foreign = torch.nn.Parameter(torch.zeros(1))

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        w = self.param_groups[0]["params"][0]
        st = self.state[w]
        with torch.no_grad():
            st["momentum_buffer"].add_(11.0)
            st["tail"].mul_(3.0)                 # corrupt via the VIEW
            _full_extent_view(st["tail"]).fill_(-7.0)   # whole storage
            st["col"].mul_(2.0)                  # non-dense strided view
            st["big_row"] = torch.full((4,), 5.0,
                                       dtype=torch.float16)   # dtype swap
            st["expanded"] = torch.full((3, 4), 6.0)     # stride swap
            st["scalar"] = "junk-non-tensor"
            st["poison_flag"] = "corrupted"
            st["cross"].add_(100.0)      # hits mirror's shared storage
            w.add_(7.0)
            for group in self.param_groups:
                group["lr"] = 123.0
                group["injected_meta"] = {"evil": [1]}
            for p in self.param_groups[0]["params"]:
                if p.grad is not None:
                    p.grad.fill_(99.0)
        g0 = self.param_groups[0]
        g0["params"].reverse()
        g0["params"].append(self.foreign)
        raise RuntimeError("alias-adversarial failure")


def test_adversarial_alias_view_storage_graph_rolls_back_exactly():
    torch.manual_seed(8)
    x = torch.randn(8, 4)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    w_a, b_a = ps_adv
    opt_adv = _AliasPoisonSGD(ps_adv, lr=0.05)
    opt_ctl = torch.optim.SGD(ps_ctl, lr=0.05, momentum=0.9)

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    # one clean step on both sides -> identical non-empty momentum state
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
    opt_ctl.step()
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach()))
    assert torch.equal(opt_adv.state[w_a]["momentum_buffer"],
                       opt_ctl.state[ps_ctl[0]]["momentum_buffer"]), \
        "pairs did not start from identical optimizer state"

    # enrich the candidate's state with an exact-storage challenge set:
    # an offset view over a LARGER hidden storage (base absent from state),
    # a non-dense strided column view sharing with its base, an expanded
    # zero-stride view, a repeated SAME-object reference, two independent
    # empty tensors, a scalar, and a cross-entry alias under a nested dict.
    tail = torch.arange(12, dtype=torch.float32) * 1.5
    tail_view = tail[8:12]                   # offset 8, storage 12 elems
    mat = torch.arange(16, dtype=torch.float32).view(4, 4)
    col = mat[:, 1]                          # shape (4,), stride (4,), off 1
    row = torch.linspace(-1.0, 1.0, 4)
    expanded = row.expand(3, 4)              # stride (0, 1), shares w/ row
    rep = torch.arange(3, dtype=torch.float32)
    cross_base = torch.linspace(0.0, 1.0, 5)
    st_w = opt_adv.state[w_a]
    st_w.update({
        "tail": tail_view, "mat": mat, "col": col,
        "big_row": row, "expanded": expanded,
        "rep_a": rep, "rep_b": rep,          # same object twice
        "empty_a": torch.empty(0), "empty_b": torch.empty(0),
        "scalar": torch.tensor(2.5), "cross": cross_base,
    })
    opt_adv.state[b_a]["nest"] = {"deep": cross_base[2:]}   # alias view
    assert st_w["rep_a"] is st_w["rep_b"]
    assert st_w["expanded"].untyped_storage().data_ptr() == \
        st_w["big_row"].untyped_storage().data_ptr()
    assert st_w["cross"].untyped_storage().data_ptr() == \
        opt_adv.state[b_a]["nest"]["deep"].untyped_storage().data_ptr()

    pristine_full_tail = _full_extent_view(tail_view).clone()
    pristine_momentum = st_w["momentum_buffer"].detach().clone()
    pristine_cross = cross_base.clone()
    meta_tail = _tensor_meta(tail_view)
    meta_col = _tensor_meta(col)

    # -- armed failure after a real, graph-corrupting update ---------------
    opt_adv.armed = True
    fresh_grads(m_adv)
    pre_params = _snapshot(ps_adv)
    pre_grads = {id(p): p.grad.detach().clone() for p in ps_adv}
    pre_groups = _param_groups_snapshot(opt_adv)
    pre_ids = _param_group_param_ids(opt_adv)

    with pytest.raises(RuntimeError, match="alias-adversarial") as e:
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)
    assert type(e.value) is RuntimeError, \
        "original step exception was masked by a rollback failure"

    # the offset view: SAME object, exact geometry, FULL storage extent
    st_w = opt_adv.state[w_a]
    assert st_w["tail"] is tail_view, \
        "offset view was replaced by a normalized dense clone"
    assert _tensor_meta(st_w["tail"]) == meta_tail, \
        "view metadata (shape/stride/offset/dtype/device/layout) not exact"
    assert st_w["tail"].untyped_storage().nbytes() == 12 * 4, \
        "backing storage was shrunk to just the visible region"
    assert torch.equal(_full_extent_view(st_w["tail"]), pristine_full_tail), \
        "storage bytes OUTSIDE the offset view survived corruption"
    # non-dense strided view: stride/offset restored AND still sharing
    assert _tensor_meta(st_w["col"]) == meta_col
    assert st_w["col"].untyped_storage().data_ptr() == \
        st_w["mat"].untyped_storage().data_ptr()
    assert torch.equal(st_w["mat"], mat) and \
        torch.equal(st_w["col"], mat[:, 1])
    # zero-stride expanded view: stride AND shared storage restored
    assert st_w["expanded"].stride() == (0, 1)
    assert st_w["expanded"].untyped_storage().data_ptr() == \
        st_w["big_row"].untyped_storage().data_ptr()
    assert torch.equal(st_w["expanded"], row.expand(3, 4))
    # repeated reference to ONE object: reinstated as one object again
    assert st_w["rep_a"] is st_w["rep_b"], \
        "repeated reference became two independent tensors"
    assert torch.equal(st_w["rep_a"], rep)
    # independent empty tensors must NOT be grouped into one storage
    assert st_w["empty_a"] is not st_w["empty_b"]
    assert st_w["empty_a"].numel() == 0 and st_w["empty_b"].numel() == 0
    assert st_w["empty_a"].untyped_storage()._cdata != \
        st_w["empty_b"].untyped_storage()._cdata
    # scalar leaf back as a tensor value; injected field gone
    assert torch.is_tensor(st_w["scalar"]) and float(st_w["scalar"]) == 2.5
    assert "poison_flag" not in st_w
    assert st_w["big_row"].dtype == torch.float32
    # cross-entry alias across a nested dict: sharing + geometry intact
    cw = st_w["cross"]
    cm = opt_adv.state[b_a]["nest"]["deep"]
    assert cm.storage_offset() == 2
    assert cw.untyped_storage().data_ptr() == \
        cm.untyped_storage().data_ptr(), \
        "cross-entry shared storage was severed by the rollback"
    assert torch.equal(cw, pristine_cross)
    assert torch.equal(cm, pristine_cross[2:])
    # real optimizer state bit-exact
    assert torch.equal(st_w["momentum_buffer"], pristine_momentum)
    # params / grads / groups topology, as in every other transaction
    assert _unchanged(ps_adv, pre_params), "param corruption survived"
    for p in ps_adv:
        assert p.grad is not None and \
            bool(torch.equal(p.grad, pre_grads[id(p)])), \
            "pre-clip gradients were not restored bit-exactly"
    assert _param_groups_unchanged(opt_adv, pre_groups), \
        "group fields differ after rollback"
    assert all("injected_meta" not in g for g in opt_adv.param_groups)
    assert any(g["lr"] == 0.05 for g in opt_adv.param_groups)
    assert _param_group_param_ids(opt_adv) == pre_ids, \
        "per-group Parameter identity/order not restored"
    assert all(id(opt_adv.foreign) != id(p)
               for g in opt_adv.param_groups for p in g["params"]), \
        "hostilely-added foreign Parameter survived the rollback"

    # -- disarmed retry of the SAME update vs independent control ----------
    opt_adv.armed = False
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    fresh_grads(m_ctl)
    torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
    opt_ctl.step()
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach())) and \
        bool(torch.equal(b_a.detach(), ps_ctl[1].detach())), \
        "post-retry trajectory diverges from the clean control"
    assert torch.equal(opt_adv.state[w_a]["momentum_buffer"],
                       opt_ctl.state[ps_ctl[0]]["momentum_buffer"]), \
        "optimizer state diverges from the clean control after retry"
    assert _tensor_meta(opt_adv.state[w_a]["tail"]) == meta_tail, \
        "retry did not preserve the restored view geometry"


class _StateMappingRebindSGD(torch.optim.SGD):
    """Performs a real update, corrupts state values, then REBINDS
    ``self.state`` to a foreign plain dict and raises."""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr, momentum=0.9)
        self.armed = False
        self.original_state = self.state

    def step(self, closure=None):
        super().step(closure)
        if not self.armed:
            return
        w = self.param_groups[0]["params"][0]
        with torch.no_grad():
            next(iter(self.state.values()))["momentum_buffer"].add_(9.0)
            w.add_(4.0)
        self.state = {"hostile": {"evil": True}}
        raise ValueError("optimizer.state mapping rebind")


def test_hostile_state_mapping_rebind_restores_original_object():
    torch.manual_seed(14)
    x = torch.randn(8, 4)
    m_adv = torch.nn.Linear(4, 4)
    m_ctl = torch.nn.Linear(4, 4)
    m_ctl.load_state_dict(m_adv.state_dict())
    ps_adv = list(m_adv.parameters())
    ps_ctl = list(m_ctl.parameters())
    w_a, b_a = ps_adv
    opt_adv = _StateMappingRebindSGD(ps_adv, lr=0.05)
    opt_ctl = torch.optim.SGD(ps_ctl, lr=0.05, momentum=0.9)
    original_state = opt_adv.original_state

    def fresh_grads(m):
        m.zero_grad(set_to_none=True)
        loss = (m(x) ** 2).mean()
        loss.backward()
        return loss

    def ctl_step():
        fresh_grads(m_ctl)
        torch.nn.utils.clip_grad_norm_(ps_ctl, 1.0)
        opt_ctl.step()

    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert opt_adv.state is original_state, "clean step replaced the mapping"

    opt_adv.armed = True
    fresh_grads(m_adv)
    pre_params = _snapshot(ps_adv)
    pre_state = _deep_opt_state(opt_adv)

    with pytest.raises(ValueError,
                       match="optimizer.state mapping rebind") as e:
        guarded_optimizer_step(opt_adv, torch.tensor(0.5), ps_adv, 1.0)
    assert type(e.value) is ValueError, \
        "original step exception was masked by a rollback failure"

    assert opt_adv.state is original_state, \
        "rollback did not reinstate the original optimizer.state mapping"
    assert _opt_state_structurally_exact(opt_adv, pre_state), \
        "state contents differ after mapping-object restoration"
    assert _unchanged(ps_adv, pre_params)

    opt_adv.armed = False
    guarded_optimizer_step(opt_adv, fresh_grads(m_adv).detach(), ps_adv, 1.0)
    ctl_step()
    assert opt_adv.state is original_state
    assert bool(torch.equal(w_a.detach(), ps_ctl[0].detach())) and \
        bool(torch.equal(b_a.detach(), ps_ctl[1].detach())), \
        "post-retry trajectory diverges from the clean control"
    assert torch.equal(opt_adv.state[w_a]["momentum_buffer"],
                       opt_ctl.state[ps_ctl[0]]["momentum_buffer"])


class _StepCounterSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def test_exotic_state_graph_fails_closed_before_clip_and_step():
    """A state/storage graph that cannot be captured exactly (one backing
    storage viewed at two dtypes; a sparse leaf) must abort the whole
    transaction during preflight — BEFORE clipping or optimizer.step can
    mutate anything — and the optimizer must stay usable afterwards."""
    torch.manual_seed(12)
    m = torch.nn.Linear(4, 4)
    x = torch.randn(6, 4)
    ps = list(m.parameters())
    opt = _StepCounterSGD(ps, lr=0.1)

    (m(x) ** 2).mean().backward()
    guarded_optimizer_step(opt, (m(x) ** 2).mean(), ps, 1.0)
    assert opt.steps == 1

    f = torch.arange(4, dtype=torch.float32)
    opt.state[ps[0]]["f32"] = f
    opt.state[ps[0]]["i32_alias"] = f.view(torch.int32)   # cross-dtype share

    params_snap = _snapshot(ps)
    grads_snap = [p.grad.detach().clone() for p in ps]
    lr_before = opt.param_groups[0]["lr"]
    state_snap = _deep_opt_state(opt)

    with pytest.raises(OptimizerStateGraphError):
        guarded_optimizer_step(opt, torch.tensor(0.5), ps, 1.0)
    assert opt.steps == 1, "optimizer.step ran despite uncapturable state"
    assert _unchanged(ps, params_snap), "parameters moved before rejection"
    for p, g0 in zip(ps, grads_snap):     # clipping must not have run either
        assert p.grad is not None and bool(torch.equal(p.grad, g0)), \
            "gradients were clipped before the preflight rejection"
    assert opt.param_groups[0]["lr"] == lr_before
    assert _opt_state_unchanged(opt, state_snap)

    del opt.state[ps[0]]["i32_alias"]     # fail-closed stays recoverable
    with torch.sparse.check_sparse_tensor_invariants(False):
        sp = torch.sparse_coo_tensor(torch.tensor([[0, 1]]).long(),
                                     torch.tensor([1.0, 2.0]), (4,))
    opt.state[ps[1]]["sparse_leaf"] = sp
    with pytest.raises(OptimizerStateGraphError):
        guarded_optimizer_step(opt, torch.tensor(0.5), ps, 1.0)
    assert opt.steps == 1
    del opt.state[ps[1]]["sparse_leaf"]

    guarded_optimizer_step(opt, (m(x) ** 2).mean(), ps, 1.0)
    assert opt.steps == 2
    assert not _unchanged(ps, params_snap), "recovery step did not run"


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
