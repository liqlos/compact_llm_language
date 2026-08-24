"""Runtime-integrity guarantees: checkpointing, fail-closed stepping, bundles.

All tests are tiny, deterministic, CPU-only and never touch the network or a
retained checkpoint (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are forced below;
the hybrid model is built locally from a config with a fixed seed).
"""

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
    """Adam that corrupts a parameter after its own update once armed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(float("inf"))


class _PoisonStateAdam(torch.optim.Adam):
    """Adam that corrupts its own state after a real update once armed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.armed = False

    def step(self, closure=None):
        super().step(closure)
        if self.armed:
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
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, (m(x) ** 2).mean(), params, 1.0)
    assert _unchanged(params, param_snap), \
        "post-step poisoned parameter survived the rollback"
    assert _opt_state_unchanged(opt, state_snap), \
        "nested optimizer state differs after rollback"


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
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(poisoned, (m(x) ** 2).mean(), params, 1.0)
    assert _unchanged(params, param_snap)
    assert _opt_state_unchanged(poisoned, state_snap), \
        "nested optimizer state is not byte-exact after rollback"


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
    """A copy that raises AFTER earlier targets were already mutated must
    leave every target bit-exactly unchanged (full rollback), and the
    raised AdapterBundleError must preserve the original cause."""
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

    incoming = {k: v + 1.234 for k, v in good.items()}  # all differ
    # inject the failure on the LAST target ("clock.weight") so that
    # "lora.0.A" and "lora.0.B" have already been copied when it fires
    late_src = incoming["clock.weight"]
    real_copy = torch.Tensor.copy_

    def failing_copy(self, src, *a, **kw):
        if src is late_src:
            raise RuntimeError("injected late-copy failure")
        return real_copy(self, src, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "copy_", failing_copy)
    with pytest.raises(AdapterBundleError) as excinfo:
        LocalizedRecurrence.load_adapter_state(rec, incoming)
    assert isinstance(excinfo.value.__cause__, RuntimeError), \
        "original cause was not preserved"
    assert "injected late-copy failure" in str(excinfo.value.__cause__)
    monkeypatch.undo()  # rollback itself never hits the failing path

    assert _unchanged(targets(rec), baseline), (
        "late copy failure left partially-copied adapter state; "
        "rollback did not restore every target bit-exactly")


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
