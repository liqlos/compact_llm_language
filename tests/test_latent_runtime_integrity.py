"""Runtime-integrity guarantees: fail-stop stepping, identity-bound bundles.

All tests are tiny, deterministic, CPU-only and never touch the network or
a retained checkpoint (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are forced below;
the hybrid model is built locally from a config with a fixed seed).

Design contract under test:
  * guarded_optimizer_step is CHEAP FAIL-STOP: pre-step rejection performs
    no step; any clip/step/post-check fault raises FatalRunInvalidError and
    the caller terminates. There is deliberately NO per-step optimizer /
    parameter / gradient snapshot or rollback (the rejected ~3x-overhead
    dead end); regressions prove the hot path snapshots nothing and that a
    fatal step cannot emit success artifacts.
  * Adapter bundles are identity-bound v2 (model_id + pinned 40-hex
    revision + exact recipe + sha256 content digest over tensor bytes);
    loads fully prevalidate before any mutation.
  * Recovery is exclusively from the last atomically committed bundle,
    loaded into a freshly constructed runtime.
"""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

torch = pytest.importorskip("torch")

from latent_lab.backends.localized import (
    LoRALinear,
    LocalizedRecurrence,
    make_step_clock,
    parse_clock_mode,
    validate_ablation,
)
from latent_lab.bench.latent_run import parse_ablation_cli
from latent_lab.train.checkpointing import (
    AdapterBundleError,
    AdapterBundleIdentityError,
    AdapterBundleSchemaError,
    BestCheckpointTracker,
    EmptyCheckpointError,
    FatalRunInvalidError,
    NonFiniteMetricError,
    NonFiniteStateError,
    NonFiniteTrainingStateError,
    OptimizerStateInspectionError,
    assert_all_finite,
    atomic_write_json,
    guarded_optimizer_step,
    load_adapter_bundle,
    owned_parameters,
    read_run_status,
    require_complete_run,
    require_pinned_revision,
    save_adapter_bundle,
    validate_optimizer_state_standard_and_finite,
    write_run_status,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

REV_A = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
REV_B = "0f9e8d7c6b5a49382715040302010ffedcba9876"
SUITE_SHA = "ab" * 32

# Full training-semantic config; every field below is canonically bound
# into the recipe identity (K/LR/steps/seed/optimizer changes all matter).
TRAIN_CFG = {
    "mode": "E-localized",
    "interval": [1, 4],
    "k": 4, "max_k": 4,
    "lora_r": 8, "lora_alpha": 16.0,
    "lr": 2e-4, "steps": 600, "seed": 0,
    "optimizer": "adamw", "weight_decay": 0.01,
    "lr_schedule": "constant", "warmup": 30, "clip": 0.5,
    "detach_z0": False,
}


def _cfg(**over) -> dict:
    return {**TRAIN_CFG, "suite_sha256": SUITE_SHA, **over}


def _recipe(**over) -> dict:
    from latent_lab.train.checkpointing import recipe_from_config
    return recipe_from_config(_cfg(**over), SUITE_SHA)


RECIPE = _recipe()


def _state(value: float, key: str = "w") -> dict:
    return {key: torch.full((3, 3), value)}


def _snapshot(params) -> dict:
    return {id(p): p.detach().clone() for p in params}


def _unchanged(params, snap) -> bool:
    """NaN-aware bitwise comparison (NaN == NaN counts as unchanged)."""
    for p in params:
        cur, ref = p.detach(), snap[id(p)]
        if cur.dtype.is_floating_point or cur.dtype.is_complex:
            same = (cur == ref) | (cur.isnan() & ref.isnan())
            if not bool(same.all()):
                return False
        elif not bool(torch.equal(cur, ref)):
            return False
    return True


def _fresh_grads(model, x):
    model.zero_grad(set_to_none=True)
    ((model(x) ** 2).mean()).backward()


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
    interval = interval or (0, model.config.num_hidden_layers)
    rec = LocalizedRecurrence(
        model, None, interval=interval,
        max_k=4, lora_r=4, lora_alpha=8)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for l in rec.injected:
            l.lora_B.copy_(
                torch.randn(l.lora_B.shape, generator=g) * 0.05)
    return rec


_IDS = torch.tensor([[5, 9, 13, 7, 200, 3, 42]])
_ANS = [100, 101]
_CANDS = [[100, 101], [102]]


def _cached_matches_full(rec, k=2) -> tuple:
    """Cached candidate ranking must match a full recomputed pass exactly."""
    order, scores, _rep = rec.rank_candidates(_IDS, _CANDS, k)
    with torch.no_grad():
        ce = rec.loss_on_example(_IDS, torch.tensor([_ANS]), k)
    full_sum_logp = -float(ce) * len(_ANS)
    cached_gold = scores[_CANDS.index(_ANS)]
    assert abs(cached_gold - full_sum_logp) < 1e-5, (
        f"cached {cached_gold} != full {full_sum_logp}")
    return order, scores


# ---------------------------------------------------------------------------
# best-checkpoint tracking (finite metric only; strict step schema)
# ---------------------------------------------------------------------------

def test_best_checkpoint_selection_reload_before_report_and_save(tmp_path):
    tr = BestCheckpointTracker()
    s1, s2, s3 = _state(1.0), _state(2.0), _state(3.0)
    assert tr.update(0.5, s1, step=1) is True
    assert tr.update(0.9, s2, step=2) is True
    assert tr.update(0.7, s3, step=3) is False          # worse: ignored
    assert tr.update(0.9, s3, step=4) is False          # tie: keep earliest
    assert tr.best_score == 0.9 and tr.best_step == 2

    s2["w"].fill_(999.0)                                # live pollution
    applied = []
    tr.apply_best(applied.append)                       # reload selected best
    assert len(applied) == 1
    assert bool(torch.equal(applied[0]["w"], torch.full((3, 3), 2.0)))

    path = tmp_path / "best_params.pt"
    tr.save(path, model_id="m", revision=REV_A, recipe=RECIPE)
    loaded = load_adapter_bundle(path, model_id="m", revision=REV_A,
                                 recipe=RECIPE)
    assert bool(torch.equal(loaded["w"], torch.full((3, 3), 2.0)))

    empty = BestCheckpointTracker()
    with pytest.raises(EmptyCheckpointError):
        empty.best_state()
    with pytest.raises(EmptyCheckpointError):
        empty.apply_best(lambda st: st)
    with pytest.raises(EmptyCheckpointError):
        empty.save(tmp_path / "never.pt", model_id="m", revision=REV_A,
                   recipe=RECIPE)
    assert not (tmp_path / "never.pt").exists(), "empty tracker wrote a file"


def test_nan_metric_and_bad_step_schema_rejected():
    tr = BestCheckpointTracker()
    with pytest.raises(NonFiniteMetricError):
        tr.update(float("nan"), _state(1.0), step=1)
    with pytest.raises(NonFiniteMetricError):
        tr.update(float("inf"), _state(1.0), step=1)
    with pytest.raises(NonFiniteMetricError):
        tr.update("not-a-number", _state(1.0), step=1)
    assert not tr.has_best()

    # strict deterministic step schema
    with pytest.raises(NonFiniteMetricError):
        tr.update(0.5, _state(1.0), step=-1)
    with pytest.raises(NonFiniteMetricError):
        tr.update(0.5, _state(1.0), step=1.5)
    with pytest.raises(NonFiniteMetricError):
        tr.update(0.5, _state(1.0), step=True)

    assert tr.update(0.25, _state(1.0), step=1) is True
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {"w": torch.tensor([0.1, float("nan")])}, step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {"w": torch.tensor([float("inf")])}, step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {"w": "not-a-tensor"}, step=2)
    with pytest.raises(NonFiniteStateError):
        tr.update(0.5, {}, step=2)
    assert tr.has_best() and tr.best_score == 0.25 and tr.best_step == 1
    assert bool(torch.equal(tr.best_state()["w"], torch.ones(3, 3)))


# ---------------------------------------------------------------------------
# pinned immutable revisions
# ---------------------------------------------------------------------------

def test_pinned_revision_acceptance_normalization_and_rejection(tmp_path):
    assert require_pinned_revision(f" {REV_A.upper()} ") == REV_A
    for bad in ("main", "latest", "v1.0", "", "   ", REV_A[:39],
                REV_A + "0", "15852E8C16360A2FEA060D615A32B45270F8A8F", None,
                0, False, ["x"]):
        with pytest.raises(AdapterBundleIdentityError):
            require_pinned_revision(bad)

    sd = _state(1.0)
    path = tmp_path / "b.pt"
    with pytest.raises(AdapterBundleIdentityError):
        save_adapter_bundle(path, sd, model_id="m", revision="main",
                            recipe=RECIPE)
    assert not path.exists(), "mutable-revision bundle was written"


def test_load_model_falsey_revision_rejected_before_any_hf_loader(monkeypatch):
    from latent_lab.bench import latent_run

    calls = []

    def _boom(*a, **kw):  # any HF loader contact is a contract violation
        calls.append((a, kw))
        return SimpleNamespace(
            eval=lambda: SimpleNamespace(to=lambda dev: None))

    fake_tok = SimpleNamespace(from_pretrained=_boom)
    fake_model = SimpleNamespace(from_pretrained=_boom)

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", fake_tok)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", fake_model)
    import latent_lab.backends.gdn_patch as gdn
    monkeypatch.setattr(gdn, "install", lambda: None)

    for bad in ("", False, 0, "main"):
        with pytest.raises(AdapterBundleIdentityError):
            latent_run.load_model("cpu", "some-model", bad)
    assert calls == [], "loader contacted before revision validation"

    # explicit default: only revision=None selects DEFAULT_REVISION; here it
    # passes validation and proceeds to the (stubbed) loader
    latent_run.load_model("cpu", None, None)
    assert len(calls) == 2  # tokenizer + model, both with the pinned default


# ---------------------------------------------------------------------------
# fail-stop optimizer stepping
# ---------------------------------------------------------------------------

def test_prepoisoned_parameter_rejected_without_mutation():
    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    with torch.no_grad():
        m.weight.add_(float("nan"))
    snap = _snapshot(m.parameters())  # AFTER poisoning: step must not move it
    _fresh_grads(m, x)
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, (m(x) ** 2).mean(),
                               list(m.parameters()), 1.0)
    assert _unchanged(m.parameters(), snap), "stepped on non-finite parameter"


@pytest.mark.parametrize("bad_clip", [float("nan"), float("inf"), 0.0, -1.0])
def test_clip_config_rejected_without_step_or_mutation(bad_clip):
    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    _fresh_grads(m, x)
    snap = _snapshot(m.parameters())
    grad_snap = {id(p): p.grad.clone() for p in m.parameters()}
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, (m(x) ** 2).mean(),
                               list(m.parameters()), bad_clip)
    assert _unchanged(m.parameters(), snap)
    assert all(bool(torch.equal(p.grad, grad_snap[id(p)]))
               for p in m.parameters()), "rejection touched gradients"


def test_caller_omitted_owned_parameter_is_fully_covered():
    """The transaction set is the union of optimizer groups and caller
    params: an omitted-but-owned Parameter's non-finite gradient must be
    caught BEFORE any step."""
    torch.manual_seed(4)
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    snap = _snapshot(m.parameters())
    _fresh_grads(m, x)
    with torch.no_grad():
        m.bias.grad.fill_(float("nan"))     # bias omitted from caller list
    loss = (m(x) ** 2).mean()
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, loss, [m.weight], 1.0)
    assert _unchanged(m.parameters(), snap)
    owned = owned_parameters(opt, [m.weight])
    assert {id(p) for p in owned} == {id(p) for p in m.parameters()}


def test_poststep_poisoned_parameter_is_fatal_not_rolled_back():
    """Fail-stop contract: after optimizer.step poisons a parameter, the
    run gets ONE fatal exception; state is NOT restored (no snapshot
    mechanism exists) and the caller must terminate."""

    class _PoisonParamSGD(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(float("inf"))

    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    poisoned = _PoisonParamSGD(m.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(6, 4)
    _fresh_grads(m, x)
    exp_avg_before = None
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(poisoned, (m(x) ** 2).mean(),
                               list(m.parameters()), 1.0)
    assert bool(torch.isinf(m.weight).any()), "post-update recheck missed"
    # no rollback happened: momentum state advanced past its pre-step value
    state = poisoned.state[m.weight]
    assert "momentum_buffer" in state
    assert float(state["momentum_buffer"].abs().sum()) > 0.0


def test_poststep_poisoned_optimizer_state_is_fatal():
    class _PoisonStateAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            super().step(closure)
            p0 = self.param_groups[0]["params"][0]
            with torch.no_grad():
                self.state[p0]["exp_avg"][0, 0] = float("nan")

    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = _PoisonStateAdamW(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    _fresh_grads(m, x)
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, (m(x) ** 2).mean(),
                               list(m.parameters()), 1.0)
    st = opt.state[next(iter(m.parameters()))]["exp_avg"]
    assert bool(torch.isnan(st).any()), \
        "optimizer state was rolled back — snapshot mechanism reappeared"


def test_optimizer_step_exception_wrapped_as_fatal():
    class _ExplodingSGD(torch.optim.SGD):
        def step(self, closure=None):
            raise ZeroDivisionError("boom inside step")

    m = torch.nn.Linear(4, 4)
    opt = _ExplodingSGD(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    _fresh_grads(m, x)
    with pytest.raises(FatalRunInvalidError) as ei:
        guarded_optimizer_step(opt, (m(x) ** 2).mean(),
                               list(m.parameters()), 1.0)
    assert isinstance(ei.value.__cause__, ZeroDivisionError)


def test_happy_path_steps_and_validates_standard_state():
    for opt_cls in (torch.optim.SGD, torch.optim.AdamW):
        m = torch.nn.Linear(4, 4)
        opt = opt_cls(m.parameters(), lr=0.1)
        x = torch.randn(6, 4)
        before = m.weight.detach().clone()
        _fresh_grads(m, x)
        guarded_optimizer_step(opt, (m(x) ** 2).mean(),
                               list(m.parameters()), 1.0)
        assert not bool(torch.equal(m.weight.detach(), before)), \
            f"{opt_cls.__name__}: happy-path step did not run"
        validate_optimizer_state_standard_and_finite(opt)


def test_complex_and_nonstandard_optimizer_states_fail_closed():
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(3, 2)
    _fresh_grads(m, x)
    opt.step()
    p = next(iter(m.parameters()))

    # complex tensors inspected correctly via torch.isfinite
    opt.state[p]["cz"] = torch.tensor([1 + 1j, -0.5 - 2j])
    validate_optimizer_state_standard_and_finite(opt)
    opt.state[p]["cz"] = torch.tensor([1 + 1j, float("nan") * 1j])
    with pytest.raises(NonFiniteTrainingStateError):
        validate_optimizer_state_standard_and_finite(opt)
    del opt.state[p]["cz"]

    # custom object: uninspectable -> fatal, never silently finite
    opt.state[p]["weird"] = object()
    with pytest.raises(OptimizerStateInspectionError):
        validate_optimizer_state_standard_and_finite(opt)
    del opt.state[p]["weird"]

    # cyclic container: fatal
    cyc = {}
    opt.state[p]["cyc"] = cyc
    cyc["self"] = cyc
    with pytest.raises(OptimizerStateInspectionError):
        validate_optimizer_state_standard_and_finite(opt)
    del opt.state[p]["cyc"]
    validate_optimizer_state_standard_and_finite(opt)


def test_assert_all_finite_handles_complex_tensors():
    ok = {"a": [torch.tensor([1 + 1j]), torch.zeros(2)]}
    assert_all_finite(ok, where="t")
    with pytest.raises(NonFiniteStateError):
        assert_all_finite({"a": torch.tensor([float("inf") + 0j])})
    with pytest.raises(NonFiniteStateError):
        assert_all_finite({"a": torch.tensor([complex(0, float("nan"))])})


# ---------------------------------------------------------------------------
# NO per-step snapshot regression (instrumented hot path)
# ---------------------------------------------------------------------------

def test_hot_path_performs_no_transaction_snapshot(monkeypatch):
    """Regression for the rejected dead end: the guarded hot path must not
    deep-copy optimizer state, clone parameters/gradients/state trees, or
    torch.save anything — on the happy path AND on a fatal step."""
    counts = {"clone": 0, "deepcopy": 0, "save": 0}
    real_clone = torch.Tensor.clone

    def counting_clone(self, *a, **kw):
        counts["clone"] += 1
        return real_clone(self, *a, **kw)

    def counting_deepcopy(x, *a, **kw):
        counts["deepcopy"] += 1
        return x  # sentinel; any deepcopy call fails the assertion anyway

    def counting_save(*a, **kw):
        counts["save"] += 1
        return None  # never actually persist anything

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)
    monkeypatch.setattr(copy, "deepcopy", counting_deepcopy)
    monkeypatch.setattr(torch, "save", counting_save)

    class _PoisonSGD(torch.optim.SGD):
        def step(self, closure=None):
            super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(float("inf"))

    torch.manual_seed(3)
    m = torch.nn.Linear(8, 8)
    x = torch.randn(4, 8)

    # happy path
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    _fresh_grads(m, x)          # backward may legitimately clone
    counts.update(clone=0, deepcopy=0, save=0)
    guarded_optimizer_step(opt, (m(x) ** 2).mean(), list(m.parameters()), 1.0)
    assert counts == {"clone": 0, "deepcopy": 0, "save": 0}, \
        f"hot path performed transaction work: {counts}"

    # fatal post-step path: same zero-snapshot guarantee
    poisoned = _PoisonSGD(m.parameters(), lr=1e-2)
    _fresh_grads(m, x)
    counts.update(clone=0, deepcopy=0, save=0)
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(poisoned, (m(x) ** 2).mean(),
                               list(m.parameters()), 1.0)
    assert counts == {"clone": 0, "deepcopy": 0, "save": 0}, \
        f"fatal path performed rollback/snapshot work: {counts}"


def test_fatal_step_cannot_emit_success_artifacts(tmp_path):
    """A mini training loop with cmd_train's artifact contract: when the
    guarded step goes fatal mid-run, ONLY a fatal status file may exist —
    never train_report.json / best_params.pt / run_manifest.json."""
    from latent_lab.train.checkpointing import RUN_STATUS_FILE

    class _PoisonAdamW(torch.optim.AdamW):
        def __init__(self, params, *, blow_up_at, **kw):
            super().__init__(params, **kw)
            self.blow_up_at = blow_up_at
            self.calls = 0

        def step(self, closure=None):
            super().step(closure)
            self.calls += 1
            if self.calls >= self.blow_up_at:
                p0 = self.param_groups[0]["params"][0]
                with torch.no_grad():
                    self.state[p0]["exp_avg_sq"].fill_(float("nan"))

    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4)
    opt = _PoisonAdamW(m.parameters(), blow_up_at=3, lr=0.1)
    x = torch.randn(4, 4)
    out = tmp_path / "run"
    out.mkdir()
    write_run_status(out, "running")
    steps = []
    with pytest.raises(FatalRunInvalidError):
        try:
            for step in range(10):
                _fresh_grads(m, x)
                guarded_optimizer_step(opt, (m(x) ** 2).mean(),
                                       list(m.parameters()), 1.0)
                steps.append(step)
        except FatalRunInvalidError as e:
            write_run_status(out, "fatal", error=str(e))
            raise
    assert len(steps) == 2, "fatal step did not stop the loop immediately"
    st = read_run_status(out)
    assert st["status"] == "fatal"
    assert not (out / "train_report.json").exists()
    assert not (out / "best_params.pt").exists()
    assert not (out / "run_manifest.json").exists()
    with pytest.raises(Exception):  # status blocks evidence readers
        require_complete_run(out)


def test_recovery_uses_last_committed_bundle_in_fresh_runtime(tmp_path):
    """Recovery contract: the last atomically committed identity-bound
    bundle restores into a FRESHLY CONSTRUCTED runtime bit-exactly."""
    layers = 4
    rec = _nonzero_adapted_rec(_tiny_qwen35(11, layers))
    path = tmp_path / "best_params.pt"
    cfg = _cfg(mode="D-full", interval=[0, layers], k=1, max_k=4,
               lora_r=4, lora_alpha=8.0)
    committed = rec.adapter_state_dict()
    rec.export_adapter_bundle(path, model_id="tiny-qwen35", revision=REV_A,
                              config=cfg, metrics={"val_acc": 0.25})

    # hostile in-memory drift AFTER commit (the mutated process is abandoned)
    with torch.no_grad():
        for l in rec.injected:
            l.lora_B.fill_(float("nan"))

    fresh = _nonzero_adapted_rec(_tiny_qwen35(23, layers))  # new weights
    fresh.load_adapter_bundle(path, model_id="tiny-qwen35", revision=REV_A,
                              config=cfg)
    now = fresh.adapter_state_dict()
    assert set(now) == set(committed)
    for k in committed:
        assert torch.equal(now[k], committed[k]), \
            f"{k}: recovered state differs from committed bundle"


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
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        rec = _nonzero_adapted_rec(_tiny_qwen35(11, 4), interval=(0, 4))
        assert rec.clock.weight.dtype == torch.float32
        assert all(p.dtype == torch.float32
                   for p in rec.trainable_parameters())
    finally:
        torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
# identity-bound v2 bundles: digest + recipe binding
# ---------------------------------------------------------------------------

def test_bundle_v2_roundtrip_digest_and_persisted_metrics(tmp_path):
    torch.manual_seed(7)
    sd = {
        "lora.0.A": torch.randn(3, 5),
        "lora.0.B": torch.randn(7, 3) * 0.01,
        "clock.weight": torch.randn(5, 5),
    }
    path = tmp_path / "best_params.pt"
    bundle = save_adapter_bundle(path, sd, model_id="M", revision=REV_A.upper(),
                                 recipe=dict(RECIPE),
                                 metrics={"acc": 0.5})
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir()), \
        "tmp file leaked"
    assert bundle["revision"] == REV_A, "revision not normalized"
    assert bundle["format_version"] == 2
    assert bundle["recipe"]["suite_sha256"] == SUITE_SHA
    assert len(bundle["content_digest"]) == 64

    loaded = load_adapter_bundle(path, model_id="M", revision=REV_A,
                                 recipe=dict(RECIPE))
    assert set(loaded) == set(sd)
    for key, v in sd.items():
        assert torch.equal(loaded[key], v), f"{key} not bit-exact"

    with pytest.raises(NonFiniteMetricError):
        save_adapter_bundle(tmp_path / "b2.pt", sd, model_id="M",
                            revision=REV_A, recipe=dict(RECIPE),
                            metrics={"acc": float("nan")})

    tampered = {**bundle, "metrics": {"acc": float("inf")}}
    p2 = tmp_path / "tampered_metric.pt"
    torch.save(tampered, p2)
    with pytest.raises((NonFiniteMetricError, AdapterBundleError)):
        load_adapter_bundle(p2, model_id="M", revision=REV_A,
                            recipe=dict(RECIPE))


def test_bundle_v2_digest_detects_finite_tensor_tampering(tmp_path):
    sd = _state(1.0, "lora.0.A")
    path = tmp_path / "best_params.pt"
    save_adapter_bundle(path, sd, model_id="M", revision=REV_A,
                        recipe=dict(RECIPE))

    raw = torch.load(path, weights_only=True)
    # FINITE value edit (not NaN/inf): metadata still consistent...
    raw["tensors"]["lora.0.A"]["data"][0, 0] = 12345.0
    torch.save(raw, path)
    with pytest.raises(AdapterBundleError) as ei:
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=dict(RECIPE))
    assert "digest" in str(ei.value).lower()

    # dropping the digest field must also fail closed
    raw = torch.load(path, weights_only=True)
    del raw["content_digest"]
    torch.save(raw, path)
    with pytest.raises(AdapterBundleSchemaError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=dict(RECIPE))


def test_equal_shaped_weights_into_different_recipe_rejected(tmp_path):
    sd = {"lora.0.A": torch.randn(4, 4)}
    path = tmp_path / "best_params.pt"
    save_adapter_bundle(path, sd, model_id="M", revision=REV_A,
                        recipe=dict(RECIPE))

    other_interval = _recipe(interval=[2, 4])              # same shapes!
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=other_interval)
    other_mode = _recipe(mode="D-full")
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=other_mode)
    other_suite = {**RECIPE, "suite_sha256": "cd" * 32}
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=other_suite)
    changed_lr = {**RECIPE, "lr": 9.9e-4}   # materially different identity
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=changed_lr)
    extra_key = {**RECIPE, "unexpected_field": 1}
    with pytest.raises(AdapterBundleSchemaError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=extra_key)
    missing_key = {k: v for k, v in RECIPE.items() if k != "seed"}
    with pytest.raises(AdapterBundleSchemaError):
        load_adapter_bundle(path, model_id="M", revision=REV_A,
                            recipe=missing_key)
    # exact match still passes
    load_adapter_bundle(path, model_id="M", revision=REV_A,
                        recipe=dict(RECIPE))


def test_bundle_identity_mismatch_rejected(tmp_path):
    sd = _state(1.0, "lora.0.A")
    path = tmp_path / "best_params.pt"
    save_adapter_bundle(path, sd, model_id="model-a", revision=REV_A,
                        recipe=dict(RECIPE))

    loaded = load_adapter_bundle(path, model_id="model-a", revision=REV_A,
                                 recipe=dict(RECIPE))
    assert bool(torch.equal(loaded["lora.0.A"], sd["lora.0.A"]))
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="model-a", revision=REV_B,
                            recipe=dict(RECIPE))
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="model-b", revision=REV_A,
                            recipe=dict(RECIPE))


# ---------------------------------------------------------------------------
# atomic adapter loads into a live runtime
# ---------------------------------------------------------------------------

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
        load_adapter_bundle(corrupt, model_id="m", revision=REV_A,
                            recipe=dict(RECIPE))


def test_late_copy_failure_restores_targets_bit_exactly():
    """A persistently failing copy on the SECOND target must leave the
    first target restored too (staged candidates + snapshot restore)."""

    class _BadCopyParameter(torch.nn.Parameter):
        def copy_(self, src):
            raise RuntimeError("hostile storage failure")

    rec = _fake_rec()
    torch.manual_seed(9)
    good = {
        "lora.0.A": torch.randn(2, 4) * 0.1,
        "lora.0.B": torch.randn(4, 2) * 0.1,
        "clock.weight": torch.randn(3, 4) * 0.1,
    }
    LocalizedRecurrence.load_adapter_state(rec, {k: v.clone()
                                                 for k, v in good.items()})
    baseline = _snapshot([rec.injected[0].lora_A, rec.injected[0].lora_B])

    original_weight = rec.clock.weight
    bad_param = _BadCopyParameter(original_weight.data)
    rec.clock.weight = bad_param

    drifted = {**good, "lora.0.A": torch.randn(2, 4)}
    with pytest.raises(AdapterBundleError) as ei:
        LocalizedRecurrence.load_adapter_state(rec, drifted)
    assert "rolled back" in str(ei.value)
    assert _unchanged([rec.injected[0].lora_A, rec.injected[0].lora_B],
                      baseline), "earlier targets survived a late failure"
    rec.clock.weight = original_weight


# ---------------------------------------------------------------------------
# cached localized/full equivalence across a nonzero fp32 roundtrip
# ---------------------------------------------------------------------------

def _assert_cache_and_positions_consumed(rec, ids, k):
    """Prove the latent loop genuinely consumes cache + positions."""
    t = ids.shape[1]
    with torch.no_grad():
        cache_a, z0_a = rec._encode(ids)
        cache_b, z0_b = rec._encode(ids)
        cache_c, _ = rec._encode(ids)
        assert torch.equal(z0_a, z0_b)
        for lyr in cache_c.layers:
            for attr in ("conv_states", "recurrent_states", "keys", "values"):
                v = getattr(lyr, attr, None)
                if torch.is_tensor(v):
                    v.mul_(100.0)
        z_a, pos_a = rec.latent_steps(z0_a, cache_a, t, k)
        z_b, pos_b = rec.latent_steps(z0_b, cache_b, t + 3, k)
        z_c, pos_c = rec.latent_steps(z0_a, cache_c, t, k)
    assert pos_a == t + k and pos_b == t + 3 + k and pos_c == t + k
    assert not torch.equal(z_a, z_b), "loop ignored position arguments"
    assert not torch.equal(z_a, z_c), "loop ignored cache contents"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("interval,k", [((0, 6), 2), ((2, 4), 2),
                                        ((0, 6), 0), ((2, 4), 1)])
def test_cached_localized_full_equivalence_across_fp32_roundtrip(
        tmp_path, interval, k):
    pytest.importorskip("transformers")
    layers = 6

    rec1 = _nonzero_adapted_rec(_tiny_qwen35(11, layers), interval=interval)
    trainables = rec1.trainable_parameters()
    assert all(p.dtype == torch.float32 for p in trainables), "fp32 contract"
    assert all(bool(torch.isfinite(p).all()) for p in trainables)
    assert sum(float(p.abs().sum()) for p in trainables) > 0.0, \
        "adapter must be nonzero for this test"

    if k > 0:
        _assert_cache_and_positions_consumed(rec1, _IDS, k)

    order1, scores1 = _cached_matches_full(rec1, k)
    order1b, scores1b = _cached_matches_full(rec1, k)
    assert order1 == order1b and torch.equal(torch.tensor(scores1),
                                             torch.tensor(scores1b)), \
        "cached scoring is not deterministic"

    mode = "D-full" if interval == (0, layers) else "E-localized"
    path = tmp_path / f"best_{interval[0]}_{interval[1]}_k{k}.pt"
    cfg = _cfg(mode=mode, interval=list(interval), k=k, max_k=4,
               lora_r=4, lora_alpha=8.0)
    rec1.export_adapter_bundle(path, model_id="tiny-qwen35",
                               revision=REV_A, config=cfg,
                               metrics={"val_acc": 0.75})
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

    rec2 = LocalizedRecurrence(_tiny_qwen35(11, layers), None,
                               interval=interval, max_k=4, lora_r=4,
                               lora_alpha=8)
    wrong_rev_kwargs = dict(model_id="tiny-qwen35", revision=REV_B,
                            config=cfg)
    with pytest.raises(AdapterBundleIdentityError):
        rec2.load_adapter_bundle(path, **wrong_rev_kwargs)
    rec2.load_adapter_bundle(path, model_id="tiny-qwen35", revision=REV_A,
                             config=cfg)
    assert all(p.dtype == torch.float32
               for p in rec2.trainable_parameters())

    if k > 0:
        _assert_cache_and_positions_consumed(rec2, _IDS, k)
    order2, scores2 = _cached_matches_full(rec2, k)
    assert order1 == order2
    assert torch.equal(torch.tensor(scores1), torch.tensor(scores2)), \
        "scores changed across the bundle roundtrip"


@pytest.mark.filterwarnings("ignore")
def test_paired_k0_vs_kpositive_same_adapter_seed_suite():
    """A causal K claim needs paired runs: identical adapter/model inputs,
    only K differs; K=0 (tail-only control) must differ from K>0."""
    rec = _nonzero_adapted_rec(_tiny_qwen35(11, 6))
    torch.manual_seed(0)                       # same seed for both arms
    _, scores_k0 = _cached_matches_full(rec, 0)
    _, scores_k1 = _cached_matches_full(rec, 1)
    assert scores_k0 != scores_k1, \
        "K has no causal effect on this adapter — claim unfalsifiable"
    assert all(isinstance(s, float) for s in scores_k0 + scores_k1)


# ---------------------------------------------------------------------------
# strict ablation parsing; shuffle permutation proof; zero_state reset
# ---------------------------------------------------------------------------

def test_ablation_parsers_reject_unknown_modes():
    with pytest.raises(ValueError):
        parse_ablation_cli("make_it_better", 4)
    with pytest.raises(ValueError):
        parse_ablation_cli("shuffle:0,2,1,3", 4)  # legacy prefix removed
    with pytest.raises(ValueError):
        parse_ablation_cli("", 4)

    ab = {"zero_state": True, "frobnicate": True}
    with pytest.raises(ValueError) as ei:
        validate_ablation(ab, 4)
    assert "frobnicate" in str(ei.value)

    with pytest.raises(ValueError):
        validate_ablation({"truncate_k": 99}, 4)
    with pytest.raises(ValueError):
        validate_ablation({"clocks": "sideways"}, 4)


def test_shuffle_requires_full_unique_compute_matched_permutation():
    assert parse_clock_mode("shuffle_perm:2,0,1", 3) == [2, 0, 1]
    assert parse_clock_mode("shuffle_perm:0,1,2,3", 4) == [0, 1, 2, 3]
    for bad in ("shuffle_perm:0,0,1",      # repeat
                "shuffle_perm:0,1",        # short: compute mismatch
                "shuffle_perm:0,1,9",      # out of range
                "shuffle_perm:",           # empty
                "shuffle_perm:0,x"):       # non-integer
        with pytest.raises(ValueError):
            parse_clock_mode(bad, 3)
    # identity/reverse stay available and compute-matched
    assert parse_clock_mode("reverse", 4) == [3, 2, 1, 0]
    assert parse_clock_mode("identity", 4) == [0, 1, 2, 3]
    assert parse_clock_mode("off", 4) == [0, 1, 2, 3]


@pytest.mark.filterwarnings("ignore")
def test_zero_state_resets_all_prompt_derived_cache_state():
    """The zero_state claim is exact: prompt z AND every conv/recurrent/KV
    storage entry derived from the prompt are zeroed before the loop."""
    from latent_lab.backends.hf_qwen import cache_snapshot

    pytest.importorskip("transformers")
    rec = _nonzero_adapted_rec(_tiny_qwen35(11, 6))

    def _all_storage_tensors(snap):
        out = []
        for group in ("conv", "recurrent", "kv"):
            for layer_entry in snap[group].values():
                if torch.is_tensor(layer_entry):
                    out.append(layer_entry)
                elif isinstance(layer_entry, dict):
                    out += [t for t in layer_entry.values()
                            if torch.is_tensor(t)]
                else:
                    out += [t for t in layer_entry
                            if torch.is_tensor(t)]
        return out

    with torch.no_grad():
        cache, z0 = rec._encode(_IDS)
        assert any(float(t.abs().sum()) > 0 for t in _all_storage_tensors(
            cache_snapshot(cache))), "test needs populated prompt cache"
        assert float(z0.abs().sum()) > 0
        rec.latent_steps(z0, cache, _IDS.shape[1], 0,
                         ablate={"zero_state": True})
        # K=0: nothing rewrote the cache after the reset -> all-zero proof
        for t in _all_storage_tensors(cache_snapshot(cache)):
            assert float(t.abs().sum()) == 0.0, \
                "prompt-derived state survived the zero_state reset"

    # behavioral: with prompt-derived state erased, two DIFFERENT prompts of
    # equal length produce IDENTICAL candidate scores
    ids_a = torch.tensor([[7, 11, 13, 5, 200, 3]])
    ids_b = torch.tensor([[42, 2, 99, 8, 201, 4]])
    _, sa, _ = rec.rank_candidates(ids_a, _CANDS, 2,
                                   ablate={"zero_state": True})
    _, sb, _ = rec.rank_candidates(ids_b, _CANDS, 2,
                                   ablate={"zero_state": True})
    assert sa == sb, "zero_state left prompt-identifying information"


# ---------------------------------------------------------------------------
# accelerator coverage: bounded-sync fused checks on real devices
# ---------------------------------------------------------------------------

def _assert_bounded_sync_step(device: str, monkeypatch) -> None:
    """Full guarded step on `device` — precheck, REAL clipping, step,
    parameter + optimizer-state postchecks — while proving:

      * every accumulator the fused checks allocate is created ON that
        bucket's device with an explicit device argument (fails on the
        CPU/device-unspecified accumulator of the rejected candidate);
      * host decisions stay bounded per phase/bucket, never one per
        gradient/parameter/tensor.
    """
    torch.manual_seed(7)
    model = torch.nn.Sequential(*[torch.nn.Linear(8, 8)
                                  for _ in range(2)]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    allocs = []
    reads = {"item": 0, "bool": 0}
    real_zeros = torch.zeros
    real_item = torch.Tensor.item
    real_bool = torch.Tensor.__bool__

    def spy_zeros(*a, **kw):
        t = real_zeros(*a, **kw)
        # attribute allocations to OUR fused checks only: anything torch's
        # own optimizers allocate internally is not a check accumulator
        frame = sys._getframe(1)
        caller = frame.f_globals.get("__name__", "")
        if "checkpointing" in str(caller):
            allocs.append((str(t.device), kw.get("device", "<UNSET>")))
        return t

    def spy_item(self, *a, **kw):
        reads["item"] += 1
        return real_item(self, *a, **kw)

    def spy_bool(self):
        if self.device.type == device.split(":")[0] and self.numel() > 0:
            reads["bool"] += 1
        return real_bool(self)

    x = torch.randn(4, 8, device=device)
    ((model(x) ** 2).mean()).backward()

    monkeypatch.setattr(torch, "zeros", spy_zeros)
    monkeypatch.setattr(torch.Tensor, "item", spy_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", spy_bool)
    # clip 0.01 forces a real foreach clipping pass; then optimizer
    # step() and BOTH postchecks run on-device tensors
    guarded_optimizer_step(opt, (model(x) ** 2).mean(),
                           list(model.parameters()), 0.01)

    assert allocs, "fused reductions allocated nothing"
    # devices that genuinely host checked tensors (params/grads/state);
    # NOTE: standard AdamW keeps float32 `step` counters on CPU even for
    # accelerator params, so a CPU bucket may legitimately exist
    domain_devices = {str(p.device) for p in model.parameters()}
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                domain_devices.add(str(v.device))
    dev_type = device.split(":")[0]
    for got_dev, declared in allocs:
        assert declared != "<UNSET>", \
            f"device-unspecified CPU accumulator allocated: {allocs}"
        assert got_dev == device or got_dev.startswith(dev_type) \
            or got_dev in domain_devices, \
            f"accumulator {got_dev} is off-bucket for {device}: {allocs}"
    n_trainables = sum(1 for _ in model.parameters())
    # phases that may read host-side: pre-params, grad-norm, post-clip,
    # post-params, opt-state (+ the single loss check) — NOT per tensor
    assert reads["item"] <= 6, \
        f"unbounded host reads for {n_trainables} trainables: {reads}"
    assert reads["bool"] <= 2, \
        f"per-gradient host decisions (post-clip loop?) returned: {reads}"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="real CUDA device required")
def test_cuda_guarded_step_bounded_sync_and_on_device_accumulators(
        monkeypatch):
    """CUDA regression: exercises precheck -> clipping -> optimizer.step
    -> parameter/optimizer-state postchecks on real CUDA tensors and
    FAILS on any CPU/device-unspecified accumulator or per-gradient
    host synchronization."""
    _assert_bounded_sync_step("cuda:0"
                              if torch.cuda.device_count() > 1 else "cuda",
                              monkeypatch)


@pytest.mark.skipif(not getattr(torch.backends, "mps", None)
                    or not torch.backends.mps.is_available(),
                    reason="MPS supplement; cannot replace the CUDA test")
def test_mps_guarded_step_bounded_sync_and_on_device_accumulators(
        monkeypatch):
    _assert_bounded_sync_step("mps", monkeypatch)




def test_run_status_blocks_incomplete_evidence(tmp_path):
    write_run_status(tmp_path, "running")
    with pytest.raises(Exception):
        require_complete_run(tmp_path)
    write_run_status(tmp_path, "fatal", error="x")
    with pytest.raises(Exception):
        require_complete_run(tmp_path)
    with pytest.raises(ValueError):
        write_run_status(tmp_path, "banana")
    write_run_status(tmp_path, "complete")
    assert require_complete_run(tmp_path)["status"] == "complete"
    assert read_run_status(tmp_path / "nowhere") is None


def test_atomic_write_json_is_durable_and_digest_linked(tmp_path):
    from latent_lab.train.checkpointing import sha256_file

    p = tmp_path / "report.json"
    sha = atomic_write_json(p, {"a": 1})
    assert sha == sha256_file(p)
    assert json.loads(p.read_text()) == {"a": 1}
    assert not any(f.name.endswith(".tmp") for f in tmp_path.iterdir())


def test_verify_generation_rejects_tampered_artifacts(tmp_path):
    from latent_lab.train.checkpointing import verify_generation

    from tests.artifact_fakes import (
        CHECKPOINT_FILE,
        RUN_MANIFEST_FILE,
        build_verified_run,
    )

    build_verified_run(tmp_path)
    verify_generation(tmp_path)

    # full schema is enforced: a two-hash manifest with no kind/status/
    # identity/recipe is NOT evidence even when its digests cohere
    minimal = {
        "report_sha256":
            __import__("hashlib").sha256(
                (tmp_path / "train_report.json").read_bytes()).hexdigest(),
        "checkpoint_sha256":
            __import__("hashlib").sha256(
                (tmp_path / "best_params.pt").read_bytes()).hexdigest(),
    }
    atomic_write_json(tmp_path / RUN_MANIFEST_FILE, minimal)
    with pytest.raises(Exception):
        verify_generation(tmp_path)

    # rebuild a coherent generation, then flip one checkpoint byte:
    # digests no longer cohere
    build_verified_run(tmp_path)
    with open(tmp_path / CHECKPOINT_FILE, "rb") as fh:
        raw = fh.read()
    with open(tmp_path / CHECKPOINT_FILE, "wb") as fh:
        fh.write(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    with pytest.raises(Exception):
        verify_generation(tmp_path)


# ---------------------------------------------------------------------------
# driver/resume contract: lock + checksum bound
# ---------------------------------------------------------------------------

def test_lock_is_exclusive_and_breaks_stale_holders(tmp_path):
    from latent_lab.bench.artifacts import acquire_lock

    lock = tmp_path / "run.lock"
    with acquire_lock(lock):
        assert lock.exists()
        with pytest.raises(RuntimeError):
            with acquire_lock(lock, timeout_s=0.1):
                pass
    assert not lock.exists(), "lock leaked after release"

    # stale holder (dead pid recorded) is broken automatically
    lock.write_text(json.dumps({"pid": 2**22 * 1024}))  # impossible pid
    entered = False
    with acquire_lock(lock, timeout_s=1.0):
        entered = True
    assert entered


def test_artifact_validation_cli_contract(tmp_path):
    from latent_lab.bench.artifacts import main as artifacts_main

    from tests.artifact_fakes import build_eval_payload, build_verified_run

    # missing everything -> invalid
    assert artifacts_main(["validate-run", str(tmp_path)]) == 1

    # a fully coherent generation validates, and the FULL expected
    # run contract (incl. canonical config digest) binds resume
    from tests.artifact_fakes import run_contract

    build_verified_run(tmp_path)
    assert artifacts_main(["validate-run", str(tmp_path)]) == 0
    rc = run_contract()
    full_flags = ["--expect-model", rc["model_id"], "--expect-rev",
                  rc["revision"], "--expect-suite", rc["suite_sha256"],
                  "--expect-seed", str(rc["seed"]), "--expect-label",
                  rc["label"], "--expect-k", str(rc["k"]),
                  "--expect-steps", str(rc["steps"]),
                  "--expect-config-sha256", rc["config_sha256"]]
    assert artifacts_main(["validate-run", str(tmp_path), *full_flags]) == 0
    wrong_seed = [*full_flags]
    wrong_seed[wrong_seed.index("--expect-seed") + 1] = "2"
    assert artifacts_main(["validate-run", str(tmp_path), *wrong_seed]) == 1
    wrong_model = [*full_flags]
    wrong_model[wrong_model.index("--expect-model") + 1] = "WRONG-MODEL"
    assert artifacts_main(["validate-run", str(tmp_path), *wrong_model]) == 1

    eval_payload = build_eval_payload()
    ep = tmp_path / "eval.json"
    ep.write_text(json.dumps(eval_payload))
    assert artifacts_main(["validate-eval", str(ep)]) == 0
    assert artifacts_main(
        ["validate-eval", str(ep), "--expect-k", "4",
         "--expect-split", "test_id"]) == 0
    assert artifacts_main(
        ["validate-eval", str(ep), "--expect-k", "8"]) == 1

    bad_status = {**eval_payload, "status": "running"}
    ep.write_text(json.dumps(bad_status))
    assert artifacts_main(["validate-eval", str(ep)]) == 1

    mutable_rev = {**eval_payload,
                   "identity": {**eval_payload["identity"],
                                "revision": "main"}}
    ep.write_text(json.dumps(mutable_rev))
    assert artifacts_main(["validate-eval", str(ep)]) == 1
