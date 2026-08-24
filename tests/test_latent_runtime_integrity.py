"""Runtime-integrity tests: best-checkpoint tracking, fail-closed optimizer
steps, identity-bound adapter bundles, fp32 trainables over a
lower-precision backbone, and cached localized/full composition
equivalence — all tiny, deterministic, CPU-only."""

import pytest
import torch

from latent_lab.backends.localized import LoRALinear, LocalizedRecurrence
from latent_lab.train.checkpointing import (
    AdapterBundleError,
    BestCheckpointTracker,
    IdentityMismatchError,
    NonFiniteMetricError,
    NonFiniteStateError,
    NonFiniteTrainingStateError,
    guarded_optimizer_step,
    load_adapter_bundle,
    require_finite_metric,
    save_adapter_bundle,
)

MODEL_ID = "tiny/stub-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------
# tiny deterministic stub runtime (no downloads, no real models)
# ---------------------------------------------------------------------------

def _stub_config(n_layers=4):
    from transformers import Qwen3Config
    return Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
        layer_types=["linear_attention"] * n_layers)


class StubAttention(torch.nn.Module):
    """Submodule names (q_proj/v_proj) are LoRA injection targets."""

    def __init__(self, d, dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        self.v_proj = torch.nn.Linear(d, d, bias=False, dtype=dtype)

    def forward(self, x):
        return self.v_proj(torch.tanh(self.q_proj(x)))


class StubBlock(torch.nn.Module):
    def __init__(self, d, dtype):
        super().__init__()
        self.attn = StubAttention(d, dtype)
        self.ff = torch.nn.Linear(d, d, bias=False, dtype=dtype)

    def forward(self, h, position_embeddings=None, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                **_):
        return h + self.ff(torch.tanh(self.attn(h)))


class StubLM(torch.nn.Module):
    def __init__(self, n_layers=4, d=16, dtype=torch.float32):
        super().__init__()
        self.config = _stub_config(n_layers)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size, d, dtype=dtype)
        self.model.layers = torch.nn.ModuleList(
            [StubBlock(d, dtype) for _ in range(n_layers)])
        self.model.norm = torch.nn.LayerNorm(d, dtype=dtype)
        self.model.rotary_emb = lambda emb, positions: (None, None)
        self.lm_head = torch.nn.Linear(
            d, self.config.vocab_size, bias=False, dtype=dtype)

    def get_output_embeddings(self):
        return self.lm_head


def make_rec(interval=(0, 4), dtype=torch.float32, lora_r=2, k_seed=0):
    torch.manual_seed(k_seed)
    return LocalizedRecurrence(
        StubLM(dtype=dtype), None, interval=interval, max_k=8,
        lora_r=lora_r, grad_checkpoint=False)


# ---------------------------------------------------------------------------
# best checkpoint tracking
# ---------------------------------------------------------------------------

def _state(fill):
    return {"w": torch.full((2, 2), float(fill))}


def test_best_checkpoint_selection_clones_and_keeps_max():
    tr = BestCheckpointTracker(mode="max")
    assert not tr.has_best
    with pytest.raises(RuntimeError, match="fall back"):
        tr.require_state()

    assert tr.update(0.5, _state(1.0), step=1)
    src = _state(2.0)
    assert tr.update(0.9, src, step=2)
    assert not tr.update(0.7, _state(3.0), step=3)
    # tie keeps the earlier checkpoint
    assert not tr.update(0.9, _state(4.0), step=4)

    assert tr.best_metric == 0.9 and tr.best_step == 2
    best = tr.require_state()
    assert torch.equal(best["w"], torch.full((2, 2), 2.0))
    # deep-cloned at acceptance: later source mutation is invisible
    src["w"].fill_(-99.0)
    assert torch.equal(tr.require_state()["w"], torch.full((2, 2), 2.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_rejected(bad):
    tr = BestCheckpointTracker()
    with pytest.raises(NonFiniteMetricError):
        tr.update(bad, _state(1.0), step=1)
    assert not tr.has_best


def test_non_finite_metric_persisted_values_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(NonFiniteMetricError):
            require_finite_metric("val accuracy", bad)
    assert require_finite_metric("acc", 0.75) == 0.75


def test_non_finite_state_rejected():
    tr = BestCheckpointTracker()
    good = tr.update(0.5, _state(1.0), step=1)
    assert good
    for bad_state in (
            {"w": torch.tensor([[float("nan"), 1.0], [1.0, 1.0]])},
            {"w": torch.tensor([[1.0, 1.0], [1.0, float("inf")]])},
            {"w": "not-a-tensor"},
            {}):
        with pytest.raises((NonFiniteStateError, TypeError)):
            tr.update(0.9, bad_state, step=2)
    # earlier accepted checkpoint untouched
    assert tr.best_step == 1 and torch.equal(
        tr.require_state()["w"], torch.ones(2, 2))


# ---------------------------------------------------------------------------
# fail-closed optimizer step
# ---------------------------------------------------------------------------

def test_no_invalid_optimizer_step():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-1)
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    params = list(m.parameters())
    x = torch.randn(3, 4)

    def unchanged():
        for n, p in m.named_parameters():
            assert torch.equal(p.detach(), snap[n])

    # non-finite loss
    with pytest.raises(NonFiniteTrainingStateError, match="loss"):
        guarded_optimizer_step(torch.tensor(float("nan")), opt, params,
                               max_norm=0.5)
    unchanged()

    # non-finite gradient
    ((m(x) ** 2).mean()).backward()
    with torch.no_grad():
        m.weight.grad[0, 0] = float("nan")
    with pytest.raises(NonFiniteTrainingStateError, match="gradient"):
        guarded_optimizer_step(None, opt, params, max_norm=0.5)
    unchanged()
    opt.zero_grad(set_to_none=True)

    # gradients become non-finite AFTER clipping -> still no step
    ((m(x) ** 2).mean()).backward()
    real_clip = torch.nn.utils.clip_grad_norm_

    def poison_after_clip(ps, mn):
        norm = real_clip(ps, mn)
        ps[0].grad[0, 0] = float("nan")
        return norm

    torch.nn.utils.clip_grad_norm_ = poison_after_clip
    try:
        with pytest.raises(NonFiniteTrainingStateError,
                           match="after clipping"):
            guarded_optimizer_step(None, opt, params, max_norm=0.5)
    finally:
        torch.nn.utils.clip_grad_norm_ = real_clip
    unchanged()
    opt.zero_grad(set_to_none=True)

    # parameters non-finite BEFORE step -> refused
    with torch.no_grad():
        m.bias[0] = float("inf")
    with pytest.raises(NonFiniteTrainingStateError, match="parameter"):
        guarded_optimizer_step(torch.zeros(()), opt, params, max_norm=0.5)
    with torch.no_grad():
        m.bias.copy_(snap["bias"])
    opt.zero_grad(set_to_none=True)

    # clean finite step goes through and reports a finite clip norm
    loss = (m(x) ** 2).mean()
    loss.backward()
    norm = guarded_optimizer_step(loss, opt, params, max_norm=0.5)
    assert torch.isfinite(torch.as_tensor(norm))
    assert all(torch.isfinite(p).all() for p in params)
    assert any(not torch.equal(p.detach(), snap[n])
               for n, p in m.named_parameters())

    # parameters corrupted by the update itself are detected immediately
    original_step = opt.step

    def poisoned_step():
        original_step()
        with torch.no_grad():
            m.bias[0] = float("nan")

    opt.step = poisoned_step
    try:
        loss = (m(x) ** 2).mean()
        loss.backward()
        with pytest.raises(NonFiniteTrainingStateError,
                           match="after update"):
            guarded_optimizer_step(loss, opt, params, max_norm=0.5)
    finally:
        opt.step = original_step


# ---------------------------------------------------------------------------
# fp32 trainables over a lower-precision backbone
# ---------------------------------------------------------------------------

def test_lora_masters_fp32_over_bf16_base_layer():
    base = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    lora = LoRALinear(base, r=2, alpha=4.0)
    assert base.weight.dtype == torch.bfloat16
    assert lora.lora_A.dtype == torch.float32
    assert lora.lora_B.dtype == torch.float32
    with torch.no_grad():
        lora.lora_B.normal_(0.0, 0.05)
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    y = lora(x)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()


def test_trainables_fp32_over_bf16_backbone():
    rec = make_rec(dtype=torch.bfloat16)
    trainables = rec.trainable_parameters()
    assert len(trainables) == 2 * len(rec.injected) + 1  # LoRA pairs + clock
    assert all(p.dtype == torch.float32 for p in trainables)
    assert rec.clock.weight.dtype == torch.float32
    assert all(p.dtype == torch.bfloat16
               for p in rec.model.parameters()
               if all(mp is not p for mp in trainables))
    with torch.no_grad():
        rec.clock.weight.add_(torch.randn_like(rec.clock.weight) * 0.01)
    ids = torch.randint(0, 64, (1, 4))
    cache, z0 = rec._encode(ids)
    z, _ = rec.latent_steps(z0, cache, ids.shape[1], k_steps=2)
    assert z.dtype == torch.bfloat16 and torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# identity-bound adapter bundles
# ---------------------------------------------------------------------------

def _random_state(spec, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(shape, generator=g, dtype=dt)
            for k, (shape, dt) in spec.items()}


def test_bundle_save_load_roundtrip_exact(tmp_path):
    rec = make_rec()
    spec = rec.adapter_spec()
    sd = _random_state(spec)
    save_adapter_bundle(tmp_path, sd, model_id=MODEL_ID, revision=REVISION)
    loaded = load_adapter_bundle(tmp_path, model_id=MODEL_ID,
                                 revision=REVISION, expected=spec)
    assert set(loaded) == set(sd)
    for k in sd:
        assert loaded[k].dtype == sd[k].dtype
        assert loaded[k].shape == sd[k].shape
        assert torch.equal(loaded[k], sd[k])  # bit-exact roundtrip
    # loading into the runtime reproduces the saved tensors exactly
    rec.load_adapter_state(loaded)
    current = rec.adapter_state_dict()
    for k in sd:
        assert torch.equal(current[k], sd[k])


def test_identity_mismatch_rejected(tmp_path):
    rec = make_rec()
    save_adapter_bundle(tmp_path, rec.adapter_state_dict(),
                        model_id="other/model", revision="rev-A")
    with pytest.raises(IdentityMismatchError, match="model_id|identity"):
        load_adapter_bundle(tmp_path, model_id=MODEL_ID, revision="rev-A")
    save_adapter_bundle(tmp_path, rec.adapter_state_dict(),
                        model_id=MODEL_ID, revision="rev-B")
    with pytest.raises(IdentityMismatchError):
        load_adapter_bundle(tmp_path, model_id=MODEL_ID, revision=REVISION)


def test_failed_load_is_atomic(tmp_path):
    rec = make_rec()
    spec = rec.adapter_spec()
    snap = {k: p.detach().clone()
            for k, p in rec._adapter_targets().items()}
    save_adapter_bundle(tmp_path, _random_state(spec),
                        model_id=MODEL_ID, revision=REVISION)

    # corrupt one persisted tensor -> bundle loader rejects whole payload
    raw = torch.load(tmp_path / "best_params.pt", weights_only=True)
    raw["tensors"]["lora.0.B"][0, 0] = float("nan")
    torch.save(raw, tmp_path / "best_params.pt")
    with pytest.raises(NonFiniteStateError):
        load_adapter_bundle(tmp_path, model_id=MODEL_ID, revision=REVISION,
                            expected=spec)

    # wrong shape rejected before any copy
    raw = torch.load(tmp_path / "best_params.pt", weights_only=True)
    raw["tensors"] = _random_state(spec)
    raw["tensors"]["lora.0.A"] = torch.zeros(1, 1)
    torch.save(raw, tmp_path / "best_params.pt")
    with pytest.raises(AdapterBundleError, match="shape"):
        load_adapter_bundle(tmp_path, model_id=MODEL_ID, revision=REVISION,
                            expected=spec)

    # missing/extra keys rejected
    bad = _random_state(spec)
    del bad["lora.0.A"]
    with pytest.raises(AdapterBundleError, match="key"):
        rec.load_adapter_state(bad)
    bad = _random_state(spec)
    bad["rogue.weight"] = torch.zeros(1)
    with pytest.raises(AdapterBundleError, match="unknown"):
        rec.load_adapter_state(bad)

    # non-finite state rejected by the runtime itself, nothing copied
    bad = _random_state(spec)
    bad["clock.weight"][0, 0] = float("nan")
    with pytest.raises(NonFiniteStateError):
        rec.load_adapter_state(bad)
    for k, p in rec._adapter_targets().items():
        assert torch.equal(p.detach(), snap[k])


# ---------------------------------------------------------------------------
# cached localized/full composition equivalence
# ---------------------------------------------------------------------------

def _manual_layers(rec, idxs, h, pos):
    pe = rec._rotary(h, h.shape[1], pos)
    pos_ids = rec._pos4(h.shape[1], pos)[0]
    for i in idxs:
        h = rec.base.layers[i](h, position_embeddings=pe,
                               attention_mask=None, position_ids=pos_ids,
                               past_key_values=None, use_cache=False)
    return h


def _reference_pipeline(rec, ids, k):
    """Same composition without the shared cache: encode -> K x interval ->
    tail, applied layer-by-layer, mirroring the per-step clock injection."""
    lo, hi = rec.interval
    h = rec.base.embed_tokens(ids)
    h = _manual_layers(rec, range(rec.n_layers), h, 0)[:, -1:, :]
    pos = ids.shape[1]
    for step in range(k):
        if rec.use_clock:
            tok = torch.tensor([step + 1])
            h = h + rec.clock(tok).view(1, 1, -1).to(h.dtype)
        h = _manual_layers(rec, range(lo, hi), h, pos)
        pos += 1
    return _manual_layers(rec, range(hi, rec.n_layers), h, pos)


@pytest.mark.parametrize("interval,k", [((0, 4), 3), ((1, 3), 2)])
def test_cached_localized_full_equivalence_across_adapter_roundtrip(
        interval, k, tmp_path):
    rec = make_rec(interval=interval)
    torch.manual_seed(11)
    ids = torch.randint(0, 64, (1, 5))

    def run_cached():
        cache, z0 = rec._encode(ids)
        z, pos = rec.latent_steps(z0, cache, ids.shape[1], k)
        return rec.tail_sequence(z, cache, pos)

    # phase 1: zero-init adapters (before roundtrip)
    ref = _reference_pipeline(rec, ids, k)
    got = run_cached()
    assert torch.equal(got, ref)

    # phase 2: nonzero fp32 adapter -> bundle save/load roundtrip
    spec = rec.adapter_spec()
    sd = _random_state(spec, seed=7)
    for v in sd.values():
        v.mul_(0.05)
    assert any(v.abs().sum() > 0 for v in sd.values())
    assert all(v.dtype == torch.float32 for v in sd.values())
    save_adapter_bundle(tmp_path, sd, model_id=MODEL_ID, revision=REVISION)
    loaded = load_adapter_bundle(tmp_path, model_id=MODEL_ID,
                                 revision=REVISION, expected=spec)
    rec.load_adapter_state(loaded)
    assert all(p.dtype == torch.float32 for p in rec.trainable_parameters())
    assert torch.equal(run_cached(), _reference_pipeline(rec, ids, k))
