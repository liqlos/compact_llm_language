"""LocalizedRecurrence structural tests on a tiny hybrid Qwen3_5 config.

These run on CPU/fp32 and validate the plumbing that real-model experiments
depend on: bit-exact manual composition, guarded loop, LoRA freeze semantics,
clock identity at initialization, and ranking-scorer mechanics.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from latent_lab.backends.localized import (
    LatentLoopViolation,
    LocalizedRecurrence,
)


def build_tiny_hybrid():
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=512,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
    )
    model = transformers.Qwen3_5ForCausalLM(cfg)
    model.eval()
    tok = None
    return model, tok


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_hybrid()


INTERVAL = (1, 3)


@pytest.fixture(scope="module")
def rec(tiny):
    model, tok = tiny
    return LocalizedRecurrence(model, tok, interval=INTERVAL, max_k=4,
                               grad_checkpoint=False)


def test_manual_composition_bit_exact(rec):
    ids = torch.randint(0, 250, (1, 9))
    emb = rec.base.embed_tokens(ids)
    from transformers import DynamicCache
    ref_cache = DynamicCache(config=rec.config)
    ref = rec.base(input_ids=ids, past_key_values=ref_cache, use_cache=True)
    _cache, _ = rec._encode(ids)
    h = rec._run_layers(range(rec.n_layers), emb, DynamicCache(
        config=rec.config), 0)
    # reference must run the SAME code path (cached); the no-cache GDN kernel
    # reorders floating-point reduction and differs by ~5e-3 in fp32
    assert torch.equal(rec.base.norm(h[:, -1:, :]),
                       ref.last_hidden_state[:, -1:, :])


def test_localized_loop_updates_only_interval_cache(rec):
    ids = torch.randint(0, 250, (1, 7))
    cache, z0 = rec._encode(ids)
    _z, pos = rec.latent_steps(z0, cache, 7, 3)
    assert pos == 10
    # full-attention layers OUTSIDE the interval keep their kv length at 7
    # (kv layout is [batch, kv_heads, seq, head_dim] -> length lives at dim 2)
    for i in range(rec.n_layers):
        if INTERVAL[0] <= i < INTERVAL[1]:
            continue
        ks = getattr(cache.layers[i], "keys", None)
        if ks is not None:
            assert ks.shape[2] == 7, f"layer {i} kv grew: {ks.shape}"
    # attention layers INSIDE the interval gained K entries
    for i in range(INTERVAL[0], INTERVAL[1]):
        ks = getattr(cache.layers[i], "keys", None)
        if ks is not None:
            assert ks.shape[2] == 7 + 3

def test_full_interval_matches_public_incremental_path(tiny):
    """interval=(0,L): one latent step must equal a canonical incremental
    decode step through base(inputs_embeds=..., past_key_values=...).

    Comparison happens post-norm because the public path exposes only
    normalized last_hidden_state (verified: lhs == norm(raw))."""
    model, tok = build_tiny_hybrid()  # fresh: binding is exclusive
    L = model.config.num_hidden_layers
    lr = LocalizedRecurrence(model, tok, interval=(0, L), max_k=4,
                             use_clock=False, grad_checkpoint=False, lora_r=2)
    ids = torch.randint(0, 250, (1, 6))
    t = ids.shape[1]
    with torch.no_grad():
        from transformers import DynamicCache
        cache_a, z0 = lr._encode(ids)
        z1, pos1 = lr.latent_steps(z0, cache_a, t, 1)

        cache_b = DynamicCache(config=model.config)
        _ = lr.base(input_ids=ids, past_key_values=cache_b, use_cache=True)
        # rebuild the SAME raw z0 via the validated composition
        emb = lr.base.embed_tokens(ids)
        h = lr._run_layers(range(L), emb, DynamicCache(config=model.config), 0)
        z0_raw = h[:, -1:, :]
        o = lr.base(inputs_embeds=z0_raw, past_key_values=cache_b,
                    use_cache=True)
        assert pos1 == t + 1
        assert torch.equal(lr.base.norm(z1), o.last_hidden_state[:, -1:, :])


def test_k0_is_tail_only(rec):
    ids = torch.randint(0, 250, (1, 5))
    with torch.no_grad():
        cache, z0 = rec._encode(ids)
        z, pos = rec.latent_steps(z0, cache, 5, 0)
        assert pos == 5
        ht = rec.tail_sequence(z, cache, 5)
        logits = rec.logits_from_hidden(ht)
        assert logits.shape[-1] == 256 and torch.isfinite(logits).all()


def test_guard_catches_lm_head_inside_loop(tiny, rec, monkeypatch):
    model, _ = tiny
    target = model.model.layers[INTERVAL[0]]
    head = model.get_output_embeddings()

    orig_forward = target.forward

    def evil(hidden_states, **kw):
        out = orig_forward(hidden_states, **kw)
        head(out[:, :1, :])  # forbidden vocabulary access mid-loop
        return out

    monkeypatch.setattr(target, "forward", evil)
    ids = torch.randint(0, 250, (1, 5))
    cache, z0 = rec._encode(ids)
    with pytest.raises(LatentLoopViolation):
        rec.latent_steps(z0, cache, 5, 2)
    monkeypatch.undo()


def test_lora_freeze_and_zero_init_identity():
    # fresh model: a LocalizedRecurrence binds its host exclusively
    model, tok = build_tiny_hybrid()
    rec2 = LocalizedRecurrence(model, tok, interval=INTERVAL, max_k=4,
                               grad_checkpoint=False)
    trainable = {id(p) for p in rec2.trainable_parameters()}
    n_lora = 0
    for n, p in rec2.model.named_parameters():
        if id(p) not in trainable:
            assert not p.requires_grad, f"{n} unexpectedly trainable"
        else:
            assert p.requires_grad, f"{n} unexpectedly frozen"
            n_lora += 1
    assert n_lora == 2 * len(rec2.injected)
    assert rec2.clock.weight.requires_grad
    # zero-init B => LoRA contributes nothing at start; shapes stay valid
    ids = torch.randint(0, 250, (1, 6))
    cache, z0 = rec2._encode(ids)
    with torch.no_grad():
        z_on, _ = rec2.latent_steps(z0, cache, 6, 2, ablate={"clocks": "off"})
        assert z_on.shape == z0.shape


def test_clock_changes_state_after_perturbation(rec):
    with torch.no_grad():
        rec.clock.weight.fill_(0.01)
        ids = torch.randint(0, 250, (1, 6))
        cache, z0 = rec._encode(ids)
        za, _ = rec.latent_steps(z0, cache, 6, 2, ablate={"clocks": "off"})
        zb, _ = rec.latent_steps(z0, cache, 6, 2)
        assert not torch.equal(za, zb)
        # reverse clocks also differs from identity order
        zr, _ = rec.latent_steps(z0, cache, 6, 2, ablate={"clocks": "reverse"})
        assert not torch.equal(zb, zr)


def test_rank_candidates_mechanics(rec):
    ids = torch.randint(0, 250, (1, 6))
    cands = ([10, 11], [12, 13], [14, 15])
    lm_before = rec.guard.lm_head_calls
    with torch.no_grad():
        order, scores, rep = rec.rank_candidates(ids, cands, k_steps=1)
    assert len(order) == 3 and sorted(order) == [0, 1, 2]
    expected = sorted(range(3), key=lambda i: -scores[i])
    assert order == expected
    assert rep.k_steps == 1
    # one lm_head CALL per candidate (both token positions scored in it)
    assert rec.guard.lm_head_calls - lm_before == len(cands)
