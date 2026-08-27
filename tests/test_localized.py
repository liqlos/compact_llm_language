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
    LoRALinear,
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
    return LocalizedRecurrence(model, tok, interval=INTERVAL, max_k=4)


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
                             use_clock=False, lora_r=2)
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
    rec2 = LocalizedRecurrence(model, tok, interval=INTERVAL, max_k=4)
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


def test_lora_runtime_toggle_is_exact_frozen_base_bypass():
    base = torch.nn.Linear(8, 8, bias=False)
    lora = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        lora.lora_B.normal_(mean=0.0, std=0.2)
        lora.enabled = False
        disabled = lora(x)
        assert torch.equal(disabled, base(x))
        lora.enabled = True
        enabled = lora(x)
    assert not torch.equal(enabled, disabled)


def test_recurrence_only_lora_routes_stages_and_makes_k0_adapter_free():
    model, tok = build_tiny_hybrid()
    rec2 = LocalizedRecurrence(
        model, tok, interval=INTERVAL, max_k=2, lora_r=2,
        use_clock=False, recurrence_only_lora=True)
    assert rec2.runtime_contract()["adapter_activation_policy"] == \
        "recurrence_only"
    assert all(not adapter.enabled for adapter in rec2.injected)

    calls = []
    handles = [adapter.register_forward_pre_hook(
        lambda module, _inputs: calls.append(module.enabled))
        for adapter in rec2.injected]
    ids = torch.randint(0, 250, (1, 6))
    answer = torch.tensor([[10, 11]])
    with torch.no_grad():
        cache, z0 = rec2._encode(ids)
        assert calls and not any(calls), "prompt prefill used LoRA residuals"
        calls.clear()
        z1, pos = rec2.latent_steps(z0, cache, ids.shape[1], 1)
        assert calls and all(calls), "latent recurrence did not enable LoRA"
        calls.clear()
        rec2._score_candidate_tokens(z1, cache, pos, answer)
        assert calls and not any(calls), "candidate readout used LoRA residuals"
    for handle in handles:
        handle.remove()
    assert all(not adapter.enabled for adapter in rec2.injected)

    candidates = ([10, 11], [12, 13])
    with torch.no_grad():
        for adapter in rec2.injected:
            adapter.lora_B.zero_()
        k0_before, _ = rec2.score_candidates(ids, candidates, 0)
        k1_before, _ = rec2.score_candidates(ids, candidates, 1)
        generator = torch.Generator().manual_seed(17)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(
                torch.randn(adapter.lora_B.shape, generator=generator) * 0.2)
        k0_after, k0_report = rec2.score_candidates(ids, candidates, 0)
        k1_after, _ = rec2.score_candidates(ids, candidates, 1)

    def sums(rows):
        return [row.raw_sum_logprob for row in rows]

    assert sums(k0_before) == sums(k0_after), \
        "K=0 changed when recurrence-only adapter weights changed"
    assert sums(k1_before) != sums(k1_after), \
        "K>0 ignored recurrence-only adapter weights"
    compute = k0_report.extra["compute"]
    assert compute["prefill_adapter_active"] is False
    assert compute["recurrence_adapter_active"] is True
    assert compute["candidate_adapter_active"] is False

    with pytest.raises(ValueError, match="no trainable path at K=0"):
        rec2.loss_on_example(ids, answer, 0)

    for parameter in rec2.trainable_parameters():
        parameter.grad = None
    loss = rec2.loss_on_example(ids, answer, 1)
    assert loss.requires_grad
    loss.backward()
    assert any(parameter.grad is not None
               and bool(torch.isfinite(parameter.grad).all())
               and float(parameter.grad.abs().sum()) > 0.0
               for parameter in rec2.trainable_parameters())
    assert all(not adapter.enabled for adapter in rec2.injected)


def build_neutral_delta(max_k=4):
    model, tok = build_tiny_hybrid()
    return model, LocalizedRecurrence(
        model, tok, interval=(0, model.config.num_hidden_layers),
        max_k=max_k, lora_r=2, recurrence_only_lora=True,
        neutral_delta=True)


def _token_logprobs(details):
    return [row.token_logprobs for row in details]


def test_neutral_delta_zero_gates_execute_k4_but_are_bit_exact_k0():
    _model, rec2 = build_neutral_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11, 12], [13, 14])
    k0, report0 = rec2.score_candidates(ids, candidates, 0)
    k4, report4 = rec2.score_candidates(ids, candidates, 4)

    assert rec2.step_gates.dtype == torch.float32
    assert rec2.step_gates.shape == (4,)
    assert torch.count_nonzero(rec2.step_gates).item() == 0
    assert "step_gates" in rec2.adapter_state_dict()
    assert any(parameter is rec2.step_gates
               for parameter in rec2.trainable_parameters())
    assert _token_logprobs(k4) == _token_logprobs(k0)
    assert report4.extra["recurrence_interval_layer_applications"] == 16
    assert report4.extra["readout_position"] == ids.shape[1]
    assert report0.extra["readout_position"] == ids.shape[1]
    assert report4.extra["recurrence_cache_at_readout"] == \
        "target_prompt_only"


def test_neutral_delta_full_reset_is_bit_exact_k0_after_opening_adapter():
    _model, rec2 = build_neutral_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11, 12], [13, 14, 15])
    generator = torch.Generator().manual_seed(91)
    with torch.no_grad():
        rec2.step_gates.fill_(0.7)
        rec2.clock.weight.normal_(mean=0.0, std=0.1, generator=generator)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.1)
    k0, _ = rec2.score_candidates(ids, candidates, 0)
    reset, report = rec2.score_candidates(
        ids, candidates, 4,
        ablate={"reset_state": True, "reset_cache": True})
    assert _token_logprobs(reset) == _token_logprobs(k0)
    assert report.extra["compute"]["k_loops"] == 4
    assert report.extra["readout_position"] == ids.shape[1]


def test_neutral_delta_zero_and_noise_intervene_on_delta_not_absolute_z():
    _model, rec2 = build_neutral_delta()
    ids = torch.randint(0, 250, (1, 7))
    generator = torch.Generator().manual_seed(92)
    with torch.no_grad():
        rec2.step_gates.fill_(0.6)
        rec2.clock.weight.normal_(mean=0.0, std=0.1, generator=generator)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.1)

        def run(ablate=None):
            cache, z0 = rec2._encode(ids)
            z, _ = rec2.latent_steps(
                z0, cache, ids.shape[1], 4, ablate=ablate)
            return z0, z

        z0, clean = run()
        zero_z0, zero = run({"zero_state": True})
        noise_z0, noise = run({"noise_state": True, "noise_seed": 17})

    assert torch.equal(z0, zero_z0) and torch.equal(z0, noise_z0)
    assert torch.equal(zero, z0)
    assert not torch.equal(zero, torch.zeros_like(zero))
    clean_delta = clean - z0
    noise_delta = noise - z0
    assert torch.allclose(noise_delta.norm(), clean_delta.norm(),
                          rtol=1e-5, atol=1e-6)
    assert not torch.equal(noise_delta, clean_delta)


def test_neutral_delta_swap_applies_partner_delta_to_target_z0(monkeypatch):
    _model, rec2 = build_neutral_delta()
    target_ids = torch.randint(0, 120, (1, 7))
    partner_ids = torch.randint(130, 250, (1, 9))
    generator = torch.Generator().manual_seed(93)
    with torch.no_grad():
        rec2.step_gates.fill_(0.6)
        rec2.clock.weight.normal_(mean=0.0, std=0.1, generator=generator)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.1)
        _target_cache, target_z0 = rec2._encode(target_ids)
        partner_cache, partner_z0 = rec2._encode(partner_ids)
        partner_z, _ = rec2.latent_steps(
            partner_z0, partner_cache, partner_ids.shape[1], 4)
        expected_delta = partner_z - partner_z0

    captured = []
    original = rec2._score_candidate_tokens

    def capture(z, cache, pos, candidate_ids, *, grad=False,
                latent_delta=None):
        captured.append((z.detach().clone(), latent_delta.detach().clone()))
        return original(z, cache, pos, candidate_ids, grad=grad,
                        latent_delta=latent_delta)

    monkeypatch.setattr(rec2, "_score_candidate_tokens", capture)
    _details, report = rec2.score_candidates(
        target_ids, ([10, 11],), 4, ablate={"swap_state": True},
        partner_input_ids=partner_ids)

    readout, actual_delta = captured[0]
    assert torch.equal(actual_delta, expected_delta)
    assert torch.equal(readout, target_z0 + expected_delta)
    assert not torch.equal(actual_delta, partner_z - target_z0)
    assert report.extra["latent_state_source"] == "partner_delta"
    assert report.extra["latent_state_ablation_target"] == "latent_delta"


def test_neutral_delta_multitoken_k4_matches_direct_incremental_hf_at_init():
    model, rec2 = build_neutral_delta()
    ids = torch.randint(0, 250, (1, 8))
    candidate = torch.tensor([[10, 11, 12, 13]])
    details, _ = rec2.score_candidates(ids, [candidate[0].tolist()], 4)

    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    oracle = []
    with torch.no_grad():
        output = model(input_ids=ids, past_key_values=cache, use_cache=True)
        for index, target in enumerate(candidate[0]):
            logp = torch.log_softmax(output.logits[:, -1:, :].float(), dim=-1)
            oracle.append(float(logp[0, 0, target]))
            if index + 1 < candidate.shape[1]:
                output = model(
                    input_ids=candidate[:, index:index + 1],
                    past_key_values=cache, use_cache=True)
    assert details[0].token_logprobs == tuple(oracle)


def test_neutral_delta_gate_gradient_at_init_then_lora_gradient_after_open():
    _model, rec2 = build_neutral_delta(max_k=2)
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11], [12, 13], [14, 15])

    loss = rec2.candidate_ce_loss_on_example(
        ids, candidates, gold_index=1, k_steps=2)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    gate_grad = rec2.step_gates.grad
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert torch.count_nonzero(gate_grad).item() > 0

    for parameter in rec2.trainable_parameters():
        parameter.grad = None
    with torch.no_grad():
        rec2.step_gates.fill_(0.1)
    opened_loss = rec2.candidate_ce_loss_on_example(
        ids, candidates, gold_index=1, k_steps=2)
    assert opened_loss.requires_grad and torch.isfinite(opened_loss)
    opened_loss.backward()
    lora_gradients = [
        adapter.lora_B.grad for adapter in rec2.injected
        if adapter.lora_B.grad is not None
    ]
    assert lora_gradients
    assert all(torch.isfinite(gradient).all()
               for gradient in lora_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0
               for gradient in lora_gradients)


def test_neutral_delta_rejects_non_full_or_shared_adapter_modes():
    model, tok = build_tiny_hybrid()
    with pytest.raises(ValueError, match="full-decoder"):
        LocalizedRecurrence(
            model, tok, interval=INTERVAL, max_k=2,
            recurrence_only_lora=True, neutral_delta=True)
    model, tok = build_tiny_hybrid()
    with pytest.raises(ValueError, match="recurrence_only_lora"):
        LocalizedRecurrence(
            model, tok, interval=(0, model.config.num_hidden_layers),
            max_k=2, neutral_delta=True)


def build_paired_delta(max_k=4, workspace_slots=1):
    model, tok = build_tiny_hybrid()
    return model, LocalizedRecurrence(
        model, tok, interval=(0, model.config.num_hidden_layers),
        max_k=max_k, lora_r=2, recurrence_only_lora=True,
        paired_delta=True, workspace_slots=workspace_slots)


def test_paired_delta_init_is_bit_exact_k0_and_has_no_step_gates():
    model, rec2 = build_paired_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11, 12], [13, 14])
    k0, report0 = rec2.score_candidates(ids, candidates, 0)
    k4, report4 = rec2.score_candidates(ids, candidates, 4)

    assert rec2.neutral_delta is True and rec2.paired_delta is True
    assert rec2.step_gates is None
    assert "step_gates" not in rec2.adapter_state_dict()
    assert _token_logprobs(k4) == _token_logprobs(k0)
    assert report4.extra["recurrence_interval_layer_applications"] == \
        2 * 4 * model.config.num_hidden_layers
    assert report4.extra["recurrence_passes_per_step"] == 2
    assert report4.extra["readout_position"] == ids.shape[1]
    assert report0.extra["readout_position"] == ids.shape[1]
    assert rec2.runtime_contract()["recurrence_passes_per_step"] == 2


def test_paired_delta_executes_two_full_decoder_passes_per_step():
    model, rec2 = build_paired_delta(max_k=3)
    ids = torch.randint(0, 250, (1, 7))
    with torch.no_grad():
        cache, z0 = rec2._encode(ids)
    counts = [0] * model.config.num_hidden_layers
    handles = []
    for index, layer in enumerate(rec2.base.layers):
        handles.append(layer.register_forward_hook(
            lambda _module, _inputs, _output, i=index:
            counts.__setitem__(i, counts[i] + 1)))
    try:
        with torch.no_grad():
            rec2.latent_steps(z0, cache, ids.shape[1], 3)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == [2 * 3] * model.config.num_hidden_layers


def test_paired_delta_first_backward_trains_b_and_clock_then_a():
    _model, rec2 = build_paired_delta(max_k=2)
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11], [12, 13], [14, 15])
    optimizer = torch.optim.SGD(rec2.trainable_parameters(), lr=0.05)

    loss = rec2.candidate_ce_loss_on_example(
        ids, candidates, gold_index=1, k_steps=2)
    loss.backward()
    b_gradients = [adapter.lora_B.grad for adapter in rec2.injected]
    a_gradients = [adapter.lora_A.grad for adapter in rec2.injected]
    assert any(gradient is not None
               and torch.count_nonzero(gradient).item() > 0
               for gradient in b_gradients)
    assert rec2.clock.weight.grad is not None
    assert torch.count_nonzero(rec2.clock.weight.grad).item() > 0
    assert all(gradient is None or torch.count_nonzero(gradient).item() == 0
               for gradient in a_gradients)

    optimizer.step()
    optimizer.zero_grad()
    loss2 = rec2.candidate_ce_loss_on_example(
        ids, candidates, gold_index=1, k_steps=2)
    loss2.backward()
    assert any(adapter.lora_A.grad is not None
               and torch.count_nonzero(adapter.lora_A.grad).item() > 0
               for adapter in rec2.injected)
    assert all(not adapter.enabled for adapter in rec2.injected)


def test_paired_delta_full_reset_is_bit_exact_k0_after_training_signal():
    _model, rec2 = build_paired_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11, 12], [13, 14, 15])
    generator = torch.Generator().manual_seed(94)
    with torch.no_grad():
        rec2.clock.weight.normal_(mean=0.0, std=0.1, generator=generator)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.1)
    k0, _ = rec2.score_candidates(ids, candidates, 0)
    reset, report = rec2.score_candidates(
        ids, candidates, 4,
        ablate={"reset_state": True, "reset_cache": True})
    assert _token_logprobs(reset) == _token_logprobs(k0)
    assert report.extra["compute"]["k_loops"] == 4


def test_paired_workspace_m1_default_is_bit_exact_after_perturbation():
    model_a, tok = build_tiny_hybrid()
    rec_a = LocalizedRecurrence(
        model_a, tok, interval=(0, model_a.config.num_hidden_layers),
        max_k=4, lora_r=2, recurrence_only_lora=True, paired_delta=True)
    model_b, tok = build_tiny_hybrid()
    rec_b = LocalizedRecurrence(
        model_b, tok, interval=(0, model_b.config.num_hidden_layers),
        max_k=4, lora_r=2, recurrence_only_lora=True, paired_delta=True,
        workspace_slots=1)
    generator = torch.Generator().manual_seed(302)
    with torch.no_grad():
        rec_a.clock.weight.normal_(std=0.03, generator=generator)
        for adapter in rec_a.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.03)
    rec_b.load_adapter_state(rec_a.adapter_state_dict())
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17]])
    candidates = ([19, 23], [29, 31, 37])

    scores_a, _ = rec_a.score_candidates(ids, candidates, 4)
    scores_b, _ = rec_b.score_candidates(ids, candidates, 4)
    assert _token_logprobs(scores_a) == _token_logprobs(scores_b)

    model_c, tok = build_tiny_hybrid()
    rec_c = LocalizedRecurrence(
        model_c, tok, interval=(0, model_c.config.num_hidden_layers),
        max_k=4, lora_r=2, recurrence_only_lora=True, paired_delta=True,
        workspace_slots=4)
    rec_c.load_adapter_state(rec_a.adapter_state_dict())
    m1_k0, _ = rec_a.score_candidates(ids, candidates, 0)
    m4_k0, _ = rec_c.score_candidates(ids, candidates, 0)
    assert _token_logprobs(m4_k0) == _token_logprobs(m1_k0)

    cache_a, z0_a = rec_a._encode(ids)
    cache_b, z0_b = rec_b._encode(ids)
    z_a, pos_a = rec_a.latent_steps(z0_a, cache_a, ids.shape[1], 4)
    z_b, pos_b = rec_b.latent_steps(z0_b, cache_b, ids.shape[1], 4)
    assert pos_a == pos_b and torch.equal(z_a, z_b)


def test_paired_workspace_m4_zero_init_and_fullreset_are_exact():
    model, rec2 = build_paired_delta(workspace_slots=4)
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17]])
    candidates = ([19, 23], [29, 31, 37])
    cache, z0 = rec2._encode(ids)
    assert z0.shape == (1, 4, model.config.hidden_size)

    k0, _ = rec2.score_candidates(ids, candidates, 0)
    k4, report = rec2.score_candidates(ids, candidates, 4)
    assert _token_logprobs(k4) == _token_logprobs(k0)
    layer_calls = 2 * 4 * model.config.num_hidden_layers
    assert report.extra["recurrence_interval_layer_applications"] == layer_calls
    assert report.extra["recurrence_interval_token_layer_applications"] == \
        4 * layer_calls

    generator = torch.Generator().manual_seed(303)
    with torch.no_grad():
        rec2.clock.weight.normal_(std=0.05, generator=generator)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.05)
    trained_k0, _ = rec2.score_candidates(ids, candidates, 0)
    reset, _ = rec2.score_candidates(
        ids, candidates, 4,
        ablate={"reset_state": True, "reset_cache": True})
    assert _token_logprobs(reset) == _token_logprobs(trained_k0)


def test_paired_workspace_early_slot_causally_changes_summary_delta():
    _model, rec2 = build_paired_delta(workspace_slots=4)
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17]])
    generator = torch.Generator().manual_seed(304)
    with torch.no_grad():
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.1)
        cache_a, z0_a = rec2._encode(ids)
        cache_b, z0_b = rec2._encode(ids)
        cache_c, z0_c = rec2._encode(ids)
        cache_d, z0_d = rec2._encode(ids)
        future_perturbed = z0_d.clone()
        future_perturbed[:, -1, :] += 0.25
        proposal_c = rec2._run_workspace_layers(
            range(rec2.n_layers), z0_c, cache_c, ids.shape[1])
        proposal_d = rec2._run_workspace_layers(
            range(rec2.n_layers), future_perturbed, cache_d, ids.shape[1])
        assert torch.equal(proposal_c[:, :-1, :], proposal_d[:, :-1, :])

        perturbed = z0_b.clone()
        perturbed[:, 0, :] += 0.25
        out_a, _ = rec2.latent_steps(z0_a, cache_a, ids.shape[1], 1)
        out_b, _ = rec2.latent_steps(perturbed, cache_b, ids.shape[1], 1)
        delta_a = (out_a - z0_a)[:, -1:, :]
        delta_b = (out_b - perturbed)[:, -1:, :]
    assert not torch.equal(delta_a, delta_b)

    cache_grad, z0_grad = rec2._encode(ids)
    workspace = z0_grad.detach().requires_grad_(True)
    proposal = rec2._run_workspace_layers(
        range(rec2.n_layers), workspace, cache_grad, ids.shape[1], grad=True)
    input_grad = torch.autograd.grad(
        proposal[:, -1, :].sum(), workspace, retain_graph=False)[0]
    assert torch.isfinite(input_grad).all()
    assert torch.count_nonzero(input_grad[:, :-1, :]).item() > 0


def test_paired_delta_rejects_non_full_or_shared_adapter_modes():
    model, tok = build_tiny_hybrid()
    with pytest.raises(ValueError, match="full-decoder"):
        LocalizedRecurrence(
            model, tok, interval=INTERVAL, max_k=2,
            recurrence_only_lora=True, paired_delta=True)
    model, tok = build_tiny_hybrid()
    with pytest.raises(ValueError, match="recurrence_only_lora"):
        LocalizedRecurrence(
            model, tok, interval=(0, model.config.num_hidden_layers),
            max_k=2, paired_delta=True)
    for invalid in (True, 0, -1, 1.5):
        model, tok = build_tiny_hybrid()
        with pytest.raises(ValueError, match="positive integer"):
            LocalizedRecurrence(
                model, tok, interval=(0, model.config.num_hidden_layers),
                max_k=2, recurrence_only_lora=True, paired_delta=True,
                workspace_slots=invalid)
    model, tok = build_tiny_hybrid()
    with pytest.raises(ValueError, match="requires full paired_delta"):
        LocalizedRecurrence(
            model, tok, interval=(0, model.config.num_hidden_layers),
            max_k=2, recurrence_only_lora=True, workspace_slots=4)


def test_paired_trace_lambda_zero_is_exact_ce_path_without_trajectory(
        monkeypatch):
    _model, rec2 = build_paired_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11], [12, 13], [14, 15])
    direct = rec2.candidate_ce_loss_on_example(
        ids, candidates, gold_index=1, k_steps=4)

    original = rec2.latent_steps
    trajectory_requests = []

    def capture(*args, **kwargs):
        trajectory_requests.append(kwargs.get("capture_trajectory", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(rec2, "latent_steps", capture)
    faded = rec2.candidate_ce_trace_loss_on_example(
        ids, candidates, gold_index=1, k_steps=4,
        trace_targets=((1, [20, 21]), (3, [22])), trace_lambda=0.0)

    assert torch.equal(faded, direct)
    assert trajectory_requests == [False]


def test_paired_trace_alignment_cache_restore_and_loss_decomposition(
        monkeypatch):
    _model, rec2 = build_paired_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11], [12, 13, 14])
    trace_targets = ((1, [20, 21]), (3, [22]))
    generator = torch.Generator().manual_seed(184)
    with torch.no_grad():
        rec2.clock.weight.normal_(mean=0.0, std=0.01, generator=generator)
        for adapter in rec2.injected:
            adapter.lora_B.copy_(torch.randn(
                adapter.lora_B.shape, generator=generator) * 0.01)

        expected_cache, expected_z0 = rec2._encode(ids)
        _z, _pos, expected_trajectory = rec2.latent_steps(
            expected_z0, expected_cache, ids.shape[1], 4,
            capture_trajectory=True)
        expected_states = tuple(state.detach().clone()
                                for state in expected_trajectory)
        expected_z0 = expected_z0.detach().clone()
    assert any(not torch.equal(expected_states[0], state)
               for state in expected_states[1:])

    from latent_lab.backends import hf_qwen

    original_restore = hf_qwen.cache_restore
    restore_snapshots = []

    def capture_restore(cache, snapshot):
        restore_snapshots.append(id(snapshot))
        return original_restore(cache, snapshot)

    monkeypatch.setattr(hf_qwen, "cache_restore", capture_restore)
    original_score = rec2._score_candidate_tokens
    scored = []

    def capture_score(state, cache, pos, candidate_ids, *, grad=False,
                      latent_delta=None):
        result = original_score(
            state, cache, pos, candidate_ids, grad=grad,
            latent_delta=latent_delta)
        scored.append((
            state.detach().clone(), pos,
            latent_delta.detach().clone(), result[0]))
        return result

    monkeypatch.setattr(rec2, "_score_candidate_tokens", capture_score)
    loss = rec2.candidate_ce_trace_loss_on_example(
        ids, candidates, gold_index=1, k_steps=4,
        trace_targets=trace_targets, trace_lambda=0.3)

    candidate_count = len(candidates)
    assert len(scored) == candidate_count + len(trace_targets)
    for offset, (step_index, _target_ids) in enumerate(trace_targets):
        state, pos, delta, _token_logprobs = scored[candidate_count + offset]
        assert pos == ids.shape[1]
        expected_delta = expected_states[step_index - 1] - expected_z0
        expected_readout = expected_z0 + expected_delta
        assert torch.equal(state, expected_readout)
        assert torch.equal(delta, expected_delta)

    # Four logical paired steps restore the same prompt snapshot three times
    # each plus the loop's fail-safe final restore. Candidate scoring uses its
    # own snapshot; every trace target restores the original prompt snapshot.
    recurrence_restores = 3 * 4 + 1
    assert len(restore_snapshots) == \
        recurrence_restores + candidate_count + len(trace_targets)
    prompt_snapshot = restore_snapshots[0]
    assert restore_snapshots[:recurrence_restores] == \
        [prompt_snapshot] * recurrence_restores
    candidate_snapshot = restore_snapshots[recurrence_restores]
    assert candidate_snapshot != prompt_snapshot
    assert restore_snapshots[
        recurrence_restores:recurrence_restores + candidate_count] == \
        [candidate_snapshot] * candidate_count
    trace_prompt_snapshot = restore_snapshots[-1]
    assert trace_prompt_snapshot != candidate_snapshot
    assert restore_snapshots[-len(trace_targets):] == \
        [trace_prompt_snapshot] * len(trace_targets)

    candidate_loss = rec2.candidate_cross_entropy(
        [row[3].mean() for row in scored[:candidate_count]], gold_index=1)
    trace_loss = torch.stack(
        [-row[3].mean() for row in scored[candidate_count:]]).mean()
    assert torch.equal(loss, candidate_loss + 0.3 * trace_loss)


def test_paired_trace_loss_alone_reaches_b_and_clock_gradients():
    _model, rec2 = build_paired_delta()
    ids = torch.randint(0, 250, (1, 7))
    # A one-candidate CE is identically zero, so this backward signal comes
    # only from the visible trace target.
    loss = rec2.candidate_ce_trace_loss_on_example(
        ids, ([10, 11],), gold_index=0, k_steps=4,
        trace_targets=((1, [12, 13]),), trace_lambda=1.0)
    assert loss.requires_grad and torch.isfinite(loss) and loss.item() > 0
    loss.backward()

    assert any(adapter.lora_B.grad is not None
               and torch.count_nonzero(adapter.lora_B.grad).item() > 0
               for adapter in rec2.injected)
    assert rec2.clock.weight.grad is not None
    assert torch.count_nonzero(rec2.clock.weight.grad).item() > 0
    assert all(adapter.lora_A.grad is None
               or torch.count_nonzero(adapter.lora_A.grad).item() == 0
               for adapter in rec2.injected)


def test_paired_trace_rejects_final_gold_supervision():
    _model, rec2 = build_paired_delta()
    ids = torch.randint(0, 250, (1, 7))
    candidates = ([10, 11], [12, 13])
    with pytest.raises(ValueError, match="final/gold"):
        rec2.candidate_ce_trace_loss_on_example(
            ids, candidates, gold_index=0, k_steps=4,
            trace_targets=((1, [10, 11]),), trace_lambda=1.0)


def test_counterfactual_margin_score_invariances_and_identical_state_a_zero():
    p_zero = torch.tensor([-1.0, 0.2, -0.4, 0.7])
    p_own = p_zero + torch.tensor([1.3, -0.2, 0.1, -0.6])
    p_other = p_zero + torch.tensor([-0.4, 1.1, 0.2, -0.1])
    q_zero = torch.tensor([0.5, -0.7, 0.4, -0.2])
    q_own = q_zero + torch.tensor([-0.3, 1.4, -0.1, 0.2])
    q_other = q_zero + torch.tensor([1.0, -0.5, 0.2, -0.2])
    args = (p_own, p_zero, p_other, 0,
            q_own, q_zero, q_other, 1)
    terms = LocalizedRecurrence.counterfactual_margin_terms_from_scores(*args)
    loss = LocalizedRecurrence.counterfactual_margin_loss_from_scores(*args)

    # Frozen-model candidate preferences cancel candidate by candidate.
    p_constants = torch.tensor([7.0, -3.0, 4.0, 0.5])
    q_constants = torch.tensor([-2.0, 8.0, 1.5, -6.0])
    shifted = (
        p_own + p_constants, p_zero + p_constants,
        p_other + p_constants, 0,
        q_own + q_constants, q_zero + q_constants,
        q_other + q_constants, 1)
    assert torch.allclose(
        terms,
        LocalizedRecurrence.counterfactual_margin_terms_from_scores(*shifted))
    assert torch.allclose(
        loss,
        LocalizedRecurrence.counterfactual_margin_loss_from_scores(*shifted))

    # Candidate order is irrelevant when both gold indices move with it.
    permutation = torch.tensor([2, 0, 3, 1])
    p_index = int((permutation == 0).nonzero()[0])
    q_index = int((permutation == 1).nonzero()[0])
    permuted = (
        p_own[permutation], p_zero[permutation], p_other[permutation], p_index,
        q_own[permutation], q_zero[permutation], q_other[permutation], q_index)
    assert torch.allclose(
        terms,
        LocalizedRecurrence.counterfactual_margin_terms_from_scores(*permuted))
    assert torch.allclose(
        loss,
        LocalizedRecurrence.counterfactual_margin_loss_from_scores(*permuted))

    identical = LocalizedRecurrence.counterfactual_margin_terms_from_scores(
        p_own, p_zero, p_own, 0, q_own, q_zero, q_own, 1)
    assert torch.equal(identical[2:], torch.zeros(2))


def test_counterfactual_correct_own_delta_lowers_loss_and_hard_distractor_wins():
    zero = torch.zeros(3)
    p_good = torch.tensor([2.0, -1.0, -1.0])
    q_good = torch.tensor([-1.0, 2.0, -1.0])
    good_args = (p_good, zero, q_good, 0,
                 q_good, zero, p_good, 1)
    bad_args = (q_good, zero, q_good, 0,
                p_good, zero, p_good, 1)
    good_loss = LocalizedRecurrence.counterfactual_margin_loss_from_scores(
        *good_args)
    bad_loss = LocalizedRecurrence.counterfactual_margin_loss_from_scores(
        *bad_args)
    assert good_loss < bad_loss

    hard_distractor = torch.tensor([0.5, 0.6, -10.0])
    terms = LocalizedRecurrence.counterfactual_margin_terms_from_scores(
        hard_distractor, zero, q_good, 0,
        q_good, zero, p_good, 1)
    assert terms[0] < 0  # Gp uses logsumexp: the 0.6 distractor beats gold.
    rescued = LocalizedRecurrence.counterfactual_margin_loss_from_scores(
        torch.tensor([1.0, 0.6, -10.0]), zero, q_good, 0,
        q_good, zero, p_good, 1)
    penalized = LocalizedRecurrence.counterfactual_margin_loss_from_scores(
        hard_distractor, zero, q_good, 0,
        q_good, zero, p_good, 1)
    assert penalized > rescued


@pytest.mark.parametrize("workspace_slots", [1, 4])
def test_counterfactual_margin_first_backward_reaches_b_and_clock(
        monkeypatch, workspace_slots):
    _model, rec2 = build_paired_delta(workspace_slots=workspace_slots)
    prompt = torch.randint(0, 250, (1, 7))
    counterfactual = torch.randint(0, 250, (1, 8))
    candidates = ([10, 11], [12, 13], [14, 15])
    original_score = rec2._score_candidate_tokens
    readout_adapter_states = []

    def capture_score(*args, **kwargs):
        readout_adapter_states.append(
            tuple(adapter.enabled for adapter in rec2.injected))
        return original_score(*args, **kwargs)

    monkeypatch.setattr(rec2, "_score_candidate_tokens", capture_score)
    loss, terms = rec2.counterfactual_margin_loss_on_example(
        prompt, counterfactual, candidates, 0, 1, 4, return_terms=True)
    assert loss.requires_grad and torch.isfinite(loss)
    assert terms.shape == (4,) and not terms.requires_grad
    assert readout_adapter_states
    assert all(not any(states) for states in readout_adapter_states)
    loss.backward()

    assert any(adapter.lora_B.grad is not None
               and torch.count_nonzero(adapter.lora_B.grad).item() > 0
               for adapter in rec2.injected)
    assert rec2.clock.weight.grad is not None
    assert torch.count_nonzero(rec2.clock.weight.grad).item() > 0
    assert all(adapter.lora_A.grad is None
               or torch.count_nonzero(adapter.lora_A.grad).item() == 0
               for adapter in rec2.injected)

    with pytest.raises(ValueError, match="exactly K=4"):
        rec2.counterfactual_margin_loss_on_example(
            prompt, counterfactual, candidates, 0, 1, 0)


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
    # Autoregressive evidence records one output-head call per scored token.
    assert rec.guard.lm_head_calls - lm_before == sum(map(len, cands))
    assert rep.extra["candidate_token_counts"] == [2, 2, 2]
    assert rep.extra["candidate_raw_sum_logprobs"] == scores


def test_candidate_cross_entropy_tracks_margin_and_candidate_permutation():
    weak = LocalizedRecurrence.candidate_cross_entropy(
        [torch.tensor(0.0), torch.tensor(0.5), torch.tensor(-0.5)], 0)
    strong = LocalizedRecurrence.candidate_cross_entropy(
        [torch.tensor(1.0), torch.tensor(0.5), torch.tensor(-0.5)], 0)
    permuted = LocalizedRecurrence.candidate_cross_entropy(
        [torch.tensor(0.5), torch.tensor(-0.5), torch.tensor(1.0)], 2)
    assert strong < weak
    assert torch.equal(strong, permuted)


def test_candidate_ce_reaches_recurrence_only_lora_gradients():
    model, tok = build_tiny_hybrid()
    rec2 = LocalizedRecurrence(
        model, tok, interval=INTERVAL, max_k=2, lora_r=2,
        use_clock=False, recurrence_only_lora=True)
    ids = torch.randint(0, 250, (1, 6))
    candidates = ([10, 11], [12, 13], [14, 15])
    loss = rec2.candidate_ce_loss_on_example(
        ids, candidates, gold_index=1, k_steps=1)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    lora_gradients = [
        parameter.grad for adapter in rec2.injected
        for parameter in (adapter.lora_A, adapter.lora_B)
        if parameter.grad is not None
    ]
    assert lora_gradients
    assert all(torch.isfinite(gradient).all() for gradient in lora_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0
               for gradient in lora_gradients)
    assert all(not adapter.enabled for adapter in rec2.injected)
