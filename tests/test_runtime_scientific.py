"""Scientific-contract regressions for the localized recurrence runtime."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from latent_lab.backends.localized import (
    AmbiguousTopTie,
    LatentLoopViolation,
    LocalizedRecurrence,
)


def _tiny_model(seed: int = 31, layers: int = 4):
    torch.manual_seed(seed)
    config = transformers.Qwen3_5TextConfig(
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=24,
        vocab_size=128,
        max_position_embeddings=256,
        layer_types=[
            "linear_attention" if index % 2 == 0 else "full_attention"
            for index in range(layers)
        ],
    )
    return transformers.Qwen3_5ForCausalLM(config).eval()


def _cache_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _cache_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _cache_tensors(nested)


def _all_cache_tensors(cache):
    for layer in cache.layers:
        for attribute in ("conv_states", "recurrent_states", "keys", "values"):
            value = getattr(layer, attribute, None)
            if value is not None:
                yield from _cache_tensors(value)


def test_full_decoder_candidate_matches_independent_incremental_oracle():
    model = _tiny_model()
    layers = model.config.num_hidden_layers
    runtime = LocalizedRecurrence(
        model, interval=(0, layers), max_k=4, use_clock=False, lora_r=2)
    prompt = torch.tensor([[3, 5, 7, 11]])
    candidate = [13, 17, 19, 23]

    details, report = runtime.score_candidates(prompt, [candidate], k_steps=2)

    # Independent answer scorer: recurrence establishes the state/cache and a
    # fixed carrier.  Each preceding answer token traverses the full decoder,
    # with the carrier added at the interval output-boundary.  It deliberately
    # does not call the runtime scoring helper.
    with torch.no_grad():
        cache, state = runtime._encode(prompt)
        state, position = runtime.latent_steps(
            state, cache, prompt.shape[1], 2)
        expected = []
        logits = model.lm_head(model.model.norm(state))
        expected.append(float(torch.log_softmax(logits.float(), -1)[0, 0, candidate[0]]))
        for previous, target in zip(candidate, candidate[1:]):
            embedded = model.model.embed_tokens(torch.tensor([[previous]]))
            position_rows = torch.full(
                (4, 1, 1), position, device=embedded.device, dtype=torch.long)
            rotary = model.model.rotary_emb(embedded, position_rows[1:])
            hidden = embedded
            for layer in model.model.layers:
                hidden = layer(
                    hidden,
                    position_embeddings=rotary,
                    attention_mask=None,
                    position_ids=position_rows[0],
                    past_key_values=cache,
                    use_cache=True,
                )
            hidden = hidden + state
            position += 1
            logits = model.lm_head(model.model.norm(hidden[:, -1:, :]))
            expected.append(float(torch.log_softmax(logits.float(), -1)[0, 0, target]))

    assert details[0].token_logprobs == pytest.approx(expected, abs=0.0, rel=0.0)
    assert report.extra["candidate_full_decoder_layer_applications"] == (
        layers * (len(candidate) - 1)
    )


def test_hidden_chain_bptt_is_live_while_cache_recurrence_is_detached():
    model = _tiny_model(seed=37)
    runtime = LocalizedRecurrence(model, interval=(1, 3), max_k=4, lora_r=2)
    prompt = torch.tensor([[2, 3, 5, 7]])
    cache, state = runtime._encode(prompt, grad=True)
    state.retain_grad()
    recurrent, _ = runtime.latent_steps(
        state, cache, prompt.shape[1], 2, grad=True)
    loss = recurrent.float().square().mean()
    loss.backward()

    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.count_nonzero(state.grad).item() > 0
    cache_tensors = list(_all_cache_tensors(cache))
    assert cache_tensors
    assert all(not tensor.requires_grad and tensor.grad_fn is None
               for tensor in cache_tensors)
    trainable_grads = [
        parameter.grad for parameter in runtime.trainable_parameters()
        if parameter.grad is not None
    ]
    assert trainable_grads
    assert all(torch.isfinite(gradient).all() for gradient in trainable_grads)
    assert any(torch.count_nonzero(gradient).item() > 0
               for gradient in trainable_grads)


def test_later_token_only_loss_has_live_carrier_and_recurrence_lora_gradient():
    runtime = LocalizedRecurrence(
        _tiny_model(seed=39), interval=(1, 3), max_k=2, lora_r=2,
        use_clock=False, recurrence_only_lora=True)
    prompt = torch.tensor([[2, 3, 5, 7]])
    answer = torch.tensor([[11, 13, 17]])

    cache, state = runtime._encode(prompt, grad=True)
    carrier, position = runtime.latent_steps(
        state, cache, prompt.shape[1], 1, grad=True)
    carrier.retain_grad()
    token_logprobs, _ = runtime._score_candidate_tokens(
        carrier, cache, position, answer, grad=True)
    later_token_loss = -token_logprobs[1:].mean()

    assert later_token_loss.requires_grad
    later_token_loss.backward()
    assert carrier.grad is not None
    assert torch.isfinite(carrier.grad).all()
    assert torch.count_nonzero(carrier.grad).item() > 0
    lora_gradients = [
        parameter.grad
        for adapter in runtime.injected
        for parameter in (adapter.lora_A, adapter.lora_B)
        if parameter.grad is not None
    ]
    assert lora_gradients
    assert all(torch.isfinite(gradient).all() for gradient in lora_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0
               for gradient in lora_gradients)


def test_gold_loss_has_finite_gradients_for_multitoken_answer():
    runtime = LocalizedRecurrence(
        _tiny_model(seed=41), interval=(1, 3), max_k=4, lora_r=2)
    loss = runtime.loss_on_example(
        torch.tensor([[3, 4, 5, 6]]),
        torch.tensor([[7, 8, 9]]),
        k_steps=2,
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in runtime.trainable_parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_one_adapter_supports_fixed_k_grid_and_raw_compute_evidence():
    runtime = LocalizedRecurrence(
        _tiny_model(seed=43), interval=(1, 3), max_k=8, lora_r=2)
    adapter_parameter_ids = tuple(id(p) for p in runtime.trainable_parameters())
    prompt = torch.tensor([[2, 4, 6]])
    candidates = [[8], [10, 12, 14]]

    for k_steps in (0, 1, 2, 4, 8):
        order, raw_sums, report = runtime.rank_candidates(
            prompt, candidates, k_steps)
        assert tuple(id(p) for p in runtime.trainable_parameters()) == (
            adapter_parameter_ids)
        assert sorted(order) == [0, 1]
        assert all(math.isfinite(score) for score in raw_sums)
        token_logprobs = report.extra["candidate_token_logprobs"]
        assert list(map(len, token_logprobs)) == [1, 3]
        assert raw_sums == [sum(values) for values in token_logprobs]
        assert report.extra["candidate_token_counts"] == [1, 3]
        assert report.extra["lm_head_calls"] == 4
        assert report.extra["tokenizer_calls"] == 0
        assert report.extra["generate_calls"] == 0
        assert report.extra["successful_candidates"] == 2
        assert report.extra["primary_score_definition"] == (
            "mean_candidate_token_logprob_v1")
        assert report.extra["total_layer_token_applications"] == (
            report.layer_applications)

    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        runtime.rank_candidates(prompt, candidates, 9)


def test_exact_primary_top_tie_fails_closed_without_index_break():
    runtime = LocalizedRecurrence(
        _tiny_model(seed=45), interval=(1, 3), max_k=2, lora_r=2)
    with torch.no_grad():
        runtime.model.get_output_embeddings().weight.zero_()
    with pytest.raises(AmbiguousTopTie, match="refusing index tie-break") as caught:
        runtime.rank_candidates(
            torch.tensor([[2, 4, 6]]), [[8], [10]], k_steps=0)
    assert len(caught.value.candidate_details) == 2
    assert caught.value.details["candidate_token_logprobs"]
    assert caught.value.details["primary_score_definition"] == (
        "mean_candidate_token_logprob_v1")
    assert caught.value.details["compute"]["successful_task"] is True
    assert caught.value.raw_sums == tuple(
        caught.value.details["candidate_raw_sum_logprobs"])


def test_fake_gradient_checkpointing_flag_is_absent_and_rejected():
    model = _tiny_model(seed=47)
    with pytest.raises(TypeError, match="grad_checkpoint"):
        LocalizedRecurrence(
            model, interval=(1, 3), max_k=4, grad_checkpoint=True)

    runtime = LocalizedRecurrence(
        _tiny_model(seed=47), interval=(1, 3), max_k=8)
    assert not hasattr(runtime, "grad_checkpoint")
    contract = runtime.runtime_contract()
    assert contract["gradient_checkpointing"] == "UNSUPPORTED_ABSENT"
    assert contract["training_gradient_semantics"] == (
        "hidden_state_chain_bptt_with_detached_cache_recurrence")
    assert contract["generic_state_controller_abi"] == (
        "SCAFFOLD_NOT_EVIDENCE_PATH")
    assert [contract["same_adapter_supported_k"][index]
            for index in (0, 1, 2, 4, 8)] == [0, 1, 2, 4, 8]


class _TokenizerProbe:
    def __call__(self, *args, **kwargs):
        return {"input_ids": [1]}

    def encode(self, *args, **kwargs):
        return [1]

    def decode(self, *args, **kwargs):
        return "x"

    def batch_decode(self, *args, **kwargs):
        return ["x"]

    def apply_chat_template(self, *args, **kwargs):
        return "x"


@pytest.mark.parametrize(
    "operation",
    ["__call__", "encode", "decode", "batch_decode", "apply_chat_template",
     "generate", "output_head"],
)
def test_no_decode_guard_fails_for_every_forbidden_path_and_restores(operation):
    tokenizer = _TokenizerProbe()
    runtime = LocalizedRecurrence(
        _tiny_model(seed=53), tokenizer, interval=(1, 3), max_k=2, lora_r=2)
    tokenizer_type = type(tokenizer)
    before = {name: tokenizer_type.__dict__[name]
              for name in runtime.guard.TOKENIZER_METHODS}

    def invoke():
        if operation == "__call__":
            tokenizer("prompt")
        elif operation in runtime.guard.TOKENIZER_METHODS:
            getattr(tokenizer, operation)("prompt")
        elif operation == "generate":
            runtime.model.generate()
        else:
            hidden = torch.zeros(1, 1, runtime.config.hidden_size)
            runtime.model.get_output_embeddings()(hidden)

    with pytest.raises(LatentLoopViolation):
        with runtime.guard.window():
            invoke()

    assert {name: tokenizer_type.__dict__[name]
            for name in runtime.guard.TOKENIZER_METHODS} == before
    assert tokenizer.encode("outside") == [1]


def test_latent_loop_guard_catches_tokenizer_batch_decode(monkeypatch):
    tokenizer = _TokenizerProbe()
    runtime = LocalizedRecurrence(
        _tiny_model(seed=59), tokenizer, interval=(1, 3), max_k=2, lora_r=2)
    target = runtime.base.layers[1]
    original = target.forward

    def forbidden(hidden_states, **kwargs):
        tokenizer.batch_decode([[1]])
        return original(hidden_states, **kwargs)

    monkeypatch.setattr(target, "forward", forbidden)
    prompt = torch.tensor([[2, 3, 5]])
    cache, state = runtime._encode(prompt)
    with pytest.raises(LatentLoopViolation, match="batch_decode"):
        runtime.latent_steps(state, cache, prompt.shape[1], 1)
