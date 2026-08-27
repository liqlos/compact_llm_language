# ADR-001: Open-Weight Latent RCC on Qwen3.8

Status: SUPERSEDED_BY_NEGATIVE_PROXY_RESULT
Date: 2026-08-22
Supersedes: implicit direction of IMPLEMENTATION_PLAN.md phases 5–7 (textual
machine dialects as end state).

> **Final addendum (2026-08-27):** the later bounded Qwen3.5-2B experiments
> produced no replicated latent-reasoning gain, so proxy failure stopped
> scaling and superseded this direction. See
> [`LATENT_EXPERIMENT_STATUS.md`](LATENT_EXPERIMENT_STATUS.md). Historical
> behavioral-v2 metrics remain invalidated.

## Context

RCC's founding goal is machine-native cognition: research state as tensors,
not serialized prose. Work to date built a storage-lossless evidence substrate
and textual views. Two strategic questions had to be settled:

1. Closed-API vs open-weight target.
2. Textual compression as end-state vs latent reasoning as end-state.

## Decision

1. Single final target: **`Qwen/Qwen3.8-27B`** (open weights, Apache-2.0).
   No runtime dependency on closed provider APIs or hidden-state-free
   providers. Development proxies: Qwen3.5-4B (science), Qwen3.5-0.8B
   (smoke), deterministic mock (unit).
2. Architecture of record: **Proof-Carrying Localized Latent RCC**
   (evidence plane + versioned model-state ABI + fixed latent workspace +
   localized recurrent interval + discrete controller + telemetry plane).
3. Existing `rcc/` package is retained unchanged as the evidence /
   compatibility layer; new work lives in `latent_lab/` with optional heavy
   dependencies.

## Alternatives considered and why rejected

| Alternative | Why not the mainline |
|---|---|
| Continue closed-API textual RCC | no hidden states, no latent control, provider lock-in; contradicts founding goal |
| Full RIR/1 dialect engineering | RIR is an audit/debug/fallback representation; optimizing it does not approach latent cognition |
| Coconut-style full-decoder recurrence only | mandatory baseline, but per-step full-decoder cost caps speedup; keep as baseline E |
| Context-only latent memory first (ICAE/Gist/LCC) | compresses context but adds no reasoning loop; deferred until after the reasoning gate (architecture F) |
| Direct DeltaNet-state writes (G) | high-risk/high-reward spike; kept strictly separate until independently validated |

## Consequences

- All quality/speed claims must come from wall-clock benchmarks on Qwen
  runtimes against matched baselines (see EVALUATION_PROTOCOL.md).
- Proxy results gate 27B work; failure on proxy stops scaling.
- Maturity vocabulary changes to HYPOTHESIS / SCAFFOLDED / UNIT_VERIFIED /
  MODEL_VERIFIED / REPLICATED / REJECTED_BY_EVIDENCE / BLOCKED.

## Primary sources verified 2026-08-22

- Qwen/Qwen3.8-27B model card (HF, Apache-2.0); Qwen/Qwen3.5-0.8B config:
  24 layers = 6 × (3 × Gated DeltaNet → FFN; 1 × gated Attention → FFN).
- `transformers` module `qwen3_5` (`Qwen3_5ForCausalLM`,
  `Qwen3_5GatedDeltaNet`, `Qwen3_5DynamicCache` with per-layer
  `conv_states[i]` / `recurrent_states[i]` writable tensors).
- Coconut arXiv:2412.06769; CODI arXiv:2502.21074; Switch arXiv:2606.13106
  (`<swi>`/`</swi>` boundary tokens); Penelope arXiv:2607.25915 (localized
  latent recurrence); ICAE arXiv:2307.06945; Gist arXiv:2304.08467;
  LCC arXiv:2602.21221; Mem-W arXiv:2605.09317 (GUI-agent scope);
  Gated DeltaNet arXiv:2412.06464; Inner Loop Inference arXiv:2602.14759;
  Huginn recurrent depth arXiv:2502.05171.
