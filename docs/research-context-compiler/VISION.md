# RCC Vision — Latent Cognition on Open-Weight Qwen

Date: 2026-08-22. Supersedes the "textual RCC as end-state" reading of earlier
plan documents.

## The goal (unchanged since the project's origin)

Research history, evidence, working memory and intermediate reasoning should
exist **predominantly as machine state** — vectors, hidden states, KV /
recurrent state — not as serialized natural language. Natural language appears
only:

1. at the user boundary;
2. in the final answer;
3. in tool-call arguments when a tool needs text;
4. in short operational telemetry;
5. in lossless raw evidence, available on demand.

Telemetry is *not* a reasoning trace, never feeds decisions, and is never fed
back to the model as working memory.

## Final target

**`Qwen/Qwen3.8-27B`** — the single release target and local product
benchmark. Runtime must not depend on closed APIs or unavailable hidden
states. Verified 2026-08-22: model card exists (Apache-2.0), Transformers has
a `qwen3_5` implementation module covering the hybrid architecture.

### Model ladder

| Role | Model |
|---|---|
| Release target / local product benchmark | `Qwen/Qwen3.8-27B` |
| Primary scientific proxy | `Qwen/Qwen3.5-4B` |
| Fast smoke/debug proxy | `Qwen/Qwen3.5-0.8B` (verified smallest open hybrid; ~1.75 GB bf16) |
| Mock/unit backend | deterministic tiny implementation, no model |

Qwen3.5-4B is used only because it reproduces the hybrid
`3 × Gated DeltaNet + 1 × Attention` layer pattern cheaply. Proxy success is
permission to scale, never proof of 27B success.

## What exists today (honest)

Current RCC is:

```
storage-lossless evidence store
+ lossy textual active view (stubs/RIR/1)
+ manual symbolic scratchpad
```

It is a useful safety/evidence substrate. It is **not latent cognition**.
There is no measured speedup, no measured behavioral fidelity, and no live
model has ever been run against compiled context.

## Terminology discipline

Three independent properties, never conflated under "lossless":

- **Storage-lossless**: original bytes recoverable byte-for-byte.
  Current status: implemented + unit-verified.
- **Access-lossless**: agent detects missing info, requests the right raw
  fragment, gets it in time. Status: only partially implemented (manual
  `expand()`; no model-directed access policy).
- **Behaviorally faithful**: compressed/latent system matches full-context
  baseline within tolerance on real tasks. Status: **NOT MEASURED**.

## New role of RIR/1

RIR/1 is no longer the target "language of thought". It becomes:
audit/debug representation, optional decoder input format, teacher-side
intermediate representation, emergency textual fallback, and reproducible
check format. We do not develop RIR further unless evidence/benchmark work
requires it.

## Chosen architecture

**Proof-Carrying Localized Latent RCC** — six planes (evidence,
model-state ABI, fixed latent workspace, localized recurrence over a selected
layer interval, discrete controller, telemetry-only observability). See
ADR-001 and LATENT_ROADMAP.md for alternatives considered and rejected.

## Deployment environment truth

Primary local machine at time of writing: Apple Silicon, 16 GB unified
memory (measured M1 Pro; original plan assumed M4/32 GB). Consequences:

- small-model inference and mock experiments: local;
- scientific training of the proxy: local-if-feasible or CUDA;
- 27B adapter training: reproducible remote CUDA config (not yet provisioned);
- final quantized inference benchmark: local Mac.

PyTorch/Transformers is the research backend; MLX is the mandatory final
local benchmark backend (blockers tracked in BLOCKERS.md).
