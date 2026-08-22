# Latent Roadmap — maturity-tracked

Status vocabulary: `HYPOTHESIS` `SCAFFOLDED` `UNIT_VERIFIED` `MODEL_VERIFIED`
`REPLICATED` `REJECTED_BY_EVIDENCE` `BLOCKED` (plus NOT_STARTED / NOT MEASURED).
A file or class existing is never evidence of maturity by itself.

Updated: 2026-08-22 (state probe + MLX blocker probe measured).

## Current status table

| Capability | Status | Evidence |
|---|---|---|
| Raw byte storage (content-addressed, hash-verified) | UNIT_VERIFIED | 102+ tests incl. tamper/roundtrip |
| Observation masking → stubs + deterministic expansion | UNIT_VERIFIED | unit + property + benchmark tests; P0 wrapper-escape/traversal/corruption fixed with regression tests |
| Manual symbolic scratch RIR/1 | SCAFFOLDED | unit tests; no live-model comprehension check |
| Journal checkpoint/restore | UNIT_VERIFIED | incl. dedup-index restore regression test |
| Router / dictenc / break-even gate | SCAFFOLDED (experimental) | unit tests only; no measured task benefit |
| Behavioral fidelity of compiled context | NOT MEASURED | no live-model run exists |
| Model-directed expansion (access-lossless) | HYPOTHESIS | design only |
| Latent pipeline plumbing (latent_lab) | UNIT_VERIFIED (mock) | 39+ tests: no-decode loop contract, K causality, ablations, provenance verifier |
| Toy recurrence trainer (T1 analog) | UNIT_VERIFIED | loss 7.58→0.016, acc 100% vs 6.25% under loop ablation ⇒ latent path causally used at toy scale |
| **Real Qwen hybrid runtime control** | MODEL_VERIFIED (Qwen3.5-0.8B proxy) | state probe JSON: cache snapshot/restore exact, inputs_embeds ok, K=3 recurrence with 0 lm_head calls, 16.9 ms/step MPS fp16 |
| MLX soft-embedding path | MEASURED / PARTIALLY BLOCKED | exact-vocab embeds bit-exact & fast; off-manifold inputs slow ~3.5–5×; hybrid prefix-cache trim unsupported |
| Localized recurrence quality/speedup | NOT MEASURED | requires T1/T2 on proxy |
| Qwen3.8-27B speedup ≥2× | NOT MEASURED | gated by proxy gate below |

## Stages

### T0 — Baselines (NEXT)
Reproducible visible-CoT traces, direct answers, labels, latency,
layer-call counts, cache sizes on Qwen3.5-0.8B/4B via the proven control
points (see B8). No closed teacher API.

### T1 — Full-decoder latent baseline (Coconut-like)
Plumbing already proven at toy scale (`latent_lab/train`, loss ↓ + causal
ablation). Remaining: same proof with the real 0.8B checkpoint — hidden
state fed back across steps, causal influence on answers, small-task
training. Gate to T2.

### T2 — Localized recurrence
Boundary memory/readout slots; selected layer interval; cache-compatible
repeated execution; fixed depth K; LoRA-scale adapters; backbone frozen.
Interval candidates: early/middle/output-side/full (implemented in
`latent_lab.intervals`).

### T3 — Visible-to-latent curriculum
Progressively replace visible reasoning spans. Log collapse cases.

### T4 — Self-distillation (CODI-like)
Teacher sees explicit CoT; student uses latent steps; answer loss primary;
representation-alignment loss secondary.

### T5 — Adaptive control
Learned halt; CONTINUE/ACT/ANSWER; correctness-gated latency reward.
RL (GRPO-style) only after supervised pipeline proves latent use.

## Memory strategy (after reasoning gate)

Online working memory: learned incremental compressor over expired segments,
fixed latent block + evidence refs, raw fallback, invalidation, conflict and
temporal-update checks. Stable project snapshot: separate expensive-once LCC
path. Never instance-compilation as working memory.

## Go/no-go gates

Proxy gate (before any 27B work): ≥1.5× end-to-end wall-clock speedup vs
native visible CoT on reasoning-heavy deterministic tasks; ≤2 pp absolute
quality loss; causal latent ablation materially worsens results; no hidden
textual CoT; reproducible manifest. Stop-scaling rule: >5 pp quality loss or
slower recurrence after two architecture attempts ⇒ REJECTED_BY_EVIDENCE or
new ADR.

27B target gate: ≥2× wall-clock per successful task, ≤2 pp quality loss,
fixed runtime, not explained by shorter answers/skipped reasoning.

Latent-memory gate: ≥4× less replayed textual context; ≤2 pp loss; exact-fact
raw fallback works; conflicts/temporal updates preserved; incremental update
proven. Agent-loop gate: baseline-equal completion, better wall-clock/task,
correct tool actions, claims linked to raw evidence, telemetry ≠ cognition.
