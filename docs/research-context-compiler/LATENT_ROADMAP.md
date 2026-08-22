# Latent Roadmap — maturity-tracked

Status vocabulary: `HYPOTHESIS` `SCAFFOLDED` `UNIT_VERIFIED` `MODEL_VERIFIED`
`REPLICATED` `REJECTED_BY_EVIDENCE` `BLOCKED` (plus NOT_STARTED / NOT MEASURED).
A file or class existing is never evidence of maturity by itself.

Updated: 2026-08-22 (environment truth session).

## Current status table

| Capability | Status | Evidence |
|---|---|---|
| Raw byte storage (content-addressed, hash-verified) | UNIT_VERIFIED | 93 tests incl. tamper/roundtrip; `tests/test_store.py` |
| Observation masking → stubs + deterministic expansion | UNIT_VERIFIED | unit + property + benchmark tests |
| Manual symbolic scratch RIR/1 | SCAFFOLDED | unit tests; no live-model comprehension check |
| Journal checkpoint/restore | UNIT_VERIFIED | `tests/test_journal.py` |
| Router / dictenc / break-even gate | SCAFFOLDED (experimental) | unit tests only; no measured task benefit |
| Behavioral fidelity of compiled context | NOT MEASURED | no live-model run exists |
| Model-directed expansion (access-lossless) | HYPOTHESIS | design only |
| Latent reasoning loop | NOT_STARTED → SCAFFOLDED this session (`latent_lab` mock) | mock unit tests; no model run yet |
| Real Qwen hybrid runtime control (state probe) | see BLOCKERS.md / probe report | pending first real run |
| Qwen3.8-27B speedup ≥2× | NOT MEASURED | requires proxy gate first |

## Stages

### T0 — Baselines (next after state probe)
Reproducible visible-CoT traces, direct answers, labels, latency,
layer-call counts, cache sizes on open-weight Qwen. No closed teacher API.

### T1 — Full-decoder latent baseline (Coconut-like)
Prove: hidden state returned without tokenization; latent state causally
influences answers; pipeline trains on a small task. Gate to T2.

### T2 — Localized recurrence
Boundary memory/readout slots; selected layer interval; cache-compatible
repeated execution; fixed depth K; LoRA-scale adapters; backbone frozen.

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
