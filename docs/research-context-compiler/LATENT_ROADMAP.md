# Latent Roadmap — maturity-tracked

> **Current real-model status:** see
> [`LATENT_EXPERIMENT_STATUS.md`](LATENT_EXPERIMENT_STATUS.md). The direct owner
> authorization on 2026-08-27 permits up to approximately USD 20 of paid
> compute for the current 2B investigation. Historical R1 artifacts remain
> invalidated; that authorization does not rehabilitate their scorer or
> checkpoint provenance.

Status vocabulary: `HYPOTHESIS` `SCAFFOLDED` `UNIT_VERIFIED` `MODEL_VERIFIED`
`REPLICATED` `REJECTED_BY_EVIDENCE` `BLOCKED` (plus NOT_STARTED / NOT MEASURED).
A file or class existing is never evidence of maturity by itself.

Updated: 2026-08-27 (R1 evidence invalidation and local runtime contracts).
Measurements remain historical snapshots unless a row names a current receipt.

## Current status table

| Capability | Status | Evidence |
|---|---|---|
| Raw byte storage (content-addressed, hash-verified) | UNIT_VERIFIED | tamper/roundtrip regression coverage |
| Observation masking → stubs + deterministic expansion | UNIT_VERIFIED | unit + property + benchmark tests; P0 wrapper-escape/traversal/corruption fixed with regression tests |
| Manual symbolic scratch RIR/1 | SCAFFOLDED | unit tests; no live-model comprehension check |
| Journal checkpoint/restore | UNIT_VERIFIED | incl. dedup-index restore regression test |
| Router / dictenc / break-even gate | SCAFFOLDED (experimental) | unit tests only; no measured task benefit |
| Behavioral fidelity of compiled context | NOT MEASURED | deterministic fixture only; no real-provider run exists |
| Model-directed expansion (access-lossless) | HYPOTHESIS | design only |
| Latent pipeline plumbing (latent_lab) | UNIT_VERIFIED (local tiny/mock) | hidden-state-chain BPTT with detached cache recurrence; unsupported `grad_checkpoint` removed; scoped no-decode guard; same-adapter K=0/1/2/4/8; autoregressive full-decoder reference |
| Toy recurrence trainer (T1 analog) | UNIT_VERIFIED | loss 7.58→0.016, acc 100% vs 6.25% under loop ablation ⇒ latent path causally used at toy scale |
| Historical Qwen hybrid runtime probe | HISTORICAL_UNBOUND | retained 0.8B state-probe JSON is runtime-only evidence, not reasoning quality or current scorer evidence |
| MLX soft-embedding path | MEASURED / PARTIALLY BLOCKED | exact-vocab embeds bit-exact & fast; off-manifold inputs slow ~3.5–5×; hybrid prefix-cache trim unsupported |
| Localized recurrence quality/speedup | INVALIDATED / VALID_EXPERIMENT_PENDING | behavioral-v2 derived-only outputs cannot be rescored; no current model result exists |
| Qwen3.8-27B speedup ≥2× | NOT MEASURED | no current scaling permission; valid small behavioral-v3 experiment pending |

## Stages

## INVALIDATED HISTORICAL TEXT — preserved, not current evidence

> The historical paragraph below is preserved to show what was claimed before
> the R1 audit; it must not be quoted as a current result. Behavioral-v2
> `obj_track` leaked final state into “Initial situation”; the 50 latent evals
> lack raw candidate scores and used the candidate-index-zero scorer; candidate
> counts vary; K comparisons used separately trained adapters. The stored 2B
> textual 0/112 flags do not establish a capability limit. Conversely, stored
> 4B direct/no-thinking flags are 97/112 ID and 84/112 OOD versus native
> thinking 1/112 and 0/112, but all are preview-only and non-rescorable.

### T0 — Baselines (INVALIDATED HISTORICAL CLAIM, 2026-08-23, 2B)
Behavioral suite v2: 7 procedural state-tracking families, parse-back
verified answers, balanced candidates, leakage audit. Textual baselines on
Qwen3.5-2B: ALL 0.0% accuracy (direct/thinking/capped, incl. 1024-token
budget; up to 100% NON_TERMINATION from re-reading loops) =>
MODEL_CAPABILITY_LIMIT for textual reasoning at this scale.

### T1/T2 — Latent recurrence @2B (INVALIDATED HISTORICAL CLAIM, 2026-08-23)
Frozen backbone + interval LoRA + zero-init step clock; guarded loop (zero
lm_head/tokenizer calls, unit-tested). Attempt-1 (constant lr): E(K=4)
test-ID 0.4375/0.5179/0.4286 (mean .461) vs K=0 control .375 vs full-decoder
.304; OOD depth5-8 mean .399 (s1 .491). K dose-response peaks at trained
depth. Ablations seed-dependent: s0 strongly causal (bypass −26pp, noise
−24pp), s1 weakly loop-reliant. Attempt-2 (cosine 5e-5, paired seeds):
E>F on ID both seeds (+5..6pp), F>E on OOD (+12..14pp); schedule variance
dominates. VERDICT: above-chance learning replicates across seeds;
advantage over single-pass control is NOT yet robust => conditional GO to
4B under MODEL_CAPABILITY_LIMIT rule with paired-seed protocol mandatory.

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

Execution environment (owner decision 2026-08-24): any serious 4B/27B
training or CUDA matrix would use rented hardware, but only when both the
fresh fail-closed gate and `artifacts/milestone_r1_verdict.json` explicitly
authorize it. Current state is `PAID_SPEND_NOT_AUTHORIZED`; the local laptop
runs unit/smoke tests, artifact validation and orchestration only (VISION.md).

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
