# Latent reasoning experiment status

Finalized: 2026-08-27.

## Final verdict

`NO_PROVEN_LATENT_REASONING_GAIN`

The repository contains a genuine hidden-state-only recurrence runtime over
`Qwen/Qwen3.5-2B`, but the completed experiments did not show a replicated
reasoning improvement over the adapter-free `K=0` control. No efficiency gain
was measured; `K=4` executes more decoder work than `K=0`.

The project is archived. This rejects the tested recurrence, readout, and
training recipes at this scale and budget. It does not claim that latent
reasoning in general is impossible.

## Validated mechanism

- Seed initialization occurs before model construction, LoRA construction, and
  data shuffling.
- The recurrent path performs hidden-state and cache operations without
  tokenizer, generation, or vocabulary-head calls.
- `K=0` is adapter-free under recurrence-only LoRA.
- Full reset restores state and prompt cache and is prediction-exact to `K=0`.
- Qwen3.5 LoRA targets resolve against the actual attention and GatedDeltaNet
  projection modules.
- State interventions (zero, norm-matched noise, cross-prompt swap) are
  independent of prompt/cache interventions.
- CUDA memory is reported separately as peak allocated and peak reserved;
  process RSS is a third, non-additive measurement.

These facts validate the experimental mechanism, not useful reasoning.

## Completed real-model results

The primary test-ID split contains 168 examples. All runs use
`Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`.

| Mechanism | Validation | Test-ID controls | Verdict |
|---|---:|---:|---|
| Paired recurrence + visible-trace curriculum, seed 0 | selected checkpoint did not improve clean accuracy | `K0=38`, `K1=38`, `K4=36`, full reset `38` | Rejected. Margin movement was calibration, not reasoning. |
| Oracle safe visible-trace diagnostic | — | base `14/96`, oracle trace `10/96` | Rejected. Even the correct visible trace hurt this readout. |
| One-slot counterfactual margin, seed 0 | `K0=51/224`; selected step 280 `53/224` | `K0=38`, `K1=38`, `K4=40`, reset `38`, noise `38` | Rejected as shortcut/null. Mean `K4-K0` gold-margin change `-0.094`; 6 gains/4 losses; gains dominated by `stack_queue`. |
| One-slot counterfactual margin, seed 1 | step 140 `50/224`; selected step 280 `55/224` | `K0=38`, `K1=36`, `K4=38`, reset `38` | Replicated null. Mean `K4-K0` gold-margin change `-0.073`; no accuracy gain. |
| Four-slot causal workspace, seed 0 | step 140 `52/224`; selected step 280 `53/224`; adapter-free `K0=51/224` | not run after validation missed the promotion gate | Rejected before test spend: `+2/224 = +0.9 pp`, below the preregistered `+5 pp` gate. |

The seed-0 one-slot `38→40` movement is not a breakthrough: its confidence
interval crossed zero, its aggregate ranking margin worsened, and the effect
disappeared on replication.

## Four-slot closing-run receipt

- Run ID: `latent-train-8ef09aafaceedb3d2d6f7076`
- Recipe: full decoder, paired delta, counterfactual margin, `K=4`,
  `workspace_slots=4`, LoRA rank 8, seed 0, 280 steps
- Selected checkpoint: step 280
- Validation accuracy: `0.23660714285714285 = 53/224`
- Wall time: `2193.7 s`
- CUDA peak allocated: `3759.5 MiB`
- CUDA peak reserved: `7828.0 MiB`
- Process peak RSS: `4580.0 MiB`
- Checkpoint file SHA-256:
  `33ff92f7c86c0ce41b40e384f9f81a81123a0cab755f65006ceda174525cd2d3`
- Canonical checkpoint-content digest:
  `16a025478848e07fa9e3d588561a200b8b9c66a0f85bcb74dcce0a1ddb6dd946`
- Train-report SHA-256:
  `837a8c41a41e9d48f0f5be1a150f9e9e5e85b41589587735f299337fc7fa9d9a`

The downloaded local and rented-machine hashes matched before instance
destruction. The ignored local run directory is
`.rcc_work/pilot-2b-m4-k4-paired-cfmargin-s0-280-v3/`; model/adapter files
are intentionally not committed.

The closing RTX 5090 instance cost approximately USD 2.51 across setup,
failed transport attempts, and the completed run. Instance `48882608` was
destroyed after artifact verification; no attached volume remained.

## Why this branch stopped

1. The old objective allowed a last-event shortcut; early-event controls
   exposed it.
2. Visible-trace supervision did not help this 2B readout, even when the trace
   was oracle-correct.
3. One-slot counterfactual training produced a null on seed replication.
4. Four causal slots produced only `+0.9 pp` on validation.
5. Published positive systems generally use much stronger curricula,
   distillation, auxiliary per-step decoders, or recurrent pretraining. A
   credible reproduction would be a different, substantially larger project.

An unvalidated decoder-mediated bridge was intentionally excluded from public
`main`: its focused tests passed, but its full suite and real-model experiment
were never completed.

## External evidence

- [COCONUT](https://arxiv.org/abs/2412.06769): hidden-state recurrence with a
  multistage visible-to-latent curriculum.
- [CODI](https://arxiv.org/abs/2502.21074): continuous-thought
  self-distillation.
- [SIM-CoT](https://arxiv.org/abs/2509.20317): training-only per-step decoder
  improves latent-step supervision.
- [Huginn](https://arxiv.org/abs/2502.05171): recurrent-depth pretraining from
  scratch.
- [Dilgren & Wiegreffe, COLM 2026](https://arxiv.org/abs/2604.04902):
  matched-data controls remove most latent/no-CoT differences on the logical
  tasks studied.
- [Aswal et al., 2026](https://arxiv.org/abs/2606.12689): causal interventions
  show task-dependent, sometimes unused latent steps.
- [Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170): soft-token
  training can regularize, while hard-token inference remains stronger.

## Code evidence anchors

- `448bf33` — visible-trace curriculum.
- `ccbed14` — counterfactual-margin objective and attribution terms.
- `366542d` — causal multi-slot workspace.
- `e7ea8af` — corrected trace-readout assertion.
- `7fdd300` — workspace identity bound into the canonical recipe.
- `fa77c78` — early-event causal probe.
