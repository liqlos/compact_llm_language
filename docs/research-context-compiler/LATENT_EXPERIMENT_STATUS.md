# Latent reasoning experiment status

Updated: 2026-08-27. This is the short human-readable authority for the
current real-model loop. Raw run artifacts and committed validators remain the
machine evidence.

## Current verdict

`NO_PROVEN_LATENT_REASONING_YET`

The project has a real hidden-state-only recurrence runtime over
Qwen3.5-2B, but no experiment has yet shown a replicated reasoning improvement
over the same mechanism at `K=0`. No efficiency gain has been measured;
`K=4` currently spends more decoder compute than `K=0`.

## What is already real

- The latent loop performs hidden-state and cache operations without tokenizer,
  generation, or vocabulary-head calls.
- `K=0` is adapter-free under recurrence-only LoRA. Seed initialization occurs
  before model/LoRA creation and data shuffling.
- The full-reset control restores both state and prompt cache and is prediction-
  exact to `K=0` in completed runs.
- Qwen3.5 LoRA targets are resolved against the actual attention and
  GatedDeltaNet projection modules.
- State interventions (`zero`, norm-matched `noise`, cross-prompt `swap`) are
  separate from prompt/cache interventions.

These facts establish a valid experimental mechanism. They do not establish
useful latent reasoning.

## Real-model result ledger

| Mechanism | Validation | Test-ID controls | Verdict |
|---|---:|---:|---|
| Paired recurrence + visible trace, seed 0 | selected checkpoint did not improve clean accuracy | `K0=38/168`, `K1=38/168`, `K4=36/168`, full-reset `38/168` | Rejected. Margin movement was calibration, not reasoning. |
| Oracle visible safe trace diagnostic | — | base `14/96`, oracle trace `10/96` | Rejected. Even the correct visible trace hurt this 2B readout. |
| M1 counterfactual margin, seed 0 | `K0=51/224`; selected step 280 `53/224` | `K0=38`, `K1=38`, `K4=40`, reset `38`, noise `38` | Rejected as shortcut/null. `K4-K0` gold-margin mean `-0.094`; 6 gains/4 losses; attribution dominated by `stack_queue`. |
| M1 counterfactual margin, seed 1 | step 140 `50/224`; selected step 280 `55/224` | `K0=38`, `K1=36`, `K4=38`, reset `38` | Replicated null. `K4-K0` gold-margin mean `-0.073`; no accuracy gain. |
| M4 causal workspace, seed 0 | active 2B run; no result yet | not evaluated | Tests whether several causally ordered latent slots fix the M1 capacity bottleneck. |

The seed-0 `38→40` observation is not a breakthrough: its confidence interval
crossed zero, its aggregate ranking margin worsened, and the effect disappeared
on replication.

## What the negative results taught us

1. The old objective always changed the final event. It allowed a latent state
   to steer answers using a last-event shortcut without learning composition.
2. The current readout exposes a direct final-boundary state-to-logits channel.
   A prompt-specific latent effect can therefore be real but still not be
   reasoning.
3. Returning a final-boundary carrier to layer 0 is only well-typed when the
   carrier is normalized as the model's latent input representation. The next
   branch makes this explicit and forces readout back through the decoder.
4. Accuracy alone is insufficient at this sample size. Paired margins, answer
   flips, hard-family slices, seed replication, and state ablations decide the
   verdict.

## Active decision loop

1. Finish the M4 run and evaluate `K0/K1/K4/full-reset`; run noise/swap only if
   clean `K4` is promising.
2. In parallel, implement normalized recurrence with decoder-mediated latent
   readout. Run a seed-0, 140-step 2B pilot before any longer training.
3. Evaluate an early-event probe whose last event is unchanged, separately from
   the shortcut-prone `stack_queue` family.
4. Promote only if `K4-K0 >= +5 pp`, `K4 > K1`, full-reset is exact `K0`, and
   noise/swap destroy at least 75% of the gain. Require a positive effect on the
   early-event/non-stack slice before replication.
5. If both M4 and decoder-mediated readout miss those gates, stop tuning this
   recurrence family and reassess whether a stronger latent-supervision
   mechanism is justified.

## Evidence anchors

- `ccbed14` — counterfactual margin objective and causal attribution terms.
- `366542d` — causal multi-slot workspace.
- `e7ea8af` — corrected trace-readout contract assertion.
- `7fdd300` — workspace identity bound to canonical recipe in artifacts.
- `.rcc_work/pilot-2b-full-k4-paired-cfmargin-s0-280-v1/` — seed-0 M1 run.
- `.rcc_work/pilot-2b-full-k4-paired-cfmargin-s1-280-v1/` — seed-1 M1 replication.

Paid compute is explicitly authorized for this project up to approximately
USD 20 for the current session. Exact provider cost and cleanup status are
recorded when the active instance is destroyed.
