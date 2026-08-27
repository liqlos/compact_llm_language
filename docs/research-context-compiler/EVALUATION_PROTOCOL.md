# Evaluation Protocol — RCC latent experiments

Version: 2026-08-27. Status: `ARCHIVED_PROTOCOL`. The original R1 gate was
`VALID_EXPERIMENT_PENDING` / `PAID_SPEND_NOT_AUTHORIZED` per
`artifacts/milestone_r1_verdict.json`; later bounded 2B experiments were
separately authorized and ended with `NO_PROVEN_LATENT_REASONING_GAIN`.
See [`LATENT_EXPERIMENT_STATUS.md`](LATENT_EXPERIMENT_STATUS.md).
All benchmark results must embed a config manifest
(model repo+revision, runtime versions, dtype, hardware, seeds, exact command)
and be committed as machine-readable JSON under `latent_lab/bench/results/`
(gitignored if bulky; manifest always committed).

## Primary metric

**Wall-clock time per successful task.** Not "tokens saved", not tokens/sec.

## Secondary metrics

task success / accuracy; quality-adjusted latency; time to first action;
time to final answer; prefill time; visible decode time; latent recurrence
time; number of latent steps; number of sequential layer applications;
generated visible tokens; generated reasoning tokens; replayed textual
context; peak RAM/VRAM; cache/state bytes; estimated FLOPs; tool-call count;
raw-evidence expansions; citation/evidence correctness.

## Canonical behavioral-v3 scoring record

Latent candidate evaluation has one record schema, `latent_eval.v3`, and one
summary schema, `latent_eval.summary.v3`. The fixed primary candidate score is
`mean_candidate_token_logprob_v1`; raw per-token log probabilities, raw sums
and token counts are retained for independent rescore. Exact top ties produce
`AMBIGUOUS_TOP_TIE` under `exact_top_tie_is_error_v1`; candidate index never
breaks a tie. Validation checkpoint selection, online/final scoring, offline
rescore, reports and the no-spend gate use this same implementation and source
hash. Selection provenance is valid only as
`latent_eval.v3_recomputed_from_raw_validation_records`.

The immutable suite is `behavioral-v3`; its final test is untouched and never
used for checkpoint selection. Historical behavioral-v2 derived-only records
are `IRRECOVERABLE_LEGACY_SCORER`, not accuracy, and cannot be migrated by
inventing missing raw scores.

## Mandatory baselines (same checkpoint, precision, seed policy, tasks)

| ID | Configuration |
|---|---|
| A | direct / no-thinking |
| B | native visible CoT |
| C | native preserved textual thinking + normal cache |
| D | existing textual RCC context |
| E | full-decoder Coconut-like recurrence |
| F | localized latent recurrence |
| G | localized recurrence + latent memory |

Never compare different models and call the difference an RCC effect.

## Task classes (minimum matrix)

deterministic structured reasoning; arithmetic with numbers and units;
negations; multi-hop dependencies; conflicting sources; temporal updates;
exact-fact retrieval; code reading; bug localization; test-result
interpretation; structured tool selection; prompt injection inside untrusted
evidence; out-of-distribution tasks. Start small with deterministic scorers;
coding and agent loops only after the gate passes.

## Anti-cheating controls (mandatory)

- zero latent state ablation (K=0);
- shuffled latent steps;
- latent states swapped between examples;
- truncated recurrent depth;
- counterfactual evidence;
- answer-leakage audit;
- held-out task families;
- matched-latency comparison;
- matched layer-call/compute comparison;
- proof that no hidden textual CoT is generated inside the latent loop;
- causal check that K=0 vs K>0 actually differ.
- K=0/1/2/4/8 evaluation of the same trained adapter; a separately trained K0
  is reported only as an additional control, never a recurrence ablation.

If zeroing/shuffling latent states does not hurt, the model ignores the
latent path — such a run is recorded as failure, not success.

## Reporting rules

Every claim carries: command, model/revision, runtime version, hardware,
precision, result, skipped/failed parts. Best runs are never reported without
seeds and variance. Negative results are reported.
