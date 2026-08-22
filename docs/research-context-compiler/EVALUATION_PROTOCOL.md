# Evaluation Protocol — RCC latent experiments

Version: 2026-08-22. All benchmark results must embed a config manifest
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

If zeroing/shuffling latent states does not hurt, the model ignores the
latent path — such a run is recorded as failure, not success.

## Reporting rules

Every claim carries: command, model/revision, runtime version, hardware,
precision, result, skipped/failed parts. Best runs are never reported without
seeds and variance. Negative results are reported.
