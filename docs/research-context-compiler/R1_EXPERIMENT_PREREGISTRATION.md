# R1 next experiment preregistration

Status: `HISTORICAL_NOT_EXECUTED`

This exact R1 plan was never authorized or executed. It is retained to show
the frozen gate that preceded the later, separately authorized bounded 2B
experiments. The project is now archived with
`NO_PROVEN_LATENT_REASONING_GAIN`; see
[`LATENT_EXPERIMENT_STATUS.md`](LATENT_EXPERIMENT_STATUS.md).

The machine-readable source of this preregistration is
`artifacts/r1_experiment_preregistration.json`. Behavioral-v3, its candidate
permutations, `mean_candidate_token_logprob_v1`, all arms, three seeds, the
paired confidence interval, materiality thresholds, stopping rules, and the
untouched final-test rule are frozen there before any model run.

The primary comparison uses the same recurrence-trained adapter at
`K=0/1/2/4/8`; a separately trained K0 adapter is secondary and is never
described as the clean causal recurrence ablation. The other mandatory controls
are direct/no-thinking text, visible scratch capped at 64 generated tokens,
correct full-decoder recurrence, a compute-matched clock permutation, and
zero/noise/swap/bypass interventions. Every comparison is paired on the exact
same example IDs.

Success is conjunctive. The recurrence result must beat both same-adapter K0
and direct text by at least 0.05 micro accuracy, with the 95% paired-bootstrap
lower bound above 0.03; macro-by-family delta must be at least 0.03; a
leave-one-family-out audit must stay positive; and every causal ablation must
hurt by at least 0.03 with a paired lower bound above zero. It must also remain
finite, keep error/nontermination at or below 2%, reproduce checkpoint
selection from raw records, survive candidate permutation, and beat the
strongest successful text control in wall-clock time per successful task.

Local smoke (no model download and no paid operation):

```bash
uv run --frozen --group dev --group lab --no-group mlx \
  python -m pytest -q --run-lab-unit --run-transformer-integration \
  tests/test_benchmark_v3.py tests/test_eval_v3.py \
  tests/test_latent_runtime_integrity.py tests/test_latent_run.py \
  tests/test_no_spend_gate.py tests/test_dictenc.py \
  tests/test_provenance_links.py tests/test_scratch.py \
  tests/test_session.py tests/test_store.py tests/test_tokens.py
```

Potential remote command, explicitly **NOT EXECUTED**:

```bash
RCC_PAID_SPEND_AUTHORIZED=1 \
RCC_R1_PREREG_ACK=5cf5cbf397510ba597b59f7ccf0839cf344e6fb795a5cb29d031f39dac218254 \
bash latent_lab/bench/r1_experiment_driver.sh \
  --seeds 0,1,2 --device cuda --out .rcc_work/r1_experiment
```

The bounded estimate is 18 GPU-hours, or USD 8.10 at the preregistered rate
cap. The hard limits are USD 0.45/hour, eight wall-hours per seed (24 GPU-hours
total), USD 10.80 compute and USD 12.00 total. These are policy caps and an
engineering estimate, not a current provider-price quote. No instance is
created by this command or by this R1 session; an already authorized and
independently bounded execution environment is required.
