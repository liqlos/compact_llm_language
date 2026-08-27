# Compact LLM Language

> Archived research repository, 2026-08-27.
>
> **STATUS: `NO_PROVEN_LATENT_REASONING_GAIN`**
>
> **MECHANISM: `IMPLEMENTED_AND_CONTROLLED`**

This repository asks a narrow question: can tokenizer-free hidden-state
recurrence improve reasoning in a pretrained Qwen model relative to the same
model with no recurrence?

The tested answer is **no**. We implemented real hidden-state recurrence over
`Qwen/Qwen3.5-2B`, fixed the main control failures, and ran matched
`K=0/K=1/K=4`, reset, intervention, and seed checks. None produced
a replicated answer-level gain. `K=4` also executes more decoder work than
`K=0`; this project demonstrated no latency, memory, or energy advantage.

This is a scoped negative result for the recipes, model scale, data, and budget
tested here. It is not a proof that latent reasoning is impossible.

## Why this needs training

An autoregressive language model uses continuous activations inside each
forward pass, but the standard boundary between generation steps is discrete:
logits select a token, and that token's embedding becomes the next input.
Feeding a final hidden state back as a new input changes the representation,
position, and cache semantics. The model was not pretrained for that loop, so
the loop must be implemented explicitly and generally needs task-specific
training or pretraining.

## What was built

- A tokenizer-free recurrent path with hidden-state and KV/cache updates.
- Localized and full-decoder recurrence, including a multi-slot latent
  workspace.
- LoRA targeting resolved against the actual Qwen3.5 attention and Gated
  DeltaNet modules.
- Seed initialization before model, LoRA, and data shuffling.
- Adapter-free `K=0`; exact full-state reset back to `K=0`.
- Separate state interventions: zero, norm-matched noise, and cross-prompt
  swap, without silently replacing the prompt or cache.
- A deterministic synthetic reasoning suite, raw-score evaluator, checkpoint
  identity binding, and causal early-event probes.

The toy recurrence trainer does learn its toy task and fails under loop
ablation. That proves the plumbing can carry a causal latent signal; it does
not establish reasoning in a real language model.

## Real-model results

All counts below use the same frozen Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. The primary held-out split has
168 examples.

| Training recipe | Seed | K=0 | K=1 | K=4 | Full reset | Other control | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Paired recurrence + visible-trace curriculum | 0 | 38 | 38 | 36 | 38 | — | Negative |
| Counterfactual-margin recurrence, one slot | 0 | 38 | 38 | 40 | 38 | noise: 38 | Null/shortcut; margin worsened |
| Counterfactual-margin recurrence, one slot | 1 | 38 | 36 | 38 | 38 | — | Replicated null |

The apparent seed-0 `38→40` movement was not promoted: its confidence
interval crossed zero, aggregate gold margin decreased, the gains concentrated
in a last-event shortcut family, and seed 1 did not reproduce it.

A final four-slot workspace run reached `53/224` on validation versus the
adapter-free `K=0` baseline of `51/224` (`+0.9` percentage points). This
missed the preregistered `+5 pp` gate, so no additional paid test/ablation
run was justified. The run took 2,193.7 seconds; CUDA reported 3,759.5 MiB peak
allocated and 7,828.0 MiB peak reserved. Total rented-GPU spend for the closing
session was approximately USD 2.51.

The detailed ledger, including artifact hashes and falsification rules, is in
[`LATENT_EXPERIMENT_STATUS.md`](docs/research-context-compiler/LATENT_EXPERIMENT_STATUS.md).

## Relation to published work

Latent reasoning is an active research area, not an ignored idea. Positive
results usually rely on a visible-to-latent curriculum, distillation, or
recurrent pretraining:

- [COCONUT](https://arxiv.org/abs/2412.06769) feeds the last hidden state back
  as the next input embedding.
- [CODI](https://arxiv.org/abs/2502.21074) distills explicit chain-of-thought
  into continuous thoughts.
- [Huginn](https://arxiv.org/abs/2502.05171) pretrains a depth-recurrent model
  from scratch.
- [SIM-CoT](https://arxiv.org/abs/2509.20317) adds a training-only decoder to
  supervise each latent step.

The strongest caution is that matched-data controls often erase the reported
advantage on logical tasks
([Dilgren & Wiegreffe, COLM 2026](https://arxiv.org/abs/2604.04902)), and causal
interventions find that latent steps are task-dependent and sometimes unused
([Aswal et al., 2026](https://arxiv.org/abs/2606.12689)). Recent soft-token work
also finds that continuous-token training can help while discrete-token
inference remains stronger
([Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)).

Our conclusion is therefore narrower than “latent reasoning does not work”:
this inexpensive pretrained-2B retrofit did not work, and scaling it without a
much stronger training signal was not justified.

## Repository map

- `latent_lab/` — recurrence runtimes, training, controls, evaluator, and
  rented-compute helpers.
- `rcc/` — the earlier storage-lossless Research Context Compiler substrate.
- `bench/`, `evals/` — deterministic context and behavioral evaluation
  harnesses.
- `artifacts/` — checked-in suite identities and historical evidence audits.
- `docs/research-context-compiler/` — design history and the final experiment
  ledger.

Historical behavioral-v2 artifacts are explicitly
`IRRECOVERABLE_LEGACY_SCORER` and are not evidence for this conclusion. Their
old milestone remains `PAID_SPEND_NOT_AUTHORIZED` in
`artifacts/milestone_r1_verdict.json`; the later small 2B runs were separately
authorized and are reported above.

## Reproduce the software checks

```bash
uv sync --frozen --group dev --no-group lab --no-group mlx
uv run --no-sync python -m pytest

uv sync --frozen --group dev --group lab --no-group mlx
uv run --no-sync python -m pytest \
  --run-lab-unit --run-transformer-integration --run-remote-policy
```

Real-model runs require downloading Qwen weights and substantially more compute
than the default tests. No model checkpoint or claim of pretrained-model
efficiency is published.

## License

MIT. See [`LICENSE`](LICENSE).
