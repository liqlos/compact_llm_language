# Live-model evaluation harness

Status: **harness DONE (fixture-tested); live run pending a configured
provider.** This closes the measurement-side gap from plan §4/§8: task quality
is no longer only proxied by deterministic recoverability checks — the harness
replays the five benchmark scenarios through an actual model and scores its
answers. What it cannot do offline is tell you how a *real* model behaves; the
deterministic fixture validates mechanics, not capability.

## Protocol (per scenario x mode)

1. **Replay**: the scenario transcript is replayed through `ResearchSession`
   exactly as `bench.harness` does; the FINAL compiled context is what the
   model sees (`baseline` = everything inline forever, `compiled` = stubs +
   RIR/1 under the default policy).
2. **Ask**: model receives `[system rules] + [context + question]`.
   If needed evidence is referenced only by a stub, it may reply with nothing
   but `EXPAND: <obs-id>` lines; the harness appends the hash-verified
   expansion blocks and asks again. Expansion is bounded by
   `--max-expand-rounds` (default 1) — this models RCC's recovery path and its
   extra read cost is *recorded*, never hidden.
3. **Score** the final answer deterministically:
   - `fact_recall`: fraction of per-task ground-truth substrings present
     verbatim (facts are unique anchors inside their source observation);
   - `citation_present`: answer references an `obs-NNNN` id;
   - `injection_resisted`: none of the scenario's payload fragments appear;
   - `constraints_passed`: e.g. "under 100 words" for the constraints task;
   - plus cost counters: `context_tokens`, `expand_requests`, call rounds.

## What the fixture run proves (and does not)

`uv run python -m evals.run_live_eval --provider fake --out .rcc_eval/results.json`

mechanically demonstrates the RCC hypothesis end-to-end with a deterministic
client that answers strictly from visible prompt content:
recall parity at far lower context cost —

| mode     | mean fact recall | context tokens | expand reads |
|---|---|---|---|
| baseline | 1.00 | 42,730 | 0 |
| compiled | 1.00 | 15,582 | 15 |

With `--max-expand-rounds 0`, compiled `exact_facts` recall drops below
baseline (facts are recoverable, not inline) — the tests assert both parity
*and* this differentiation. **The fixture is not a capability model**; a real
run may score worse and that difference is exactly the measurement this
harness exists to take.

## Real run (any OpenAI-compatible endpoint)

No SDK or paid service is installed; a stdlib client POSTs to
`{base_url}/chat/completions`. Works with OpenAI, Ollama (`ollama serve`),
llama.cpp `llama-server`, LM Studio, vLLM.

```bash
# local server (no key needed)
RCC_EVAL_BASE_URL=http://127.0.0.1:11434/v1 RCC_EVAL_MODEL=llama3.1:8b \
    uv run python -m evals.run_live_eval --provider openai-compat \
    --out .rcc_eval/results.json

# hosted API
RCC_EVAL_BASE_URL=https://api.openai.com/v1 RCC_EVAL_MODEL=gpt-4o-mini \
    RCC_EVAL_API_KEY=sk-... \
    uv run python -m evals.run_live_eval --provider openai-compat
```

Cost & privacy bounds, by construction:

- at most `scenarios x 2 x (1 + max_expand_rounds)` completion calls
  (default ≤ 15), `temperature=0`, `--max-completion-tokens` cap (default 512);
- the API key comes from args or `RCC_EVAL_API_KEY` **only**, and is never
  logged or serialized (`config.api_key` is always `null`);
- no other environment state is touched; results land under `.rcc_eval/`
  (gitignored) as machine-readable JSON with per-answer records and a
  `summary_by_mode` block.

Exit code is 0 when every task completed without provider errors.
