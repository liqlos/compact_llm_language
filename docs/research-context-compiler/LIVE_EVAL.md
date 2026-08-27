# Live-model evaluation harness

> R1 status: harness-only work is allowed locally; paid or remote provider use
> is `PAID_SPEND_NOT_AUTHORIZED` until
> `artifacts/milestone_r1_verdict.json` explicitly says otherwise.

Status: **harness DONE, fixture-tested, ground-truth vectors incorporated;
live run pending a configured provider.** Implements the measurement the plan
(§4/§8) called for: task quality scored over real model answers instead of
deterministic recoverability proxies only.

## Protocol (per scenario x mode x expansion-channel)

1. **Replay**: scenario transcript replayed through `ResearchSession` exactly
   as `bench.harness` does; the FINAL compiled context is what the model sees
   (`baseline` = everything inline forever; `compiled` = stubs + RIR/1 under
   the default policy). The exact `Policy`, tokenizer id and a sha256 context
   hash are recorded per result.
2. **Ask** over an EXPLICIT expansion channel (recorded on every result):
   - `tool` (default): model may reply solely with `EXPAND: <obs-id>` lines;
     harness appends hash-verified expansions and re-asks, bounded by
     `--max-expand-rounds`. This models RCC's recovery path; its read cost is
     recorded (`expand_requests`), never hidden.
   - `closed`: no expansion option exists (system prompt omits it).
     Compiled fact recall is NEVER scored on this arm — facts may legitimately
     be absent (`facts_inline=0.0` measured for exact_facts/compiled); EF3
     honest-unavailability is scored instead, and confident fabrication is a
     critical failure. **Arms are never compared across channels.**
3. **Score** deterministically (`evals/scoring.py`, per
   [GROUND_TRUTH_SPEC.md](GROUND_TRUTH_SPEC.md)):
   - EF1 verbatim recall + lenient regexes (e.g. `142 ms`, `V9.2.1`);
   - EF2 numeric integrity vs an evidence allowlist — hard gate only where
     the allowlist is discriminative (exact_facts, rir_state), report-only
     elsewhere (spec A4);
   - EF3 honest unavailability (compiled closed-book only);
   - CIT1 validity / CIT2 coverage / CIT3 sha-integrity;
   - CON1 word limit / CON2 destructive-exec deny-list;
   - INJ1 compliance markers / INJ2 system-prompt leak / INJ3 task completion.

## Decisions taken on spec ambiguities

- **A1 expansion channel**: option (a) adopted — expand-as-tool via a bounded
  text protocol, matching "fact recall via expansion" design intent; the
  closed-book arm exists as a separate channel (`--expansion closed|both`) to
  measure refusal honesty and hallucination pressure.
- **A2 "under 100 words"**: DECIDED strict `<N words`; exactly 100 fails.
- Citation granularity: obs-id or label accepted (spec §Ambiguities);
  prose-title citations not counted.
- The `injection` scenario is a **CONTROL PAIR** (contexts byte-identical
  across modes — asserted by test): reported separately from mode aggregates;
  divergence there means provider noise, not mode effect.

## RIR/1 & router scope (claims narrowed)

The five bench scenarios never attach scratch atoms nor enable the router, so
they cannot speak to `<SCRATCH>` comprehension. One focused eval-local case,
`rir_state`, was added: release_notes is masked in the compiled arm so its
gold facts survive ONLY inside `<SCRATCH format=RIR/1>` F-atoms, with
`router_enabled=True` (router lands on EXPERT at 9 observations). It is a
minimal comprehension probe — not a routing study — and results must not be
read as broader RIR validation.

## What the fixture run proves (and does not)

```bash
uv run python -m evals.run_live_eval --provider fake --expansion both \
    --out .rcc_eval/results.json
```

With the deterministic fixture (answers strictly from visible prompt content),
`--expansion both`, the exact_facts task illustrates the channel rule — recall
is only comparable where the facts are visible:

| arm (exact_facts) | EF1 | EF3 | note |
|---|---|---|---|
| baseline/tool    | 1.00 | – | facts inline |
| compiled/tool    | 1.00 | – | recovered via EXPAND obs-0001 |
| baseline/closed  | 1.00 | – | facts inline |
| compiled/closed  | 0.00 | PASS | honest unavailability, no fabrication |

Suite-level aggregates for the same run: compiled/tool matches baseline/tool
at mean EF1 1.00 with −62% context tokens and 7 recorded expansion reads;
compiled/closed mean EF1 is 0.80 (only exact_facts drops; fact-free tasks and
tasks whose facts stay visible score normally); zero critical failures
anywhere; control hashes byte-identical across modes.

**The fixture validates mechanics, not capability** — a real model may score
worse, and that difference is exactly what this harness measures.

## Real run (any OpenAI-compatible endpoint)

No SDK or paid service installed; stdlib client POSTs to
`{base_url}/chat/completions`. Works with OpenAI, Ollama (`ollama serve`),
llama.cpp `llama-server`, LM Studio, vLLM.

```bash
# optional pre-flight: one tiny completion verifies endpoint+model+auth
uv run python -m evals.run_live_eval --provider openai-compat --check

# local server (no key needed)
RCC_EVAL_BASE_URL=http://127.0.0.1:11434/v1 RCC_EVAL_MODEL=llama3.1:8b \
    uv run python -m evals.run_live_eval --provider openai-compat \
    --out .rcc_eval/results.json

# hosted API
RCC_EVAL_BASE_URL=https://api.openai.com/v1 RCC_EVAL_MODEL=gpt-4o-mini \
    RCC_EVAL_API_KEY=sk-... \
    uv run python -m evals.run_live_eval --provider openai-compat
```

Cost, budget & privacy bounds:

- hard total-call budget (`Budget`): default equals the suite bound
  (6 cases × arms × modes × (1+expand rounds) → ≤ 24 calls on default
  settings; override with `--max-calls`). When exhausted, remaining tasks
  record `budget_exhausted` while all completed results stay in the report
  (partial-result isolation).
- per request: timeout 60 s, temperature 0, `--max-completion-tokens` cap,
  bounded retry with backoff for transient failures (connection errors,
  HTTP 429/5xx) — auth/4xx fail fast.
- API key from args or `RCC_EVAL_API_KEY` env ONLY; never logged or
  serialized (`config.api_key` always null).
- results land under gitignored `.rcc_eval/` as versioned JSON
  (`schema_version: 1`) with config, per-arm summary, control results and
  per-answer records (policy, context hash, raw responses, errors).

Exit code 0 iff no task errored.
