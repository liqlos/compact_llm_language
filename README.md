# RCC — Research Context Compiler

> **R1 evidence status (2026-08-27): `VALID_EXPERIMENT_PENDING` ·
> `PAID_SPEND_NOT_AUTHORIZED`.** No retained historical model result is valid
> current evidence for latent recurrence. The machine-readable source of truth
> is `artifacts/milestone_r1_verdict.json`; artifact dispositions are in
> `artifacts/ARTIFACT_CLASSIFICATION.json` and
> `artifacts/HISTORICAL_EVAL_INVALIDATION.json`. A historical READY receipt or
> prose roadmap never authorizes spend or scaling.

**Strategic direction (2026-08-22):** RCC is pivoting to latent cognition on
open-weight Qwen (`Qwen/Qwen3.8-27B` release target; Qwen3.5 proxies).
Document authority:

- current direction and decisions: `VISION.md` and `ADR-001`;
- current maturity and spend authority: `artifacts/milestone_r1_verdict.json`
  and the freshly generated no-spend verdict; `LATENT_ROADMAP.md` is explanatory;
- contribution/integration process: `CONTRIBUTING.md`;
- experiment protocols: `EVALUATION_PROTOCOL.md`, `LIVE_EVAL.md`, and
  `docs/NO_SPEND_GATE.md`;
- historical implementation/evidence records: `IMPLEMENTATION_PLAN.md` and
  `BLOCKERS.md` (not an autonomous task queue).

The existing package is the **evidence substrate**: it is storage-lossless,
deterministic and auditable — but no real-provider live model has been run against
compiled context, so behavioral fidelity is NOT MEASURED and no speedup is
claimed anywhere below.

Compiles long research/coding transcripts into compact active context backed
by an immutable, content-addressed evidence store. Optimizes cost per task,
not raw character counts: bulky old observations become stable reference
stubs; semantic state lives in a symbolic machine layer (RIR/1); originals
stay hash-verifiable and deterministically recoverable.

```
messages (trusted) ──► always inline; protected=True never masked
observations (untrusted) ──► RawStore (content-addressed, write-once)
        │
compile():  recent/small ─────────► inline verbatim
            old/bulky/duplicate ──► [OBS obs-0007 label=… sha=ab12cd34 tok=512]
machine state (RIR/1) ───► appended last: F/Q/N/C atoms with @obs refs
        │
expand(obs_id) ─► byte-exact original inside <UNTRUSTED_OBSERVATION> markers
timeline()   ─► live per-turn audit: what is inline/masked/failopen + tokens
```

## Guarantees (scoped precisely)

- **Storage-lossless**: masked ≠ lost. `expand()` returns the full original
  bytes, SHA-256 verified on every read. No referential dangling by
  construction. (Access-losslessness and behavioral fidelity are separate,
  weaker claims — see VISION.md; both unmeasured with a real-provider model.)
- **Prefix-stable**: stubs are written once and never rewritten; the RIR/1
  block only appends → prompt-prefix caching stays valid across turns.
- **Fine-fact guard**: signed integer, decimal and exponent tokens in
  machine-state atoms are normalized and must match whole numeric tokens in
  their cited sources. Substrings such as `142` in `1420` or `2` in `20.1`
  are rejected; this is a provenance guard, not a truth oracle.
- **Fail-open availability / fail-closed integrity**: masking requires a
  verified store object (else verbatim text or an honest
  `<OBSERVATION_UNAVAILABLE/>` marker); tampered objects raise.
- **Run isolation**: references carry run IDs; load, compile and expand all
  fail closed on cross-run or tampered persisted references.
- **Injection quarantine**: untrusted content appears only inside
  `<OBSERVATION>` / `<UNTRUSTED_OBSERVATION>` wrappers.

## Machine reasoning state — RIR/1

```python
from rcc import ResearchSession, RawStore, Policy

s = ResearchSession("run-1", RawStore(".rcc_data"), Policy())
oid = s.observe("release_notes", tool_output_text)

sc = s.attach_scratch()
sc.add("F", "p95 latency 142ms under LT-77", src=(oid,), conf=0.98)
sc.add("Q", "writer lock ordering?")
sc.add("N", "verify lock ordering via simulation")

ctx = s.compile()      # <SCRATCH format=RIR/1> block appended last
```

Format cost for identical working state (o200k_base, see
`bench/formats.py`): RIR/1 216 tok < csv 226 < markdown 229 < yaml 269 <
prose 253* < json 340 (*prose reads cheaper than YAML/JSON here — matches
arXiv:2605.29676). RIR wins on both counters and keeps numbers verbatim.

## Baseline mode & observability

`Policy(enabled=False)` inlines everything (legacy behaviour).
`s.timeline()` shows per-turn what the model sees — inline vs masked vs
fail-open — so nothing happens invisibly "until the end".

## Tests & benchmark

```bash
uv sync --frozen --group dev --no-group lab --no-group mlx
uv run --no-sync python -m pytest       # dependency-light core contract
uv sync --frozen --group dev --group lab --no-group mlx
uv run --no-sync python -m pytest \
  --run-lab-unit --run-transformer-integration --run-remote-policy
                                           # all local/offline opt-in tiers
uvx ruff check rcc bench tests evals       # lint
uv run python -m bench.run_bench --exact --json .rcc_bench/results.json
uv run python -m evals.run_live_eval --provider fake   # live-eval harness (offline fixture)
```

## R1 benchmark and scorer contract

New latent measurements use immutable `behavioral-v3` and record
`latent_eval.v3` (`latent_eval.summary.v3`). The preregistered primary
candidate score is `mean_candidate_token_logprob_v1`; raw per-token values,
raw sums and token counts remain in every record. Exact top ties are
`AMBIGUOUS_TOP_TIE` errors under `exact_top_tie_is_error_v1`, never index-based
winners. Checkpoint selection is valid only from raw validation records
recomputed by the same canonical scorer used for final and offline evaluation.
No such model experiment has been executed yet.

## No-spend integrity gate (latent evidence)

Deterministic, hardware-free-capable gate over the retained 2B/4B artifacts:
inventories + hashes every file, validates train-report pins (strict JSON,
finite metrics, non-negative integer steps), classifies checkpoints via
safe/strict identity-bound bundle loading with FP32 verified from the
payload tensors, joins every retained checkpoint to exact-model/revision/
suite eval evidence covering both ID and OOD splits, discovers orphans and
duplicate bindings symmetrically, verifies the rejected 4B batch is nonempty,
markered and fully contained, proves inputs unchanged via before/after
streaming fingerprints, and emits one canonical verdict.

```bash
uv run python -m latent_lab.bench.no_spend_gate   # full mode (CPU-only; runs proof regressions)
uv run python -m latent_lab.bench.no_spend_gate --dry-run   # hashes/metadata only
```

Regenerate the historical classification from the canonical private evidence
root without copying model files into Git:

```bash
RCC_R1_EVIDENCE_ROOT=/absolute/path/to/repository/.rcc_work
uv run --frozen --group dev --group lab --no-group mlx \
  python tools/classify_r1_artifacts.py \
  --evidence-root "$RCC_R1_EVIDENCE_ROOT"
```

Exit codes: `0` READY · `1` NOT_READY · `2` execution error (bad invocation,
unreadable inputs, inputs mutated mid-run, unwritable outputs). `--out` may
never overlap an input root. Every known fail-open is a blocker; negative
controls for each audited fail-open are pinned in
`tests/test_no_spend_gate.py`.
Gate outputs are byte-stable across reruns against unchanged inputs and contain
no wall-clock value (the timestamp is separate telemetry). Retained receipts are
`.rcc_work/rcc.pre_spend.v2.json` and `.rcc_work/rcc.canary_attempt.v1.json`;
they are historical only. The current authority is
`artifacts/milestone_r1_verdict.json`, which keeps
`paid_spend_authorized: false` (`PAID_SPEND_NOT_AUTHORIZED`). See
`docs/NO_SPEND_GATE.md` for the contract and historical result.

Live-model evaluation (`evals/`) replays the five benchmark scenarios plus a
focused RIR/1+router case, baseline vs compiled, over an explicit expansion
channel (`tool` / `closed`). It scores exact-fact recall, numeric integrity,
citations, constraints and injection resistance against
`docs/research-context-compiler/GROUND_TRUTH_SPEC.md`. The provider interface is
OpenAI-compatible, credentials are environment-only, and calls have a hard
budget. The five benchmark scenarios do not exercise SCRATCH/router
comprehension; only the added `rir_state` case does, minimally. Usage:
`docs/research-context-compiler/LIVE_EVAL.md`.

Token-savings numbers below are **synthetic token estimates**, not model
latency or task success. They justify nothing about end-to-end speed.

Measured (o200k_base, exact tokenizer): peak active context −61…−78% on
representative scenarios; fact recall via expansion 100%; cumulative spend
−42% over a full long-research run (grows with horizon). Details:
`docs/research-context-compiler/IMPLEMENTATION_PLAN.md`.

## Status & layout (maturity scale: HYPOTHESIS / SCAFFOLDED / UNIT_VERIFIED / MODEL_VERIFIED / REPLICATED / REJECTED_BY_EVIDENCE / BLOCKED)

| Layer | State |
|---|---|
| 0 Measurement / baseline (token estimates) | UNIT_VERIFIED |
| 1 Stable refs, dedup, safe masking | UNIT_VERIFIED |
| 2 Immutable raw store | UNIT_VERIFIED |
| 3 Protected exact channel | SCAFFOLDED (caller-controlled flag; not a trust boundary) |
| 4 Checkpoint + delta journal | UNIT_VERIFIED |
| 5.1 Symbolic machine state (RIR/1) | SCAFFOLDED (audit/fallback role; see VISION.md) |
| 5.2 Mode router | SCAFFOLDED (experimental) |
| 5.3 Compressor plug point + gate | interface only |
| 6 Dictionary encoding | SCAFFOLDED (experimental) |
| 7 Break-even gate | SCAFFOLDED (deterministic core; q/N never calibrated) |
| Behavioral fidelity on live model | NOT MEASURED — blocks all downstream claims |
| Latent pipeline plumbing (`latent_lab/`) | UNIT_VERIFIED locally: hidden-state-chain BPTT with explicitly detached KV/cache recurrence; same adapter accepts K=0/1/2/4/8; full-decoder candidate tails are autoregressive; unsupported `grad_checkpoint` was removed and historical `true` fails closed |
| No-decode recurrence guard | UNIT_VERIFIED locally and scoped to the guarded recurrence context: tokenizer call/encode/decode/batch-decode/chat-template, model generate, and direct output-head paths raise; the guard is restored on exit |
| Historical Qwen hybrid runtime probe | HISTORICAL_UNBOUND — retained 0.8B state-probe JSON is runtime evidence only, not reasoning-quality or current scorer evidence |
| MLX off-vocabulary embedding path | MEASURED: exact-vocab embeds safe/fast; off-manifold inputs slow ~3.5–5×; hybrid prefix-cache trim unsupported (`results/mlx_soft_embedding_probe.json`) |
| Historical 2B latent outputs | IRRECOVERABLE_LEGACY_SCORER — 50 derived-only files cannot be rescored; their old `correct`, rank, above-chance and K-dose values are not accuracy or current evidence; 13 selected checkpoints have `selection_provenance_invalid` |
| Historical textual baselines @2B | HISTORICAL_UNBOUND — stored 0/112 derived correct flags do not establish a model capability limit; full outputs and canonical raw score records are absent |
| Historical textual baselines @4B | HISTORICAL_UNBOUND, preview-only — stored direct/no-thinking flags are 97/112 ID and 84/112 OOD versus native-thinking 1/112 and 0/112; this strong contrary observation is not independently rescorable and must be rerun on behavioral-v3 |
| MLX internal-recurrence path | MEASURED: bit-exact composition, speed BLOCKED without compile | `latent_lab/bench/mlx_internal_recurrence_probe.py` |
| Qwen3.8-27B speedup ≥2× | NOT MEASURED | no scaling permission; valid small behavioral-v3 experiment pending |
| Live-model evaluation | UNIT_VERIFIED harness + deterministic fixture; real-provider run pending |

```
rcc/            core library (tokens, store, scratch, session, journal,
                router, dictenc, gate) — evidence/compatibility layer
latent_lab/     latent-cognition experiments (protocols, mock backend,
                state probe, bench scaffolding); optional heavy deps
bench/          deterministic token-estimate scenarios (synthetic!)
evals/          provider-neutral baseline-vs-compiled live-model harness
tests/          unit / security / property / integration suites
docs/research-context-compiler/
                vision, ADRs, roadmap, evaluation protocol, blockers,
                implementation plan with measurements
```

## Research grounding

AgentFold (2510.24699), ACM (2607.23809), referential dangling (2608.04569),
Telegraph English (2605.04426), Notation Matters (2605.29676), OpenHands SDK
(2511.03690), LLMLingua (EMNLP'23), rate-distortion limits (NeurIPS'24);
latent-direction roadmap: ICAE (2307.06945), Gist (2304.08467),
Coconut (2412.06769) — see plan §0 and §7.
