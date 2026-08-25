# Research Context Compiler — Implementation Plan

Status legend (legacy phases): `NOT_STARTED` / `IN_PROGRESS` / `BLOCKED` / `DONE` / `REJECTED_BY_EVIDENCE`

> **2026-08-22 strategic reset:** the project's end-state is now latent
> cognition on open-weight Qwen (`Qwen/Qwen3.8-27B`). This plan's DONE
> markers describe the deterministic evidence substrate only; they do NOT
> imply live-model validation, behavioral fidelity, or any speedup. Current
> direction lives in VISION.md, ADR-001, LATENT_ROADMAP.md,
> EVALUATION_PROTOCOL.md, BLOCKERS.md. RIR/1 is repositioned as
> audit/debug/fallback representation, not the "language of thought".

## 0. Evidence review (2026-08-22 literature pass)

Key findings that shaped or validated this implementation:

| Source | Finding | Implication for RCC |
|---|---|---|
| AgentFold (arXiv:2510.24699) | Re-summarizing history loses ~1%/step: detail survival ≈ 0.99¹⁰⁰ ≈ 36.6% by step 100 | Masked stubs are written once and never rewritten (`test_stub_never_rewritten_across_turns`) |
| ACM (arXiv:2607.23809) | Lossless agentic CM: offload raw turns to external memory, query by ID on demand; reduces peak token pressure | Independent confirmation of RCC's stub→store reference pattern |
| Referential dangling (arXiv:2608.04569) | Hard compressors sever dependent evidence in **34–60%** of multi-hop cases at r=0.30 | RCC masks whole observations atomically; expansion returns full bytes (`test_no_referential_dangling_by_construction`) |
| Rate-distortion limits (NeurIPS 2024) | Large optimality gap of existing compressors; low-distortion compression barely beats no-compression | Supports deterministic masking over model-based rewriting for observations |
| Telegraph English (arXiv:2605.04426) | Structured machine dialects work; *fine facts* (numbers+units, conditions, negations) are exactly what entropy-based deletion destroys | Motivates protected channel + byte-exact recovery (`test_negation_and_fine_facts_survive_mask_expand_cycle`) |
| LLMLingua (EMNLP 2023) | Quality degrades sharply beyond ~20–25× compression; Selective-Context −33 pts on GSM8K | Stay far below lossy ratios; mask only what stays recoverable |
| OpenHands condenser (blog + SDK arXiv:2511.03690) | LLM summarizing condenser: ~2× per-turn cost cut, 54% vs 53% SWE-bench solve; earlier config cost **+$40 due to worse cache utilization**; event log immutable, condensers stateless | Cache/prefix stability matters more than raw ratio; immutable log = same principle as RCC store; RCC adds losslessness (no compressor LLM call at all) |

Positioning: RCC occupies the currently sparse cell *compact + lossless + deterministic* (cf. ACM's taxonomy table); it defers agent-initiated folding policy to a later phase.


## 1. Repository assessment (initial inspection)

- The working directory was **completely empty** (no AGENTS.md, CLAUDE.md,
  SPEC.md, README, manifests, code, tests; not a git repository).
- Therefore there was no existing context-management implementation to extend;
  this is a greenfield build. All conventions below were chosen by the
  implementer and are recorded as assumptions.

### Assumptions made (reversible)

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python ≥3.11, stdlib-only core | No manifest existed; avoids unnecessary deps |
| Tests | pytest (dev dependency-group, uv) | De-facto standard; installed via `uv sync` |
| Tokenizer | deterministic BPE-style estimator (`rcc-approx-1`) | Reproducible, no provider dependency; deltas meaningful, absolutes approximate |
| Persistence | write-once JSON objects + full-state session save/load | Simplest reversible form enabling resume measurement |
| Storage layout | `.rcc_data/runs/<run_id>/<kind>/<sha[:2]>/<sha>.json` | Content-addressed ⇒ free dedup; per-run prefix ⇒ isolation |

## 2. Phases

### Phase 0 — Measurement & baseline — DONE
- `rcc/tokens.py`: deterministic estimator (`rcc-approx-1`) + exact
  `count_tokens_exact` (tiktoken o200k/cl100k, optional dev dependency).
- `bench/harness.py`: replays scenarios turn-by-turn; records total/peak/final
  tokens, inline/stub split, fail-open counts, per-turn token series,
  compile latency; pluggable tokenizer.
- Baseline = `Policy(enabled=False)` (everything inline forever), same code path.

### Phase 1 — Stable references, dedup, safe observation masking — DONE
- `RawStore`: content-addressed, write-once, SHA-256 verified on read;
  identical payloads dedupe to one object per run (`tests/test_store.py`).
- Canonical stable IDs: identical content maps to one `obs-NNNN`; repeat
  occurrences become stubs referencing the canonical ID.
- Masking policy (`Policy`): keep_recent window, min-token floor (a stub must
  actually save tokens), immediate dup stubbing, global feature flag.
- Fail-open: mask only when store object verifies; else verbatim text or
  honest `<OBSERVATION_UNAVAILABLE/>` marker (`test_failopen_when_store_object_missing`).
- Literature-grounded property tests (`tests/test_evidence_properties.py`):
  stub byte-stability across turns (anti-drift, AgentFold), negation/fine-fact
  survival through mask→expand (Telegraph English fine_facts), no referential
  dangling by construction (arXiv:2608.04569).

### Phase 2 — Immutable raw store — DONE (merged into Phase 1 by design)
The store IS the raw evidence layer: originals are never rewritten; expansion
is byte-exact and hash-checked (`test_expand_roundtrip_exact`,
tamper test). A typed ResearchIR (Fact/Claim/Conflict/…) remains future work.

### Phase 3 — Protected exact channel — DONE (minimal)
`say(..., protected=True)` renders as `<CONSTRAINT>` and is never masked.
Covered by unit + benchmark assertions (constraint retention under full
compaction). Security-policy separation beyond the protected flag is future work.

### Phase 4 — Checkpoint/delta journaling — DONE
`rcc/journal.py`: append-only JSONL (meta header + say/observe/atom deltas +
full-state checkpoints). `restore()` = latest checkpoint (or meta) + replayed
deltas; `upto_seq` rollback; `compact()` atomically folds history into a
single checkpoint. Replay divergence detected via deterministic obs-ID
assertion. Whole-state save/load kept as the simple path on top.

### Phase 5 — Adaptive compact scratch (DIRECT/DRAFT/SYMBOLIC/EXPERT/FULL) — IN_PROGRESS (step 1 DONE)
**Step 1 — symbolic machine state (RIR/1): DONE.**
`rcc/scratch.py`: append-only typed atoms `F/Q/N/C` with store-backed source
refs and confidence; rendered as the LAST section of compiled context
(`<SCRATCH format=RIR/1>`) so every update is a pure suffix append and the
prompt prefix stays byte-stable (cache-friendly, anti-drift).
Exactness guard enforced at `add()`: every numeric token must appear verbatim
in cited sources (resolved through the hash-verified store) — fabricated or
drifted digits raise `ScratchError`; negations survive as plain text.
Persisted in session save/load. 12 dedicated tests.
Format cost micro-benchmark (`bench/formats.py`, same content, 6 formats):
RIR/1 cheapest on both counters (approx 365 tok vs prose 515; o200k 216 vs
prose 253; JSON worst at +34% over prose under o200k — consistent with
arXiv:2605.29676). All formats verified to carry critical values verbatim.
**Step 2 — mode router: DONE.**
`rcc/router.py`: deterministic signal-based selection — FULL (conflicts /
2+ failures / suspected injection) > EXPERT (8+ observations) > SYMBOLIC
(3+ atoms) > DRAFT > DIRECT. Mode controls visibility only; atoms always
remain in state, journal and timeline. Feature-flagged: `router_enabled`.

**Step 3 — model-assisted distillation: interface DONE, live compressor pending.**
`rcc/gate.py`: `Compressor` protocol + `NullCompressor` (deterministic
default) + `SafeCompressor` (any failure → None → verbatim fallback). An
actual model-backed compressor requires a live provider and stays behind the
Phase-7 gate by construction.

### Phase 6 — Dictionary encoding for structured outputs — DONE
`rcc/dictenc.py`: lossless JSON-object dictionary codec — schema stated once,
objects become positional rows; exact type round-trip (strings stay strings,
ints stay ints); declines non-conforming or non-shrinking payloads;
fail-open at render time (`Policy.encode_jsonl`, off by default).
Measured (o200k): 20-row JSONL tool dump 400 → 310 tokens (−22%), byte-exact decode.

### Phase 7 — Query-aware break-even gate — DONE (deterministic core)
`rcc/gate.py`: per-turn amortised comparison `S·N + q·N·(s+c) < D·N·(1+pen)`;
wired via `Policy.gate="window"|"breakeven"` with conservative defaults
(q=0.05, horizon=20). Boundary tests: monotone in q, never masks hot data,
zero-horizon keeps verbatim. Calibration of q/N against measured workloads:
future measurement task.

### Phase 8 — Experimental machine dialects — MERGED INTO Phase 5
(Structured symbolic scratch IS the first machine dialect; generic
tokenizer-level dialect experiments remain out of scope.)

## 3. Measured results

Two suites: `approx` (rcc-approx-1, deterministic) and `exact-o200k_base`
(tiktoken). Raw data: `.rcc_bench/results.json` (includes per-turn series and
compile latency).

### Exact tokenizer (o200k_base)

| metric | baseline | compiled | delta |
|---|---|---|---|
| long_research peak active | 8,456 | 3,272 | **−61%** |
| long_research total (all turns) | 165,098 | 95,084 | −43% |
| repeated_sql peak active | 3,588 | 774 | **−78%** |
| exact_facts peak / fact recall | 6,861 | 2,682 | −61% / 100% recoverable |
| constraints peak / retained | 7,228 | 2,394 | −67% / verbatim |
| injection quarantine | — | pass | no promotion |

### Scaling shape (cumulative spend, long_research, o200k)

Savings grow with horizon as the recent-window overhead amortises:
25% turns → 0%, 50% → 9%, 75% → 29%, 100% → **42%** (repeated_sql: 69%).
Matches OpenHands' reported linear-vs-quadratic scaling behaviour.

### Overheads

- Stub cost: 126–359 approx-tokens per scenario (~2–9% of gross savings).
- Compile latency: ~1 ms/call in-process; negligible vs any LLM round-trip.
- Compressor LLM calls: **zero** by design (vs OpenHands condenser, which pays
  a summarizer call + cache invalidation).

## 3b. Observability

`ResearchSession.timeline()` exposes a live per-turn audit record
(inline/masked_stub/failopen_inline per observation, protected flags, token
counts) so an operator can see what the model sees at every step, not only at
the end. CompiledContext.metrics carries the same breakdown programmatically.


## 4. Validation

- `uv run pytest` → **93 passed** (unit: tokens/store/session; exact-tokenizer;
  literature-grounded properties; security: injection, cross-run,
  schema-version, tamper; integration: benchmark gates). Environment note:
  this count requires tiktoken; without it, 85 pass and 2 modules skip
  (`test_real_tokenizer`, exact-tokenizer parts of the bench integration).
  Test counts are environment-dependent and must not be quoted as universal.
- `uvx ruff check rcc bench tests` → 7 style findings in legacy files as of
  2026-08-22 (E402/E741/E702); core behaviour unaffected.
- Known limitation of the evaluation: "task success" is proxied by deterministic
  checks (fact recoverability, constraint retention, injection quarantine),
  not end-to-end LLM task quality. That requires a live-model harness.

## 5. Risks / limitations

- Exact counts use o200k_base; other providers' tokenizers differ (direction of
  savings verified identical on both counters, magnitudes vary ±25%).
- Whole-state save/load rewrites one JSON file; fine for benchmark scale,
  not yet an append-only auditable journal (Phase 4).
- Stub format is plain text; no model has been prompted with it, so downstream
  comprehension of `[OBS …]` references is unverified with a live LLM.
- No eviction/TTL in RawStore: storage grows monotonically within a run.
- Injection defence is structural (wrapping), not semantic; a consumer LLM can
  still be fooled by content *inside* wrappers unless the system prompt forbids it.
- Scenarios are synthetic and bounded; real trajectories not yet ingested.

## 6. Native-representation roadmap (user goal: "reasoning not in words")

The end-state the project aims at is model-native reasoning state — vectors /
latent tokens instead of prose. Honest status and evidence:

| Direction | What it is | Status here | Why deferred |
|---|---|---|---|
| ICAE (arXiv:2307.06945, ICLR'24) | LoRA encoder → memory-slot embeddings; 4× compression on Llama; decoder = frozen LLM | NOT_STARTED | Requires training pipeline + white-box access to embeddings/forward hooks |
| Gist tokens (arXiv:2304.08467) | Learned prompt compression into KV activations | NOT_STARTED | Same + tuned-model coupling |
| Coconut (arXiv:2412.06769) | Reasoning in continuous latent space | NOT_STARTED | Architecture-level; curriculum training |
| AGCLR (arXiv:2606.07720) | Shows Coconut loses intermediate facts across passes ("concept bottleneck"); needs persistent gated memory | Evidence only | Confirms even latent reasoning needs an external persistent store — i.e. RCC's substrate is a prerequisite, not a rival |
| COCOM / xRAG / SeleCom / PISCO | Soft-compression RAG via context embeddings | NOT_STARTED | SeleCom (2602.15856): full-compression underperforms non-compressed RAG; query-conditioned selection required |

Sequencing rationale: every latent method still needs (a) an immutable
recovery path for exact facts, (b) stable references for auditability,
(c) measurement harness — exactly Phases 0–3 built here. RIR/1 is the
discrete-token approximation of the same idea: machine state over prose.
When white-box access becomes available, atoms map naturally onto slot
embeddings (one embedding per atom ≈ typed soft token), preserving the
provenance/exactness semantics.

## 8. Next highest-ROI action

All deterministic layers are now implemented and tested (93 tests). The
largest remaining unknown cannot be closed offline: **live-model evaluation**
— do downstream LLMs actually answer correctly over `[OBS …]` stubs + RIR/1
blocks, and how does mode routing affect answer quality? Evidence: every
risk in §5 that remains open ("no model has been prompted with it") blocks
the same thing. Recommended shape: small harness replaying the existing five
scenarios through one provider with baseline vs compiled contexts, scoring
exact-fact recall / citation presence / constraint adherence per run.

**Update (2026-08-25): harness DONE** — `evals/` implements exactly that shape,
provider-neutral (any OpenAI-compatible endpoint or a deterministic fixture),
with a bounded EXPAND protocol so compiled-mode facts are reachable the way
RCC intends them to be. Ground-truth vectors + scoring dimensions from the
support-lane spec (`GROUND_TRUTH_SPEC.md`) are incorporated: EF/CIT/CON/INJ
dimensions, explicit expansion channel per result (tool/closed, never compared
cross-channel), control-pair marking for the byte-identical injection case,
hard call budget with partial-result isolation, versioned results with policy/
tokenizer/model/context-hash/raw-response records. Scope honesty: the bench
scenarios do not exercise RIR/1 or the router, so claims are narrowed to the
one focused eval-local `rir_state` case (minimal `<SCRATCH>` comprehension
probe under EXPERT routing). Fixture-tested: parity at −62% context tokens on
the tool channel; honest-unavailability on the closed channel. See
`LIVE_EVAL.md`. Remaining open item: one real provider run — no provider was
already configured in this environment (no API keys, no local chat-model
service), so it stays a documented, single-command follow-up rather than a
code task.

