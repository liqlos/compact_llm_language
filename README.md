# RCC — Research Context Compiler

**Strategic direction (2026-08-22):** RCC is pivoting to latent cognition on
open-weight Qwen (`Qwen/Qwen3.8-27B` release target; Qwen3.5 proxies).
See `docs/research-context-compiler/VISION.md`, `ADR-001`,
`LATENT_ROADMAP.md`, `EVALUATION_PROTOCOL.md`, `BLOCKERS.md`.

The existing package is the **evidence substrate**: it is storage-lossless,
deterministic and auditable — but no live model has ever been run against
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
  weaker claims — see VISION.md; both unmeasured with a live model.)
- **Prefix-stable**: stubs are written once and never rewritten; the RIR/1
  block only appends → prompt-prefix caching stays valid across turns.
- **Fine-fact guard**: numeric tokens in machine-state atoms must appear
  verbatim in their cited sources or `Scratch.add()` raises — fabricated or
  drifted digits cannot enter machine state.
- **Fail-open availability / fail-closed integrity**: masking requires a
  verified store object (else verbatim text or an honest
  `<OBSERVATION_UNAVAILABLE/>` marker); tampered objects raise.
- **Run isolation**: references carry run IDs; cross-run resolution raises.
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
uv sync && uv run pytest          # full unit, security, property, integration,
                                  # runtime, gate, and live-eval suite
uvx ruff check rcc bench tests evals    # lint
uv run python -m bench.run_bench --exact --json .rcc_bench/results.json
uv run python -m evals.run_live_eval --provider fake   # live-eval harness (offline fixture)
```

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

Exit codes: `0` READY · `1` NOT_READY · `2` execution error (bad invocation,
unreadable inputs, inputs mutated mid-run, unwritable outputs). `--out` may
never overlap an input root. Every known fail-open is a blocker; negative
controls for each audited fail-open are pinned in
`tests/test_no_spend_gate.py`.
Canonical outputs under `.rcc_work/no_spend_gate_20260824/` are byte-stable
across reruns against unchanged inputs and contain no wall-clock value (the
only timestamp lives in the separate `telemetry_timestamp.json`).
Current verdict and blocker codes: see `.rcc_work/no_spend_gate_20260824/GATE_REPORT.md`
(uncommitted evidence) and `docs/NO_SPEND_GATE.md`.

Live-model evaluation (`evals/`) replays the benchmark scenarios baseline vs
compiled, scores fact recall, citations and constraint adherence over real
model answers through any OpenAI-compatible endpoint, and supports a bounded
explicit expansion channel. The deterministic fake provider is the offline
smoke path; see `docs/research-context-compiler/LIVE_EVAL.md`.

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
| Latent pipeline plumbing (`latent_lab/`) | UNIT_VERIFIED (mock; no-decode loop contract enforced) |
| Real Qwen hybrid runtime control | MODEL_VERIFIED on Qwen3.5-0.8B proxy — cache snapshot/restore exact, K-step recurrence with zero lm_head calls (`latent_lab/bench/results/state_probe_*.json`) |
| MLX off-vocabulary embedding path | MEASURED: exact-vocab embeds safe/fast; off-manifold inputs slow ~3.5–5×; hybrid prefix-cache trim unsupported (`results/mlx_soft_embedding_probe.json`) |
| Localized recurrence quality / speedup | MODEL_MEASURED @ Qwen3.5-2B behavioral gate | suite v2 (7 families): latent E(K=4) LoRA 0.43–0.52 test-ID vs chance 0.07 vs K=0 control 0.375; partial OOD length-gen 0.29–0.49; causal ablations seed-dependent (`latent_lab/bench/GATE_SUMMARY.md`, `.rcc_work/GATE_SUMMARY.md`) |
| Textual baselines @2B | REJECTED_BY_EVIDENCE (capability limit) | direct/thinking/capped all 0.0% acc, up to 100% NON_TERMINATION incl. 1024-token budget |
| MLX internal-recurrence path | MEASURED: bit-exact composition, speed BLOCKED without compile | `latent_lab/bench/mlx_internal_recurrence_probe.py` |
| Qwen3.8-27B speedup ≥2× | NOT MEASURED | gated by 4B step (MODEL_CAPABILITY_LIMIT at 2B) |
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
