# RCC — Research Context Compiler

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

## Guarantees

- **Lossless**: masked ≠ lost. `expand()` returns the full original bytes,
  SHA-256 verified on every read. No referential dangling by construction.
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
uv sync && uv run pytest          # 93 tests: unit, security, properties, integration
uvx ruff check rcc bench tests    # lint
uv run python -m bench.run_bench --exact --json .rcc_bench/results.json
```

Measured (o200k_base, exact tokenizer): peak active context −61…−78% on
representative scenarios; fact recall via expansion 100%; cumulative spend
−42% over a full long-research run (grows with horizon). Details:
`docs/research-context-compiler/IMPLEMENTATION_PLAN.md`.

## Status & layout

| Layer | State |
|---|---|
| 0 Measurement / baseline | DONE |
| 1 Stable refs, dedup, safe masking | DONE |
| 2 Immutable raw store | DONE |
| 3 Protected exact channel | DONE |
| 4 Checkpoint + delta journal | DONE |
| 5.1 Symbolic machine state (RIR/1) | DONE |
| 5.2 Mode router (DIRECT…FULL) | DONE |
| 5.3 Compressor plug point + gate | interface DONE; live compressor pending provider |
| 6 Dictionary encoding (JSONL) | DONE |
| 7 Break-even gate | DONE (deterministic core; q/N calibration = future measurement) |
| Live-model evaluation | NOT_STARTED — next |

```
rcc/            core library (tokens, store, scratch, session, journal,
                router, dictenc, gate)
bench/          deterministic scenarios, harness, format micro-benchmark
tests/          unit / security / property / integration suites
docs/research-context-compiler/
                implementation plan with evidence review and measurements
```

## Research grounding

AgentFold (2510.24699), ACM (2607.23809), referential dangling (2608.04569),
Telegraph English (2605.04426), Notation Matters (2605.29676), OpenHands SDK
(2511.03690), LLMLingua (EMNLP'23), rate-distortion limits (NeurIPS'24);
latent-direction roadmap: ICAE (2307.06945), Gist (2304.08467),
Coconut (2412.06769) — see plan §0 and §7.
