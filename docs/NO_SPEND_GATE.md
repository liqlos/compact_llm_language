# No-Spend Integrity Gate (2026-08-24, fail-closed v2)

Bounded, deterministic, zero-spend gate that decides whether the retained
latent-recurrence evidence justifies ANY further GPU spend. It implements
requirement R002/R005 of the 2026-08-24 requirements brief: a fail-closed
READY/NOT_READY verdict BEFORE any remote 4B spend.

Schema version 2 repairs every proven READY fail-open of the initial gate:
readiness is now a per-artifact RELATIONAL JOIN, not a set of global counts,
and every invalid evidence shape maps to an explicit blocker or invalid
status instead of passing silently.

## Command

From the repository root (inputs are private, git-ignored copies under
`.rcc_work/`):

```bash
# full mode — CPU-only; also executes the proof-test regressions
uv run python -m latent_lab.bench.no_spend_gate

# hardware-free dry run — hashing + metadata + report schema only,
# no tensor payload loading, no proof tests
uv run python -m latent_lab.bench.no_spend_gate --dry-run

# optional flags
--skip-proof-tests     # skip regression execution in full mode
--no-telemetry         # do not write telemetry_timestamp.json
```

## Exit semantics

| code | meaning |
|---|---|
| `0` | READY — every mandatory prerequisite PROVEN |
| `1` | NOT_READY — evidence-backed negative verdict; see `blockers` |
| `2` | execution error — missing inputs, output/input overlap, mutated or unreadable sources; no verdict inferable |

Before scanning anything the CLI rejects an output directory that overlaps
either input root (equal, nested either way). A bounded streaming SHA-256
fingerprint of both input trees is taken before and after the scan and
re-checked after outputs are written; any disagreement aborts with exit
code 2 and proves the read-only guarantee per run.

## Outputs (`--out`, default `.rcc_work/no_spend_gate_20260824/`)

- `artifact_inventory.json` — every scanned file: path/kind/size/SHA-256,
  duplicate-content groups, hardlink groups. Sorted, versioned.
- `artifact_verdicts.json` — per-run report validation, checkpoint
  classification, eval rescoring eligibility, 4B quarantine analysis,
  proof-test outcomes.
- `gate_verdict.json` — THE canonical machine-readable verdict:
  prerequisites (PROVEN/FAILED/UNPROVEN), blockers with codes and the
  smallest next executable action, counts, digests of the two JSONs above.
- `GATE_REPORT.md` — human-readable rendering of the same evidence.
- `proof_tests.log` (full mode) and `telemetry_timestamp.json` (optional)
  — non-canonical companions.

**Determinism:** rerunning against unchanged inputs reproduces all four
canonical files byte-for-byte. They contain no wall-clock value; JSON is
emitted with `sort_keys` + strict parsing (non-finite floats become tagged
strings, never `NaN` tokens).

## What READY requires (all PROVEN, else NOT_READY)

1. `inventory_hashed_complete` — every discovered file hashed.
2. `train_reports_schema_identity_and_pins` — model id, pinned immutable
   revision, suite hash, mode/interval/K/seed/steps, trainable precision
   claiming fp32. Missing fields AND schema violations (float/bool/negative
   steps, non-finite metrics, malformed history entries, non-fp32 precision)
   are blockers — records are never skipped-and-selected around; strict JSON
   rejects NaN/Infinity literals in every retained report.
3. `checkpoints_strict_loadable_identity_bound` — every retained candidate
   checkpoint loads through the project's identity-bound bundle loader,
   carries a unique binding (no byte-identical duplicate payloads), and its
   ACTUAL loaded trainables are fp32 — correct metadata over a bf16 payload
   is classified invalid, not loadable.
4. `retained_evals_rescored_with_corrected_scorer` — each record carries
   EXACTLY ONE lossless raw representation: fully finite aligned per-candidate
   scores or a full unique candidate ranking. Derived-only records are
   `NON_RESCORABLE_MISSING_RAW_PREDICTION`; top-score ties, duplicated
   candidates/examples, conflicting representations, partial/unknown
   rankings and non-finite scores are explicit `INVALID_*` statuses — never
   uncaught exceptions, never silently scored. Unreadable/malformed eval
   files (incl. NaN literals) never count as rescored.
5. `checkpoint_selection_uses_corrected_metric` — best-step selection
   reproducible from histories produced under the corrected scorer
   (`config.scorer == "corrected-gold-aware-v1"`); stored historical
   `best_val_acc`/`best_step` are never trusted.
6. `runtime_integrity_regressions_pass` — adapter strict metadata +
   roundtrip, fp32 trainables over bf16 backbone, non-finite rejection,
   cached-recurrence equivalence, gold-position scoring invariance.
7. `invalid_4b_quarantined_not_live` — rejected evidence is nonempty and
   complete (marker + at least one rejected report/checkpoint pair), the
   live 4B tree contains ZERO known-invalid artifacts (corrupt/unbound
   payloads, NaN/degenerate/noncanonical reports) and ZERO byte-identical
   copies of quarantined files — one differing file no longer masks
   duplication.
8. `retained_2b_relational_join_complete` — every retained loadable 2B run
   joins one-to-one: schema-valid identity-bound report <-> fp32 uniquely
   bound checkpoint <-> valid current-suite corrected raw evals bound to
   that run covering BOTH mandatory splits (`test_id` AND `test_ood`),
   with identical model/revision across all three. Orphan reports,
   checkpoints or eval files, and evals whose adapter binds to no retained
   run, block this prerequisite.

Every non-PROVEN prerequisite also emits an explanatory blocker — a
NOT_READY verdict can never be silent.

## Trust boundary

The gate is read-only over an isolated private COPY of historical
`.rcc_work` evidence. `.pt` payloads deserialize only via
`torch.load(weights_only=True)` plus the project loader
(`latent_lab.train.checkpointing.load_adapter_bundle`). Nothing is modified,
moved, deleted, executed beyond deserialization, or sent off-device. No
cloud, GPU, training, or paid service is touched by design.

## Result on this evidence set (2026-08-24)

NOT_READY — see the committed gate code for the checks and the uncommitted
`.rcc_work/no_spend_gate_20260824/GATE_REPORT.md` for the full blocker list
(legacy-unbound bf16 checkpoints, non-fp32 trainables, unrescorable
derived-only eval records, missing trainable-precision fields, poisoned
selection provenance, live-tree duplication of the rejected NaN batch,
incomplete ID/OOD eval coverage). The gate was NOT weakened to produce
READY; a positive-control test inside the suite proves the READY path fires
when every prerequisite genuinely holds.

An independent negative audit (nine repro cases: bf16 bundles inside
identity-bound metadata, wrong eval suite hashes, empty/malformed/nonfinite
records, tied/duplicated/conflicting rankings, uncovered checkpoints and
splits, orphan checkpoints, masked live 4B duplication) found eight READY
fail-opens in the initial gate; every one now resolves to NOT_READY with its
specific blocker code (`tests/test_no_spend_gate.py::TestFailClosedRepairs`).
