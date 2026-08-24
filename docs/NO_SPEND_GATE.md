# No-Spend Integrity Gate (2026-08-24)

Bounded, deterministic, zero-spend gate that decides whether the retained
latent-recurrence evidence justifies ANY further GPU spend. It implements
requirement R002/R005 of the 2026-08-24 requirements brief: a fail-closed
READY/NOT_READY verdict BEFORE any remote 4B spend.

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
| `2` | execution error — missing inputs or crash; no verdict inferable |

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
   revision, suite hash, mode/interval/K/seed/steps, trainable precision.
   Missing fields are blockers, never defaults.
3. `checkpoints_strict_loadable_identity_bound` — every retained candidate
   checkpoint loads through the project's identity-bound bundle loader.
4. `retained_evals_rescored_with_corrected_scorer` — raw per-candidate
   scores were retained, so corrected gold-aware rescoring actually ran.
   Records holding only derived `correct`/`rank_of_gold` are marked
   `NON_RESCORABLE_MISSING_RAW_PREDICTION`, never relabelled as a rescore.
5. `checkpoint_selection_uses_corrected_metric` — best-step selection
   reproducible from histories produced under the corrected scorer
   (`config.scorer == "corrected-gold-aware-v1"`); stored historical
   `best_val_acc`/`best_step` are never trusted.
6. `runtime_integrity_regressions_pass` — adapter strict metadata +
   roundtrip, fp32 trainables over bf16 backbone, non-finite rejection,
   cached-recurrence equivalence, gold-position scoring invariance.
7. `invalid_4b_quarantined_not_live` — rejected NaN batch fully contained
   in `_rejected_nan_batch/` with marker, nothing invalid left live, and
   no byte-duplication between live `runs/` and the quarantine tree.

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
(legacy-unbound bf16 checkpoints, unrescorable derived-only eval records,
missing trainable-precision fields, poisoned selection provenance, live-tree
duplication of the rejected NaN batch). The gate was NOT weakened to produce
READY; a positive-control test inside the suite proves the READY path fires
when every prerequisite genuinely holds.
