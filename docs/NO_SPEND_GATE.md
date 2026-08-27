# No-Spend Integrity Gate (2026-08-24)

> **Current R1 authority (2026-08-27):**
> `artifacts/milestone_r1_verdict.json` records
> `PAID_SPEND_NOT_AUTHORIZED`; a valid model experiment is pending. Historical
> READY receipts documented below are preserved audit records, not current
> authorization. Paid or remote execution requires a fresh machine verdict
> that explicitly changes `paid_spend_authorized` from false.

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
| `2` | execution error — bad invocation (`--out` overlapping an input root), missing/unreadable inputs, inputs modified mid-run, unwritable outputs, or crash; NO verdict is inferable |

## Integrity guarantees (fail-closed)

- **Strict JSON everywhere.** Reports and eval files are parsed with
  NaN/Infinity literals rejected; a malformed file is an explicit blocker,
  never skipped, truncated, or silently coerced.
- **One unambiguous raw representation per record.** Conflicting
  `candidate_scores` + `ranked_candidates` are INVALID; ranked lists must
  be full unique permutations of the candidate set (duplicates / unknown /
  partial entries are INVALID, never uncaught exceptions); scores must be
  finite real numbers (NaN/Inf never sink to `-inf`); a top-score tie
  between different candidates is ambiguous instead of being decided by
  array position, so candidate permutation cannot change correctness;
  record flags/status are preserved through aggregation and invalid
  records can never yield `RESCORED_CORRECTED`.
- **Payload-verified FP32 trainables.** Retained checkpoints must load
  through the identity-bound bundle loader AND their returned payload
  tensors must be floating-point-fp32; report strings alone prove nothing.
  A bf16 bundle classifies `non-fp32-payload` and blocks READY.
- **Symmetric discovery.** Orphan checkpoints (no owning discovered run),
  orphan eval files (adapter resolving to no run), byte-duplicate
  checkpoints bound to more than one run directory, and unreadable
  artifacts are explicit blockers — discovery does not start and stop at
  reports.
- **Per-checkpoint eval joins.** Every retained loadable checkpoint (2B
  AND live 4B) needs valid rescored raw-score evidence bound to it by
  adapter path + exact model id + pinned revision + current suite hash,
  covering every required behavioral-v3 split (`test_id`,
  `test_ood_length`, `test_ood_semantic`, and untouched `final_test`) over
  the COMPLETE preregistered example set of each declared split; records
  whose `ex_id` lies outside the preregistered membership of the file's
  declared split are explicit blockers. One unrelated, one-split, or
  partial eval proves nothing globally.
- **Uniform retained prerequisites.** Report schema validity (including
  `trainable_precision`), suite-hash pinning, corrected-metric selection,
  and full eval coverage apply to every retained run — a live 4B
  checkpoint with an incomplete report or no eval evidence forces
  NOT_READY exactly like a 2B one.
- **Quarantine completeness.** The rejected 4B batch must be nonempty,
  markered, and fully contained: ANY known-invalid or byte-duplicate 4B
  artifact left in any live tree blocks READY even when sibling files
  differ; a marker-only empty quarantine proves nothing.
- **Input immutability proof.** Bounded streaming SHA-256 Merkle
  fingerprints of both input roots are recorded before and after gating;
  any change aborts with exit 2 and embeds the fingerprints in
  `gate_verdict.json`. `--out` equal to, inside, or containing either
  input root is rejected before anything is written, so the gate can
  never self-inventory its outputs.

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

1. `inventory_hashed_complete` — every discovered file hashed; streaming
   input fingerprints taken before and after the gate and matched.
2. `train_reports_schema_identity_and_pins` — nonempty model id, pinned
   immutable revision, current suite hash, mode/interval/K/seed/steps
   (steps a non-negative integer, never bool), finite `best_val_acc` /
   `final_train_loss`, per-entry integer steps and finite accuracies in
   `val_history`, trainable precision present. Missing or non-finite
   fields are blockers, never defaults.
3. `checkpoints_strict_loadable_identity_bound_fp32_payload` — every
   retained candidate checkpoint loads through the project's identity-bound
   bundle loader with fp32 verified from the payload tensors; no orphan,
   no ambiguous duplicate binding.
4. `retained_evals_rescored_with_corrected_scorer` — every retained
   loadable checkpoint has valid `latent_eval.v3` raw per-token evidence for
   all four required splits, exactly identity-bound; records holding only
   derived `correct`/`rank_of_gold` are marked
   `IRRECOVERABLE_LEGACY_SCORER`; malformed/empty/
   invalid-score/conflicting eval files are `EVAL_FILE_INVALID`.
5. `checkpoint_selection_uses_corrected_metric` — best-step selection is
   reproduced from raw validation records with canonical `latent_eval.v3`;
   stored historical `best_val_acc`/`best_step` are never trusted.
6. `runtime_integrity_regressions_pass` — adapter strict metadata +
   roundtrip, fp32 trainables over bf16 backbone, non-finite rejection,
   cached-recurrence equivalence, gold-position scoring invariance.
7. `invalid_4b_quarantined_not_live` — rejected batch nonempty + markered;
   nothing byte-duplicating it remains live, no known-invalid artifact in
   any live tree, no marker-only empty quarantine.

A dry run leaves 3/4/6 UNPROVEN by construction, so dry runs can never
emit READY.

## Trust boundary

The gate is read-only over an isolated private COPY of historical
`.rcc_work` evidence. `.pt` payloads deserialize only via
`torch.load(weights_only=True)` plus the project loader
(`latent_lab.train.checkpointing.load_adapter_bundle`). Nothing is modified,
moved, deleted, executed beyond deserialization, or sent off-device. No
cloud, GPU, training, or paid service is touched by design.

## Historical result on the pre-fix evidence set (2026-08-24)

The snapshot evaluated on 2026-08-24 was NOT_READY — evidence-backed, never
hardcoded. This section is provenance, not the current spend authorization. An
in-suite positive control
(`tests/test_no_spend_gate.py::...READY_positive_control_exit0_path`)
proves READY fires when every prerequisite genuinely holds, and the
verdict for any given tree is recomputed from its artifacts on every run.
On the retained snapshot the gate reports blockers such as
legacy-unbound bf16 checkpoints, unrescorable derived-only eval records
(no raw per-candidate scores), missing trainable-precision fields,
poisoned selection provenance, live-tree duplication of the rejected NaN
batch, and missing per-split eval coverage/identity bindings — see the
generated report for that run's exact list. The gate was NOT
weakened to produce READY; 45 negative controls (audit-reproduced
fail-opens) now fail closed and each is pinned by a test that fails on
the pre-fix commit `3774569`.

Later retained receipts are `.rcc_work/rcc.pre_spend.v2.json` (READY at its
recorded revision) and `.rcc_work/rcc.canary_attempt.v1.json` (fail-stop before
training/model contact). Neither is a standing authorization: rerun and inspect
the gate against the exact current inputs before any new spend.
