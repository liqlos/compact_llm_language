# Parallel-work consolidation — 2026-08-25

> **Historical snapshot:** paths, counts and aggregate `.rcc_work` digest below
> describe the 2026-08-25 layout. The R1 audit preserved all content but moved
> eight duplicate non-finite 4B live run directories into recoverable
> quarantine. Current per-file hashes and dispositions are in
> `artifacts/ARTIFACT_CLASSIFICATION.json`; spend remains
> `PAID_SPEND_NOT_AUTHORIZED` per `artifacts/milestone_r1_verdict.json`.

## Recovery boundary

- Canonical branch before consolidation: `main`.
- Consolidation base: `2ac14da2d0cacca2149230bd98adaf1baead1fbe`.
- Safety tag: `pre-consolidation-20260825T172551Z`.
- Integration branch: `integration/consolidation`.
- One dirty worktree was preserved as recovery commit
  `87ab82e75b7c7e9d122dc69c9ccd90245eb3fbad` on
  `archive/pre-consolidation/longrun-3c120aec-dirty`.
- Twelve archive refs preserve all 19 commits that were unreachable during the
  initial audit. A post-recovery `git fsck --full --no-reflogs --unreachable`
  reported no unreachable objects.

Initial inventory: 7 worktrees, 30 local branches, 1 dirty worktree, no stashes,
and 12 dangling sequences containing 19 commits.

## Workstream decisions

| Source | Purpose | State | Tests/audit | Verdict |
|---|---|---|---|---|
| `main` through `2ac14da` | selected fail-stop runtime, integrity gate, capped Vast provisioner | clean, 26 commits ahead of `origin/main` | baseline `488 passed, 1 skipped`; wheel built | KEEP |
| `ao/compact_llm_language-2/live-eval-harness` | provider-neutral baseline-vs-compiled evaluation | three independent commits | branch `134 passed`; integrated suite `529 passed, 1 skipped`; fake run `13/24` calls | KEEP |
| `codex/oc-vast-docs-20260824` | owner-approved execution-environment documentation | one independent commit | documentation diff reviewed | KEEP |
| runtime kernel/optimizer alias/state/topology branches | competing per-step rollback engines | divergent from `339478d` | representative alternate suites require the rejected snapshot/rollback API | SUPERSEDED |
| `codex/oc-no-spend-gate-20260824` | earlier gate implementation | divergent | its regression cases exist in the canonical recipe-bound suite | SUPERSEDED |
| `codex/oc-no-spend-gate-b-20260824` | broad alternate gate rewrite | divergent | historical adversarial repro failed; current API incompatible | INVALID |
| AO root/orchestrator branches | unused worker slots at `origin/main` | clean, no unique commits | no diff | OBSOLETE |
| interrupted Longrun worktree and recovered stash/dangling snapshots | forensic/WIP runtime variants | safely referenced | superseded by canonical fail-stop runtime | ARCHIVE_ONLY |

## Integrated work

- Three live-eval commits were selectively cherry-picked with README conflicts
  resolved in favor of current maturity claims and no-spend documentation.
- The final harness adds explicit tool/closed expansion channels, ground-truth
  scoring, hard call budgets, deterministic fake-provider coverage, and a
  focused RIR/router scenario.
- Integration audit found that the new `evals` package was absent from the
  wheel; `358f120` adds it to the Hatch package list. The rebuilt wheel contains
  all six `evals/*.py` modules.
- `820fc98` integrates the Vast/laptop execution-environment decision without
  changing runtime behavior or launching hardware.

## Non-Git artifacts

- `.rcc_work/`: ARCHIVE/KEEP in place; 161 files, approximately 61 MB. It
  contains retained 2B/4B reports, checkpoints, logs, sealed-environment
  contracts, recovery bundles, and canary receipts. Aggregate manifest digest
  (SHA-256 over the sorted per-file SHA-256 manifest):
  `a922465392d109816ed07fa6c68adbde11953cf147057f5b4b5998ae552ffee2`.
- The recorded canary attempt is `FAIL_STOP_PRE_TRAINING`: model/training contact
  did not start, teardown was confirmed, estimated upper-bound cost was USD
  0.03, and provider invoice verification remains outstanding.
- Current read-only Vast inventory check during consolidation: 0 instances and
  0 volumes.
- `.rcc_eval/`, `.rcc_bench/`, and `dist/`: REGENERABLE outputs.
- `.longrun/`: obsolete ignored orchestration state; not part of the canonical
  development process.

## Canonical process

`CONTRIBUTING.md` defines the orchestrator-neutral workflow: explicit task
contract, one branch/worktree/owner per implementation, commit every valuable
state, one integrator, evidence-backed completion, bounded retries, and no
repository-local agent queue. No tracked claims, locks, session loops, worker
registries, or competing schedulers were found.

The process smoke test created an isolated branch/worktree, committed
`CONTRIBUTING.md`, ran `uv run pytest -q tests/test_suite.py` (`28 passed`),
reviewed and cherry-picked the commit, then removed the temporary worktree and
branch.

## Verification before final merge

- Baseline: `uv run pytest` -> `488 passed, 1 skipped`.
- Integrated: `uv run pytest` -> `529 passed, 1 skipped`.
- Offline live eval: successful, 13 calls spent from a hard budget of 24.
- `uv build`: source distribution and wheel built successfully.
- Wheel inspection: `evals` package present after the integration fix.
- Ruff: 163 findings on the pre-consolidation base and 163 after integration;
  the integrated `evals` package and its tests have 0 findings. The historical
  baseline findings were not rewritten as part of consolidation.
- No real-provider live-model evaluation or new GPU experiment was run.
