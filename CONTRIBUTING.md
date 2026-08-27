# Development workflow

> This repository is archived as a research record. The workflow below is kept
> for reproducibility; no active roadmap is maintained.

The repository stores product state and evidence; an external tool may schedule
work, but the repository does not run its own agent queue or session loop.

## Task contract

Before implementation, record a task ID, goal, scope, dependencies, acceptance
criteria, expected tests and artifacts, stop conditions, and integration owner.
Keep the contract small enough to review as one outcome.

## Isolated implementation

Use one branch and one worktree per implementation task, with one owner. Avoid
unrelated files. Commit every valuable state, including an explicit WIP/recovery
commit before risky changes; a stash is not the sole recovery mechanism.

## Evidence and integration

Completion evidence consists of the commit, reviewed diff, exact test command
and result, and required artifacts. One designated integrator compares competing
implementations, selects commits or hunks, runs integration tests, and updates
the canonical branch. Workers do not merge concurrently into that branch.

Use bounded retries. After repeated failure, commit the recoverable state and
record the actual blocker instead of continuing an unbounded repair loop.

## Artifacts and cleanup

Do not commit large generated outputs by default. Preserve their command,
configuration, provenance and hash; classify them as retained evidence,
archived, reproducible, invalid, or discardable. A finished task leaves its
worktree clean, and the integrator removes temporary worktrees, branches, locks
and claims only after useful work has a durable Git reference.
