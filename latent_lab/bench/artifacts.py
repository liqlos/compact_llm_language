"""Run-lock + artifact validation primitives for remote drivers.

Contract (fail closed):
  * Runs serialize through an exclusive lock file (O_CREAT|O_EXCL with
    dead-holder recovery). Two drivers can never interleave writes.
  * A training directory counts as DONE only when its generation verifies:
    run status complete + coherent digest-linked report/checkpoint/manifest
    under the FULL manifest schema, AND the checkpoint is a genuine
    identity-bound v2 bundle bound to the SAME canonical recipe as the
    report and manifest.
  * An eval payload counts as DONE only when it parses as JSON, carries
    status "complete", binds full identity (model/revision/suite/adapter
    digest/split/ablation/K/seed), and carries lossless finite per-candidate
    raw records whose corrected scoring is independently recomputable.
  * A directory or file NAME is never evidence: drivers must supply an
    explicit expected contract (model/revision/suite/seed/label/k/...)
    that is compared against artifact CONTENTS.

CLI:
    python -m latent_lab.bench.artifacts validate-run DIR [expectations]
    python -m latent_lab.bench.artifacts validate-eval FILE [expectations]

Expectation flags (repeatable, all optional but enforced when present):
    --expect-model M --expect-rev R40 --expect-suite SHA256
    --expect-seed N --expect-label L --expect-k K --expect-steps S
    --expect-split SP --expect-ablation AB --expect-digest SHA256
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_SHA256_HEX_LEN = 64
_REV_HEX_LEN = 40


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError, ValueError):
        return False
    except PermissionError:
        return True
    except OSError as e:  # pragma: no cover - errno-based fallback
        return e.errno != errno.ESRCH
    return True


@contextmanager
def acquire_lock(path, *, timeout_s: float = 600.0, poll_s: float = 0.25):
    """Exclusive cross-process lock; raises RuntimeError on contention."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = json.loads(path.read_text() or "{}")
                pid = info.get("pid")
                if isinstance(pid, int) and not _pid_alive(pid):
                    # dead holder: break the stale lock
                    path.unlink()
                    continue
            except (ValueError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"another process holds the run lock {path}; refusing "
                    "to interleave artifact writes")
            time.sleep(poll_s)
    try:
        os.write(fd, json.dumps(
            {"pid": os.getpid(), "command": " ".join(sys.argv),
             "acquired": time.time()}).encode())
        yield
    finally:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# expected contracts
# ---------------------------------------------------------------------------

_EXPECT_FLAGS = {
    "model_id": "--expect-model",
    "revision": "--expect-rev",
    "suite_sha256": "--expect-suite",
    "seed": "--expect-seed",
    "label": "--expect-label",
    "k": "--expect-k",
    "steps": "--expect-steps",
    "split": "--expect-split",
    "ablation": "--expect-ablation",
    "checkpoint_content_digest": "--expect-digest",
}


def _require_exact(actual, expected, field: str, where: str) -> None:
    """Exact-equality binding; a mismatch is a hard rejection."""
    if expected is None:
        return
    if actual != expected:
        raise ValueError(
            f"{where}: {field} mismatch — evidence carries {actual!r} but "
            f"the driver's expected contract requires {expected!r}")


def _check_sha(value, field: str, where: str, *, length: int = _SHA256_HEX_LEN) -> None:
    if not isinstance(value, str) or len(value) != length \
            or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{where}: {field} {value!r} is not pinned "
                         f"{length}-hex")


def _check_revision(value, where: str) -> None:
    _check_sha(value, "identity.revision", where, length=_REV_HEX_LEN)


# ---------------------------------------------------------------------------
# run validation
# ---------------------------------------------------------------------------

def validate_run(run_dir, *, expected: dict | None = None) -> dict:
    """Return the verified manifest or raise; never trust bare existence.

    ``expected`` is the driver's preregistered contract; every provided
    key must match artifact CONTENTS exactly:
      model_id, revision, suite_sha256, seed, label, k, steps,
      checkpoint_content_digest.
    """
    from latent_lab.bench.latent_run import recipe_from_config
    from latent_lab.train.checkpointing import (
        CHECKPOINT_FILE, TRAIN_REPORT_FILE, inspect_adapter_bundle,
        verify_generation,
    )

    adapter_dir = Path(run_dir)
    where = str(adapter_dir)
    manifest = verify_generation(adapter_dir)

    report_path = adapter_dir / TRAIN_REPORT_FILE
    try:
        report = json.loads(report_path.read_text())
    except Exception as e:  # noqa: BLE001 - unreadable report fails closed
        raise ValueError(f"{where}: unreadable train report: {e}") from e
    cfg = report.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: train report has no config object")

    # canonical recipe rebuilt from the report's own config must equal
    # BOTH the report's bound recipe AND the manifest's bound recipe
    suite_sha = manifest["suite_sha256"]
    if report.get("suite_sha256") != suite_sha:
        raise ValueError(
            f"{where}: report suite_sha256 {report.get('suite_sha256')!r} "
            f"!= manifest {suite_sha!r}")
    recipe = recipe_from_config(cfg, suite_sha)
    if report.get("recipe") != recipe:
        raise ValueError(
            f"{where}: report recipe disagrees with the canonical recipe "
            f"of its own config ({recipe['config_sha256']})")
    if manifest.get("recipe") != recipe:
        raise ValueError(
            f"{where}: manifest recipe disagrees with the canonical "
            f"recipe of the report config ({recipe['config_sha256']})")

    ident = manifest["identity"]
    model_id = ident["model_id"]
    revision = ident["revision"]
    if model_id != cfg.get("model"):
        raise ValueError(
            f"{where}: manifest identity.model_id {model_id!r} != report "
            f"config model {cfg.get('model')!r}")
    if revision != cfg.get("revision"):
        raise ValueError(
            f"{where}: manifest identity.revision != report config revision")
    if manifest.get("seed") != cfg.get("seed"):
        raise ValueError(
            f"{where}: manifest seed {manifest.get('seed')!r} != report "
            f"config seed {cfg.get('seed')!r}")
    if manifest.get("label") != cfg.get("label"):
        raise ValueError(
            f"{where}: manifest label {manifest.get('label')!r} != report "
            f"config label {cfg.get('label')!r}")

    # strict bundle verification: the checkpoint must BE a genuine
    # identity-bound v2 bundle for this exact identity + recipe, with a
    # content digest linked into both report and manifest
    ckpt = adapter_dir / CHECKPOINT_FILE
    bundle_meta = inspect_adapter_bundle(ckpt, model_id=model_id,
                                         revision=revision, recipe=recipe)
    content_digest = bundle_meta["content_digest"]
    if content_digest != manifest.get("checkpoint_content_digest"):
        raise ValueError(
            f"{where}: checkpoint content digest {content_digest} != "
            "manifest checkpoint_content_digest")
    if content_digest != report.get("checkpoint_content_digest"):
        raise ValueError(
            f"{where}: checkpoint content digest {content_digest} != "
            "report checkpoint_content_digest")

    if expected:
        _require_exact(model_id, expected.get("model_id"),
                       "identity.model_id", where)
        _require_exact(revision, expected.get("revision"),
                       "identity.revision", where)
        _require_exact(suite_sha, expected.get("suite_sha256"),
                       "suite_sha256", where)
        _require_exact(manifest.get("seed"), expected.get("seed"),
                       "seed", where)
        _require_exact(manifest.get("label"), expected.get("label"),
                       "label", where)
        _require_exact(cfg.get("k"), expected.get("k"), "config.k", where)
        _require_exact(cfg.get("steps"), expected.get("steps"),
                       "config.steps", where)
        _require_exact(content_digest,
                       expected.get("checkpoint_content_digest"),
                       "checkpoint_content_digest", where)
    return manifest


# ---------------------------------------------------------------------------
# eval validation
# ---------------------------------------------------------------------------

_EVAL_IDENTITY_KEYS = ("model_id", "revision", "suite_sha256",
                       "checkpoint_content_digest", "split", "ablation",
                       "k_steps", "seed")


def _bad_number(v) -> bool:
    """True for bools/non-numbers/NaN/±Inf."""
    return isinstance(v, bool) or not isinstance(v, (int, float)) \
        or v != v or v in (float("inf"), float("-inf"))


def _validate_eval_result(where: str, name: str, res) -> None:
    """One eval result block must carry recomputable lossless evidence."""
    if not isinstance(res, dict):
        raise ValueError(f"{where}: results[{name!r}] is not an object")
    acc = res.get("accuracy")
    if _bad_number(acc) or not 0.0 <= float(acc) <= 1.0:
        raise ValueError(
            f"{where}: results[{name!r}].accuracy {acc!r} is not a value "
            "in [0, 1]")
    n = res.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{where}: results[{name!r}].n missing/invalid")
    records = res.get("records")
    if not isinstance(records, list) or len(records) != n:
        raise ValueError(
            f"{where}: results[{name!r}] has no complete raw records "
            f"(n={n}, records={len(records) if isinstance(records, list) else '??'}); "
            "summary-only evidence is rejected")
    for r in records:
        if not isinstance(r, dict):
            raise ValueError(f"{where}: results[{name!r}]: bad record")
        scores = r.get("scores_raw")
        cands = r.get("candidates")
        order = r.get("score_order")
        if not isinstance(scores, list) or not scores \
                or any(_bad_number(s) for s in scores):
            raise ValueError(
                f"{where}: results[{name!r}] record {r.get('ex_id')!r}: "
                "missing/non-finite per-candidate raw scores")
        if not isinstance(cands, list) or len(cands) != len(scores):
            raise ValueError(
                f"{where}: record {r.get('ex_id')!r}: candidates vs "
                "scores length mismatch")
        if not isinstance(order, list) \
                or sorted(order) != list(range(len(scores))):
            raise ValueError(
                f"{where}: record {r.get('ex_id')!r}: score_order is not "
                "a permutation of the candidate set")
    # independently recomputable corrected scoring, right now
    from latent_lab.bench.latent_run import rescore_records
    recomputed = rescore_records(records)
    if abs(recomputed - float(acc)) > 1e-9:
        raise ValueError(
            f"{where}: results[{name!r}].accuracy {acc!r} does not match "
            f"independently recomputed accuracy {recomputed!r}")


def validate_eval(eval_path, *, expected: dict | None = None) -> dict:
    """An eval payload is evidence only if complete + fully identity-bound.

    ``expected`` is the driver's preregistered contract; every provided
    key must match payload CONTENTS exactly:
      model_id, revision, suite_sha256, checkpoint_content_digest, split,
      ablation, k, seed.
    """
    p = Path(eval_path)
    where = str(p)
    try:
        d = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 - unreadable payload fails closed
        raise ValueError(f"{where}: unreadable eval payload: {e}") from e
    if not isinstance(d, dict):
        raise ValueError(f"{where}: payload is not a JSON object")
    if d.get("status") != "complete":
        raise ValueError(f"{where}: status={d.get('status')!r}, "
                         "not 'complete'")
    ident = d.get("identity") or {}
    if not isinstance(ident, dict):
        raise ValueError(f"{where}: no identity block")
    for key in _EVAL_IDENTITY_KEYS:
        v = ident.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"{where}: identity.{key} missing")
    rev = ident["revision"]
    _check_revision(rev, where)
    _check_sha(ident["suite_sha256"], "identity.suite_sha256", where)
    _check_sha(ident["checkpoint_content_digest"],
               "identity.checkpoint_content_digest (adapter digest)", where)
    k_steps = ident["k_steps"]
    if isinstance(k_steps, bool) or not isinstance(k_steps, int) or k_steps < 0:
        raise ValueError(f"{where}: identity.k_steps must be a non-negative int")
    seed = ident["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{where}: identity.seed must be an int")
    split = ident["split"]
    ablation = ident["ablation"]
    results = d.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{where}: no results block")
    for name, res in results.items():
        _validate_eval_result(where, name, res)
    if ablation not in results:
        raise ValueError(
            f"{where}: identity.ablation {ablation!r} has no matching "
            "results entry")

    if expected:
        _require_exact(ident["model_id"], expected.get("model_id"),
                       "identity.model_id", where)
        _require_exact(rev, expected.get("revision"),
                       "identity.revision", where)
        _require_exact(ident["suite_sha256"], expected.get("suite_sha256"),
                       "identity.suite_sha256", where)
        _require_exact(ident["checkpoint_content_digest"],
                       expected.get("checkpoint_content_digest"),
                       "identity.checkpoint_content_digest", where)
        _require_exact(split, expected.get("split"), "identity.split", where)
        _require_exact(ablation, expected.get("ablation"),
                       "identity.ablation", where)
        _require_exact(k_steps, expected.get("k"), "identity.k_steps", where)
        _require_exact(seed, expected.get("seed"), "identity.seed", where)
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_expectations(argv: list) -> tuple[list, dict]:
    """Split leading subcommand args from trailing expectation flags."""
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    for dest, flag in _EXPECT_FLAGS.items():
        ap.add_argument(flag, dest=dest, default=None)
    known, rest = ap.parse_known_args(argv)
    unknown = [a for a in rest if a.startswith("--")]
    if unknown:
        raise SystemExit(f"unknown expectation flags: {unknown}")
    expected = {k: getattr(known, k) for k in _EXPECT_FLAGS}
    expected = {k: v for k, v in expected.items() if v is not None}
    for k in ("seed", "k"):
        if k in expected:
            expected[k] = int(expected[k])
    return rest, expected


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("validate-run", "validate-eval"):
        print(__doc__, file=sys.stderr)
        return 2
    mode, target, *rest = argv
    try:
        _, expected = _parse_expectations(rest)
    except SystemExit as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 2
    try:
        d = validate_run(target, expected=expected or None) \
            if mode == "validate-run" \
            else validate_eval(target, expected=expected or None)
    except Exception as e:  # noqa: BLE001 - drivers branch on failure text
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print("VALID", json.dumps(d.get("identity", {}), sort_keys=True)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
