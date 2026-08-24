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

from latent_lab.train.checkpointing import (
    assert_json_numbers_finite,
    strict_json_loads,
)

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
    "config_sha256": "--expect-config-sha256",
}

# The FULL canonical run contract: resume validation MUST bind every one
# of these at once. The canonical config/recipe digest (config_sha256)
# canonically covers ALL behavior-changing training fields (mode,
# interval, k, max_k, LoRA rank/alpha, LR, steps, seed, optimizer,
# weight decay, schedule, warmup, grad clip, detach policy, gradient
# checkpointing, suite) — so a partial hand-written allowlist that could
# silently omit a semantic field is structurally impossible here.
_RUN_EXPECT_REQUIRED = frozenset((
    "model_id", "revision", "suite_sha256", "seed", "label", "k",
    "steps", "config_sha256",
))

# The exact config key set a training report may carry; anything else is
# unexpected identity/config metadata and is rejected, never ignored.
_TRAIN_CONFIG_KNOWN_KEYS = frozenset((
    "mode", "interval", "k", "max_k", "lora_r", "lora_alpha", "lr",
    "steps", "seed", "optimizer", "weight_decay", "lr_schedule",
    "warmup", "clip", "detach_z0", "grad_checkpoint", "model",
    "revision", "label", "device", "train_examples", "suite_sha256",
))


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

    ``expected`` is the driver's preregistered contract. When ANY
    expectation is supplied it must be the COMPLETE canonical run
    contract — model_id, revision, suite_sha256, seed, label, k, steps
    AND the canonical config/recipe digest (config_sha256) that binds
    every remaining behavior-changing field. Unknown expectation keys
    are rejected; nothing is silently ignored.
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
        # strict reader: NaN/Infinity constants fail at parse time ...
        report = strict_json_loads(report_path.read_text())
    except Exception as e:  # noqa: BLE001 - unreadable report fails closed
        raise ValueError(f"{where}: unreadable train report: {e}") from e
    try:
        # ... and decoded non-finite numbers fail closed as well
        assert_json_numbers_finite(report, where=f"{where}: train report")
    except Exception as e:
        raise ValueError(f"{where}: train report is not finite evidence: "
                         f"{e}") from e
    cfg = report.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: train report has no config object")
    unknown_cfg = sorted(set(cfg) - _TRAIN_CONFIG_KNOWN_KEYS)
    if unknown_cfg:
        raise ValueError(
            f"{where}: unexpected config metadata {unknown_cfg}; "
            "identity/config fields are never ignored")

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
        unknown = sorted(set(expected) - set(_EXPECT_FLAGS))
        if unknown:
            raise ValueError(
                f"{where}: unexpected expectation keys {unknown}; the "
                "preregistered contract must not carry unrecognized "
                "identity metadata")
        missing = sorted(_RUN_EXPECT_REQUIRED - set(expected))
        if missing:
            raise ValueError(
                f"{where}: incomplete expected contract — missing {missing};"
                " resume validation must bind the FULL canonical recipe: "
                "model/revision/suite/seed/label/k/steps plus the exact "
                "canonical config digest (config_sha256)")
        _require_exact(model_id, expected["model_id"],
                       "identity.model_id", where)
        _require_exact(revision, expected["revision"],
                       "identity.revision", where)
        _require_exact(suite_sha, expected["suite_sha256"],
                       "suite_sha256", where)
        _require_exact(manifest.get("seed"), expected["seed"],
                       "seed", where)
        _require_exact(manifest.get("label"), expected["label"],
                       "label", where)
        _require_exact(cfg.get("k"), expected["k"], "config.k", where)
        _require_exact(cfg.get("steps"), expected["steps"],
                       "config.steps", where)
        # the canonical digest over EVERY behavior-changing field — this
        # single binding rejects wrong LR, interval, LoRA, optimizer,
        # schedule, warmup, clip, detach/checkpoint policy, mode and
        # max_k without any hand-maintained field list
        _require_exact(recipe["config_sha256"], expected["config_sha256"],
                       "recipe.config_sha256 (canonical training identity)",
                       where)
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


def _require_cond(field, actual, canonical, where: str) -> None:
    """A duplicated field, when present, must equal the canonical value."""
    if actual != canonical:
        raise ValueError(
            f"{where}: contradictory identity — {field} {actual!r} != "
            f"canonical {canonical!r}")


def _reconcile_eval_identity(where: str, d: dict, ident: dict) -> None:
    """Every duplicated semantic field must agree with the canonical
    identity block; contradictions and extra conflicting aliases are
    rejected, never ignored."""
    model = d.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{where}: top-level 'model' missing/invalid")
    _require_cond("top-level model", model, ident["model_id"], where)

    top_rev = d.get("revision")
    if not isinstance(top_rev, str):
        raise ValueError(f"{where}: top-level 'revision' missing")
    _check_revision(top_rev, where)
    _require_cond("top-level revision", top_rev, ident["revision"], where)

    top_suite = d.get("suite_sha256")
    if not isinstance(top_suite, str):
        raise ValueError(f"{where}: top-level 'suite_sha256' missing")
    _check_sha(top_suite, "top-level suite_sha256", where)
    _require_cond("top-level suite_sha256", top_suite,
                  ident["suite_sha256"], where)

    _require_cond("top-level split", d.get("split"), ident["split"], where)
    seed_top = d.get("seed")
    if isinstance(seed_top, bool) or not isinstance(seed_top, int) \
            or seed_top != ident["seed"]:
        raise ValueError(
            f"{where}: contradictory identity — top-level seed "
            f"{d.get('seed')!r} != canonical {ident['seed']!r}")

    # config fields, when present, must agree with the identity
    cfgp = d.get("config")
    if cfgp is not None:
        if not isinstance(cfgp, dict):
            raise ValueError(f"{where}: config is not an object")
        for ckey, ikey in (("model", "model_id"),
                           ("revision", "revision"),
                           ("suite_sha256", "suite_sha256"),
                           ("seed", "seed"),
                           ("k", "k_steps"),
                           ("max_k", "max_k"),
                           ("interval", "interval")):
            if ckey in cfgp:
                _require_cond(f"config.{ckey}", cfgp[ckey],
                              ident[ikey], where)


def _reconcile_selected_result(where: str, d: dict, ident: dict) -> None:
    """Exactly ONE selected result may exist — keyed by the canonical
    ablation — and its metadata/tag must encode the same K/split/
    ablation as the identity."""
    results = d["results"]
    if set(results) != {ident["ablation"]}:
        raise ValueError(
            f"{where}: unexpected selected results {sorted(results)} for "
            f"ablation {ident['ablation']!r}; extra conflicting aliases "
            "fail closed")
    res = results[ident["ablation"]]
    k_steps = ident["k_steps"]
    res_k = res.get("k_steps")
    if isinstance(res_k, bool) or not isinstance(res_k, int) \
            or res_k != k_steps:
        raise ValueError(
            f"{where}: contradictory identity — result k_steps {res_k!r} "
            f"!= canonical {k_steps!r}")
    ablate = res.get("ablate")
    if ident["ablation"] == "clean":
        if ablate not in (None, {}):
            raise ValueError(
                f"{where}: contradictory identity — clean eval carries "
                f"ablate {ablate!r}")
    elif not isinstance(ablate, dict) or not ablate:
        raise ValueError(
            f"{where}: contradictory identity — ablation "
            f"{ident['ablation']!r} carries no ablate spec")
    tag = res.get("tag")
    if not isinstance(tag, str) or tag.count("|") != 3:
        raise ValueError(
            f"{where}: result tag {tag!r} is not the canonical "
            "'<mode>|<split>|<ablation>|K=<k>' encoding")
    tag_mode, tag_split, tag_abl, tag_k = tag.split("|")
    if tag_split != ident["split"] or tag_abl != ident["ablation"] \
            or tag_k != f"K={k_steps}" or not tag_mode.strip():
        raise ValueError(
            f"{where}: contradictory identity — result tag {tag!r} does "
            f"not match split={ident['split']!r} ablation="
            f"{ident['ablation']!r} k_steps={k_steps!r}")


def validate_eval(eval_path, *, expected: dict | None = None) -> dict:
    """An eval payload is evidence only if complete + fully identity-bound.

    ``expected`` is the driver's preregistered contract; every provided
    key must match payload CONTENTS exactly:
      model_id, revision, suite_sha256, checkpoint_content_digest, split,
      ablation, k, seed.
    Unknown expectation keys are rejected; duplicated semantic fields are
    reconciled against the canonical identity (contradictions fail).
    """
    p = Path(eval_path)
    where = str(p)
    try:
        # strict reader: NaN/Infinity/-Infinity reject at parse time ...
        d = strict_json_loads(p.read_text())
    except Exception as e:  # noqa: BLE001 - unreadable payload fails closed
        raise ValueError(f"{where}: unreadable eval payload: {e}") from e
    if not isinstance(d, dict):
        raise ValueError(f"{where}: payload is not a JSON object")
    try:
        # ... and any decoded non-finite number fails closed as well
        assert_json_numbers_finite(d, where=f"{where}: payload")
    except Exception as e:
        raise ValueError(f"{where}: payload is not finite evidence: "
                         f"{e}") from e
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
    for key in ("model_id", "split", "ablation"):
        if not isinstance(ident[key], str):
            raise ValueError(f"{where}: identity.{key} must be a string")
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

    # optional extended identity fields, when present, must be well-formed
    interval = ident.get("interval")
    if interval is not None:
        if (not isinstance(interval, list) or len(interval) != 2
                or any(isinstance(x, bool) or not isinstance(x, int)
                       for x in interval)):
            raise ValueError(f"{where}: identity.interval must be [lo, hi]")
    max_k = ident.get("max_k")
    if max_k is not None and (isinstance(max_k, bool)
                              or not isinstance(max_k, int) or max_k < 1):
        raise ValueError(f"{where}: identity.max_k must be a positive int")

    # EVERY duplicated semantic field must agree with this identity
    _reconcile_eval_identity(where, d, ident)

    results = d.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{where}: no results block")
    for name, res in results.items():
        _validate_eval_result(where, name, res)
    if ablation not in results:
        raise ValueError(
            f"{where}: identity.ablation {ablation!r} has no matching "
            "results entry")
    _reconcile_selected_result(where, d, ident)

    if expected:
        unknown = sorted(set(expected) - set(_EXPECT_FLAGS))
        if unknown:
            raise ValueError(
                f"{where}: unexpected expectation keys {unknown}; the "
                "preregistered contract must not carry unrecognized "
                "identity metadata")
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
    for k in ("seed", "k", "steps"):
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
