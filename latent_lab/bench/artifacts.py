"""Run-lock + artifact validation primitives for remote drivers.

Contract (fail closed):
  * Runs serialize through an exclusive lock file (O_CREAT|O_EXCL with
    dead-holder recovery). Two drivers can never interleave writes.
  * A training directory counts as DONE only when its generation verifies:
    run status complete + coherent digest-linked report/checkpoint/manifest.
  * An eval payload counts as DONE only when it parses as JSON, carries
    status "complete", and binds suite/model/revision identity.

CLI:
    python -m latent_lab.bench.artifacts validate-run DIR
    python -m latent_lab.bench.artifacts validate-eval FILE
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path


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


def validate_run(run_dir) -> dict:
    """Return the verified manifest or raise; never trust bare existence."""
    from latent_lab.train.checkpointing import verify_generation

    return verify_generation(run_dir)


def validate_eval(eval_path) -> dict:
    """An eval payload is evidence only if complete + identity-bound."""
    p = Path(eval_path)
    d = json.loads(p.read_text())
    if d.get("status") != "complete":
        raise ValueError(f"{p}: status={d.get('status')!r}, not 'complete'")
    ident = d.get("identity") or {}
    for key in ("model_id", "revision", "suite_sha256"):
        if not ident.get(key):
            raise ValueError(f"{p}: identity.{key} missing")
    rev = str(ident["revision"])
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise ValueError(f"{p}: revision {rev!r} is not pinned 40-hex")
    results = d.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{p}: no results block")
    return d


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] not in ("validate-run", "validate-eval"):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        d = validate_run(argv[1]) if argv[0] == "validate-run" \
            else validate_eval(argv[1])
    except Exception as e:  # noqa: BLE001 - drivers branch on failure text
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print("VALID", json.dumps(d.get("identity", {}), sort_keys=True)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
