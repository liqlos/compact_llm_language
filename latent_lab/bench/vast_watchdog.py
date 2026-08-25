"""Local fail-closed watchdog for the single capped Vast canary.

The watchdog never creates provider resources.  It waits for either the
7200-second wall deadline or a local completion sentinel, destroys exactly
the preregistered instance, and requires both instance and volume inventories
to be empty within 120 seconds.  Provider output is never copied to its log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

MAX_WALL_SECONDS = 7200.0
MAX_TEARDOWN_SECONDS = 120.0
DEFAULT_POLL_SECONDS = 5.0
INACTIVE_STATUSES = frozenset({"exited", "destroyed"})


class WatchdogError(RuntimeError):
    """The provider state could not be proved safe."""


def _items(raw: str, key: str) -> list[dict]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise WatchdogError(f"invalid {key} inventory JSON") from exc
    if isinstance(value, dict):
        value = value.get(key, [])
    if not isinstance(value, list) or any(not isinstance(x, dict)
                                          for x in value):
        raise WatchdogError(f"invalid {key} inventory shape")
    return value


def _active_instance(raw: str, instance_id: int) -> bool:
    for item in _items(raw, "instances"):
        try:
            same = int(item.get("id")) == instance_id
        except (TypeError, ValueError):
            same = False
        status = str(item.get("actual_status", "")).lower()
        if same and status not in INACTIVE_STATUSES:
            return True
    return False


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=5, check=False)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, "", "")
    except OSError:
        return subprocess.CompletedProcess(command, 127, "", "")


def _inventory(runner) -> tuple[str, str]:
    instances = runner(["vastai", "show", "instances", "--raw"])
    volumes = runner(["vastai", "show", "volumes", "--raw"])
    if instances.returncode or volumes.returncode:
        raise WatchdogError("provider inventory command failed")
    return instances.stdout, volumes.stdout


def _event(log_path: Path, event: str, **fields) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "unix_time": int(time.time()), **fields}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def watch(*, instance_id: int, stop_file: Path, log_path: Path,
          wall_seconds: float = MAX_WALL_SECONDS,
          teardown_seconds: float = MAX_TEARDOWN_SECONDS,
          poll_seconds: float = DEFAULT_POLL_SECONDS,
          runner=_run, monotonic=time.monotonic, sleep=time.sleep) -> int:
    if instance_id <= 0:
        raise WatchdogError("instance_id must be positive")
    if not 0 < wall_seconds <= MAX_WALL_SECONDS:
        raise WatchdogError("wall_seconds exceeds the 7200-second cap")
    if not 0 < teardown_seconds <= MAX_TEARDOWN_SECONDS:
        raise WatchdogError("teardown_seconds exceeds the 120-second cap")
    if not 0 < poll_seconds <= teardown_seconds:
        raise WatchdogError("invalid poll_seconds")

    wall_deadline = monotonic() + wall_seconds
    seen_instance = False
    _event(log_path, "watchdog_started", instance_id=instance_id,
           wall_seconds=wall_seconds, teardown_seconds=teardown_seconds)
    while monotonic() < wall_deadline and not stop_file.exists():
        try:
            instances_raw, volumes_raw = _inventory(runner)
            active = _active_instance(instances_raw, instance_id)
            seen_instance = seen_instance or active
            volumes = _items(volumes_raw, "volumes")
            if seen_instance and not active and not volumes:
                _event(log_path, "teardown_confirmed",
                       instance_id=instance_id, trigger="provider_absent")
                return 0
        except WatchdogError:
            # A transient inventory failure cannot declare success.
            pass
        sleep(min(poll_seconds, max(0.0, wall_deadline - monotonic())))

    trigger = "completion_sentinel" if stop_file.exists() else "wall_deadline"
    _event(log_path, "teardown_started", instance_id=instance_id,
           trigger=trigger)
    teardown_deadline = monotonic() + teardown_seconds
    last_destroy = float("-inf")
    while monotonic() < teardown_deadline:
        now = monotonic()
        if now - last_destroy >= poll_seconds:
            result = runner(["vastai", "destroy", "instance",
                             str(instance_id), "-y", "--raw"])
            _event(log_path, "destroy_attempt", instance_id=instance_id,
                   returncode=int(result.returncode))
            last_destroy = now
        try:
            instances_raw, volumes_raw = _inventory(runner)
            if (not _active_instance(instances_raw, instance_id)
                    and not _items(volumes_raw, "volumes")):
                _event(log_path, "teardown_confirmed",
                       instance_id=instance_id, trigger=trigger)
                return 0
        except WatchdogError:
            pass
        sleep(min(poll_seconds,
                  max(0.0, teardown_deadline - monotonic())))

    _event(log_path, "teardown_unconfirmed", instance_id=instance_id,
           trigger=trigger)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float,
                        default=MAX_WALL_SECONDS)
    parser.add_argument("--teardown-seconds", type=float,
                        default=MAX_TEARDOWN_SECONDS)
    parser.add_argument("--poll-seconds", type=float,
                        default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args(argv)
    try:
        return watch(instance_id=args.instance_id,
                     stop_file=args.stop_file, log_path=args.log,
                     wall_seconds=args.wall_seconds,
                     teardown_seconds=args.teardown_seconds,
                     poll_seconds=args.poll_seconds)
    except WatchdogError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
