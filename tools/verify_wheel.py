"""Fail closed unless a built wheel contains every public package surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile


REQUIRED_WHEEL_PATHS = frozenset({
    "bench/__init__.py",
    "bench/run_bench.py",
    "evals/__init__.py",
    "latent_lab/__init__.py",
    "rcc/__init__.py",
})


class WheelContractError(RuntimeError):
    """A wheel is unreadable or omits a required import surface."""


def verify_wheel(path: Path) -> dict:
    if path.suffix != ".whl" or not path.is_file():
        raise WheelContractError(f"not a wheel file: {path}")
    try:
        with ZipFile(path) as archive:
            names = frozenset(archive.namelist())
    except (BadZipFile, OSError) as error:
        raise WheelContractError(f"unreadable wheel {path}: {error}") from error
    missing = sorted(REQUIRED_WHEEL_PATHS - names)
    if missing:
        raise WheelContractError(
            f"wheel {path.name} misses required files: {missing}")
    return {
        "wheel": path.name,
        "required_files": sorted(REQUIRED_WHEEL_PATHS),
        "status": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify_wheel(args.wheel), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WheelContractError as error:
        raise SystemExit(f"wheel verification failed: {error}") from error
