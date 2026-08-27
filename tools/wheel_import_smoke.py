"""Import the installed public packages and prove they are not source-shadowed."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


PUBLIC_PACKAGES = ("bench", "evals", "latent_lab", "rcc")


def import_origins(*, forbidden_root: Path | None = None) -> dict[str, str]:
    origins: dict[str, str] = {}
    forbidden = forbidden_root.resolve() if forbidden_root is not None else None
    for name in PUBLIC_PACKAGES:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if forbidden is not None and origin.is_relative_to(forbidden):
            raise RuntimeError(f"{name} imported from source tree: {origin}")
        origins[name] = str(origin)
    importlib.import_module("bench.run_bench")
    return origins


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forbid-root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(
        {"origins": import_origins(forbidden_root=args.forbid_root),
         "status": "PASS"},
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
