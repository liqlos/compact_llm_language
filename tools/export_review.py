"""Build a deterministic, tracked-file-only review export manifest.

The tool never discovers untracked files, never follows symlinks, and never
includes private evidence unless each private path is explicitly allowlisted.
It can optionally materialize the selected files into a new directory outside
the repository; it does not create or commit a bulk archive.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


SCHEMA_VERSION = "rcc.review_export.v1"
POLICY_VERSION = "rcc.review_export_policy.v1"

GENERATED_OUTPUTS = frozenset({
    "artifacts/export_manifest.json",
    "artifacts/export_size_summary.json",
})

DENIED_DIRECTORY_NAMES = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".rcc_work",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
})

PRIVATE_PREFIXES = (
    ".private/",
    "artifacts/private/",
    "artifacts/operational/",
    "evidence/private/",
    "private/",
)

ARCHIVE_SUFFIXES = (
    ".7z", ".rar", ".tar", ".tar.bz2", ".tar.gz", ".tar.xz",
    ".tgz", ".zip",
)

MODEL_WEIGHT_SUFFIXES = (
    ".bin", ".ckpt", ".gguf", ".npy", ".npz", ".onnx", ".pt",
    ".pth", ".safetensors",
)

OPERATIONAL_PATTERNS = (
    ".env",
    ".env.*",
    "*.lock",
    "*.pid",
    "*provider_receipt*.json",
    "*telemetry_timestamp*.json",
    "*vast_instance*.json",
)

REPRODUCIBILITY_LOCKFILES = frozenset({"uv.lock"})


class ExportPolicyError(RuntimeError):
    """The requested export is unsafe, ambiguous, or non-reproducible."""


@dataclass(frozen=True)
class TrackedEntry:
    path: str
    mode: str


def _run_git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _normalize_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", "..")
                                                   for part in path.parts):
        raise ExportPolicyError(f"path must be normalized and relative: {value!r}")
    return path.as_posix()


def tracked_entries(root: Path) -> tuple[TrackedEntry, ...]:
    """Read tracked index entries without relying on newline-delimited paths."""
    records = _run_git(root, "ls-files", "--cached", "--stage", "-z")
    entries = []
    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0].decode("ascii")
        path = _normalize_relative(os.fsdecode(raw_path))
        entries.append(TrackedEntry(path=path, mode=mode))
    return tuple(sorted(entries, key=lambda entry: entry.path))


def is_private_path(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.startswith(prefix) for prefix in PRIVATE_PREFIXES)


def exclusion_reason(path: str, *, allow_private: frozenset[str],
                     generated_outputs: frozenset[str]) -> str | None:
    lowered = path.lower()
    parts = PurePosixPath(lowered).parts
    name = parts[-1]
    if path in generated_outputs:
        return "generated_export_metadata"
    if any(part in DENIED_DIRECTORY_NAMES for part in parts):
        return "cache_or_work_directory"
    if name == ".ds_store":
        return "os_metadata"
    if lowered.endswith(ARCHIVE_SUFFIXES):
        return "archive"
    if lowered.endswith(MODEL_WEIGHT_SUFFIXES):
        return "model_weight"
    if name not in REPRODUCIBILITY_LOCKFILES and any(
            fnmatch.fnmatch(name, pattern)
            for pattern in OPERATIONAL_PATTERNS):
        return "operational_metadata"
    if is_private_path(path) and path not in allow_private:
        return "private_evidence_not_allowlisted"
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_root(root: Path) -> Path:
    resolved = root.resolve(strict=True)
    git_root = Path(os.fsdecode(
        _run_git(resolved, "rev-parse", "--show-toplevel").strip()
    )).resolve(strict=True)
    if resolved != git_root:
        raise ExportPolicyError(
            f"--root must be the git top level: {resolved} != {git_root}")
    return resolved


def build_manifest(
    root: Path,
    *,
    allow_private: Iterable[str] = (),
    generated_outputs: Iterable[str] = GENERATED_OUTPUTS,
) -> tuple[dict, dict]:
    root = _validate_root(root)
    allowed = frozenset(_normalize_relative(path) for path in allow_private)
    generated = frozenset(
        _normalize_relative(path) for path in generated_outputs)
    entries = tracked_entries(root)
    tracked_paths = {entry.path for entry in entries}

    for path in sorted(allowed):
        if not is_private_path(path):
            raise ExportPolicyError(
                f"--allow-private path is not under a private prefix: {path}")
        if path not in tracked_paths:
            raise ExportPolicyError(
                f"--allow-private path is not a tracked file: {path}")

    included = []
    excluded = []
    for entry in entries:
        reason = exclusion_reason(
            entry.path, allow_private=allowed,
            generated_outputs=generated)
        if reason is not None:
            excluded.append({"path": entry.path, "reason": reason})
            continue
        if entry.mode == "120000":
            raise ExportPolicyError(
                f"tracked symlink is not exportable: {entry.path}")
        if entry.mode not in ("100644", "100755"):
            raise ExportPolicyError(
                f"unsupported tracked mode {entry.mode} for {entry.path}")
        source = root / entry.path
        if not source.is_file():
            raise ExportPolicyError(
                f"tracked path is missing or not a regular file: {entry.path}")
        data = source.read_bytes()
        included.append({
            "mode": entry.mode,
            "path": entry.path,
            "sha256": _sha256(data),
            "size_bytes": len(data),
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "source": "git_index_paths_with_worktree_bytes",
        "private_allowlist": sorted(allowed),
        "files": included,
        "excluded_tracked_files": excluded,
    }
    top_level = Counter(
        PurePosixPath(item["path"]).parts[0] for item in included)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "file_count": len(included),
        "excluded_tracked_file_count": len(excluded),
        "total_bytes": sum(item["size_bytes"] for item in included),
        "files_by_top_level": dict(sorted(top_level.items())),
    }
    return manifest, summary


def canonical_json(payload: dict) -> bytes:
    return (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def write_json(path: Path, payload: dict, *, check: bool) -> None:
    expected = canonical_json(payload)
    if check:
        try:
            actual = path.read_bytes()
        except FileNotFoundError as error:
            raise ExportPolicyError(f"generated file missing: {path}") from error
        if actual != expected:
            raise ExportPolicyError(f"generated file is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(expected)
    os.replace(temporary, path)


def materialize(root: Path, output_dir: Path, manifest: dict) -> None:
    root = root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    if output == root or root in output.parents or output in root.parents:
        raise ExportPolicyError(
            f"output directory overlaps repository root: {output}")
    if output.exists():
        raise ExportPolicyError(f"output directory already exists: {output}")

    payloads = []
    for item in manifest["files"]:
        source = root / item["path"]
        data = source.read_bytes()
        if len(data) != item["size_bytes"] or _sha256(data) != item["sha256"]:
            raise ExportPolicyError(
                f"source changed after manifest generation: {item['path']}")
        payloads.append((item, data))

    output.mkdir(parents=True)
    for item, data in payloads:
        destination = output / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(0o755 if item["mode"] == "100755" else 0o644)
        os.utime(destination, (0, 0), follow_symlinks=False)


def _relative_if_within(root: Path, path: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create deterministic safe review-export metadata")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/export_manifest.json"))
    parser.add_argument(
        "--summary", type=Path,
        default=Path("artifacts/export_size_summary.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-private", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    root = _validate_root(args.root)
    manifest_path = (args.manifest if args.manifest.is_absolute()
                     else root / args.manifest)
    summary_path = (args.summary if args.summary.is_absolute()
                    else root / args.summary)
    generated = set(GENERATED_OUTPUTS)
    for path in (manifest_path, summary_path):
        relative = _relative_if_within(root, path)
        if relative is not None:
            generated.add(relative)

    manifest, summary = build_manifest(
        root,
        allow_private=args.allow_private,
        generated_outputs=generated,
    )
    write_json(manifest_path, manifest, check=args.check)
    write_json(summary_path, summary, check=args.check)
    if args.output_dir is not None:
        if args.check:
            raise ExportPolicyError("--check cannot be combined with --output-dir")
        materialize(root, args.output_dir, manifest)
    print(canonical_json(summary).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExportPolicyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"export failed: {error}") from error
