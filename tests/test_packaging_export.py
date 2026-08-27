"""Regression tests for test tiers, wheel surface, and safe exports."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

from tests.conftest import MODULE_TIERS, resolve_tier, tier_for_path
from tools.export_review import (
    ExportPolicyError,
    build_manifest,
    canonical_json,
    materialize,
)
from tools.verify_wheel import WheelContractError, verify_wheel
from tools.wheel_import_smoke import import_origins


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_registers_tiers_and_packages_bench():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    wheel_packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert wheel_packages == ["rcc", "latent_lab", "evals", "bench"]
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    for tier in (
        "core", "lab_unit", "transformer_integration", "remote_policy",
        "expensive_local_model", "remote_paid",
    ):
        assert any(entry.startswith(f"{tier}:") for entry in markers)


def test_repository_test_modules_have_one_deterministic_tier():
    existing = {path.name for path in (REPO_ROOT / "tests").glob("test_*.py")}
    assert set(MODULE_TIERS) == existing
    assert tier_for_path("tests/test_store.py") == "core"
    assert tier_for_path("tests/test_no_spend_gate.py") == "lab_unit"
    assert tier_for_path("tests/test_localized.py") == "transformer_integration"
    assert tier_for_path("tests/test_runtime_scientific.py") == (
        "transformer_integration")
    assert tier_for_path("tests/test_expensive_local_model_qwen.py") == (
        "expensive_local_model")
    assert tier_for_path("tests/test_remote_paid_canary.py") == "remote_paid"
    with pytest.raises(ValueError, match="no intentional tier assignment"):
        tier_for_path("tests/test_new_unclassified.py")


def test_remote_provider_policy_never_lands_in_core_or_paid_execution_tier():
    for name in (
        "test_paid_driver_sealed.py",
        "test_sealed_launch_contract.py",
        "test_vast_provision_capped_policy.py",
        "test_vast_watchdog.py",
    ):
        assert MODULE_TIERS[name] == "remote_policy"
        assert resolve_tier(f"tests/{name}", []) == "remote_policy"
        with pytest.raises(ValueError, match="fixed to remote_policy"):
            resolve_tier(f"tests/{name}", ["core"])


def _write_fake_wheel(path: Path, *, include_bench: bool) -> None:
    files = [
        "rcc/__init__.py",
        "latent_lab/__init__.py",
        "evals/__init__.py",
    ]
    if include_bench:
        files.extend(("bench/__init__.py", "bench/run_bench.py"))
    with ZipFile(path, "w") as archive:
        for name in files:
            archive.writestr(name, b"")


def test_wheel_verifier_requires_bench_and_all_public_packages(tmp_path):
    good = tmp_path / "rcc-0.1-py3-none-any.whl"
    bad = tmp_path / "rcc-0.1-bad-py3-none-any.whl"
    _write_fake_wheel(good, include_bench=True)
    _write_fake_wheel(bad, include_bench=False)
    assert verify_wheel(good)["status"] == "PASS"
    with pytest.raises(WheelContractError, match="bench/__init__.py"):
        verify_wheel(bad)


def test_wheel_import_smoke_rejects_source_shadowing():
    with pytest.raises(RuntimeError, match="imported from source tree"):
        import_origins(forbidden_root=REPO_ROOT)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _policy_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    files = {
        "src/good.py": b"print('safe')\n",
        "uv.lock": b"version = 1\n",
        ".venv/cache.py": b"cached\n",
        "nested/__pycache__/x.pyc": b"cache\n",
        ".DS_Store": b"metadata\n",
        "bundle.tar.gz": b"archive\n",
        "weights/model.safetensors": b"weights\n",
        "artifacts/operational/vast_instance.json": b"{}\n",
        "artifacts/private/allowed.json": b"{\"private\": true}\n",
        "artifacts/export_manifest.json": b"stale\n",
        "artifacts/export_size_summary.json": b"stale\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(root, "add", "-f", ".")
    return root


def test_export_is_deterministic_and_excludes_unsafe_tracked_files(tmp_path):
    root = _policy_repo(tmp_path)
    first_manifest, first_summary = build_manifest(root)
    second_manifest, second_summary = build_manifest(root)
    assert canonical_json(first_manifest) == canonical_json(second_manifest)
    assert canonical_json(first_summary) == canonical_json(second_summary)

    included = {entry["path"] for entry in first_manifest["files"]}
    assert included == {"src/good.py", "uv.lock"}
    excluded = {
        entry["path"]: entry["reason"]
        for entry in first_manifest["excluded_tracked_files"]
    }
    assert excluded[".venv/cache.py"] == "cache_or_work_directory"
    assert excluded["nested/__pycache__/x.pyc"] == "cache_or_work_directory"
    assert excluded[".DS_Store"] == "os_metadata"
    assert excluded["bundle.tar.gz"] == "archive"
    assert excluded["weights/model.safetensors"] == "model_weight"
    assert excluded["artifacts/operational/vast_instance.json"] == (
        "operational_metadata")
    assert excluded["artifacts/private/allowed.json"] == (
        "private_evidence_not_allowlisted")
    assert excluded["artifacts/export_manifest.json"] == (
        "generated_export_metadata")

    allowed_manifest, _ = build_manifest(
        root, allow_private=("artifacts/private/allowed.json",))
    allowed_paths = {entry["path"] for entry in allowed_manifest["files"]}
    assert "artifacts/private/allowed.json" in allowed_paths
    with pytest.raises(ExportPolicyError, match="not under a private prefix"):
        build_manifest(root, allow_private=("src/good.py",))


def test_materialized_export_is_new_outside_repo_and_byte_exact(tmp_path):
    root = _policy_repo(tmp_path)
    manifest, _summary = build_manifest(root)
    output = tmp_path / "export"
    materialize(root, output, manifest)
    for entry in manifest["files"]:
        exported = output / entry["path"]
        assert exported.read_bytes() == (root / entry["path"]).read_bytes()
        assert int(exported.stat().st_mtime) == 0
    with pytest.raises(ExportPolicyError, match="already exists"):
        materialize(root, output, manifest)
    with pytest.raises(ExportPolicyError, match="overlaps"):
        materialize(root, root / "unsafe-export", manifest)


def test_materialize_fails_closed_if_source_changed_after_manifest(tmp_path):
    root = _policy_repo(tmp_path)
    manifest, _summary = build_manifest(root)
    (root / "src/good.py").write_text("print('changed')\n", encoding="utf-8")
    output = tmp_path / "export"
    with pytest.raises(ExportPolicyError, match="source changed"):
        materialize(root, output, manifest)
    assert not output.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_export_rejects_tracked_symlink_instead_of_following(tmp_path):
    root = _policy_repo(tmp_path)
    os.symlink("src/good.py", root / "linked.py")
    _git(root, "add", "linked.py")
    with pytest.raises(ExportPolicyError, match="tracked symlink"):
        build_manifest(root)


def test_committed_export_metadata_matches_current_tracked_sources():
    manifest, summary = build_manifest(REPO_ROOT)
    assert json.loads((REPO_ROOT / "artifacts/export_manifest.json").read_text(
        "utf-8")) == manifest
    assert json.loads((REPO_ROOT / "artifacts/export_size_summary.json").read_text(
        "utf-8")) == summary
