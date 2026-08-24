"""Sealed-driver invariants for the preregistered 4B canary.

These tests inspect latent_lab/bench/remote_driver_4b.sh SOURCE and
statically prove the paid-driver contract:
  * exact dependency pins (==, no ranges) incl. project uv.lock identity
  * focused tests gate BEFORE any GPU/model work
  * preregistered seed/recipe; NO result-based selection (BEST/MEDIAN)
  * paired same-adapter K>0 / K=0 evals; bounded (one train + 4 evals)
  * resume only through full-contract validators; no existence-only skip
  * no package installs/upgrades; no ignored failures
"""

import hashlib
import platform
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "latent_lab" / "bench" / "remote_driver_4b.sh"
UVLOCK = REPO / "uv.lock"


def _src() -> str:
    return DRIVER.read_text()


def _code(src: str) -> str:
    """Driver source without comment lines (comments document the bans)."""
    return "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))


def _locked_version(package: str) -> str:
    text = UVLOCK.read_text()
    m = re.search(
        r'\[\[package\]\]\s*\nname = "' + re.escape(package)
        + r'"\s*\nversion = "([^"]+)"', text)
    assert m, f"{package} not found in uv.lock"
    return m.group(1)


def _pin(src: str, var: str) -> str:
    m = re.search(rf'^{var}="([^\"]+)"$', src, re.M)
    assert m, f"pin {var} missing from driver"
    return m.group(1)


# ---------------------------------------------------------------------------
# shell validity + exact environment identity
# ---------------------------------------------------------------------------

def test_driver_shell_syntax_is_valid():
    r = subprocess.run(["bash", "-n", str(DRIVER)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dependency_pins_are_exact_and_match_the_project_lock():
    src = _src()
    expected = {
        "PIN_PYTHON": platform.python_version(),
        "PIN_TORCH": _locked_version("torch"),
        "PIN_TRANSFORMERS": _locked_version("transformers"),
        "PIN_HUGGINGFACE_HUB": _locked_version("huggingface-hub"),
    }
    for var, locked in expected.items():
        pin = _pin(src, var)
        assert pin == locked, (
            f"{var}={pin} drifted from uv.lock {locked}")
        # equality against the pin must be enforced in the env check
        assert f'"{pin}"' in src
    assert "__version__ ==" in src or "require(" in src
    assert re.search(r"require\(.*__version__", src), \
        "dependency versions are not equality-checked"
    # python equality is checked too (not merely presence)
    assert 'require(sys.version.split()[0]' in src


def test_uvlock_identity_pin_matches_repo_lock():
    src = _src()
    pin = _pin(src, "PIN_UVLOCK_SHA256")
    actual = hashlib.sha256(UVLOCK.read_bytes()).hexdigest()
    assert pin == actual, "driver uv.lock sha pin drifted from repo lock"
    assert "ACTUAL_LOCK_SHA" in src and "uv.lock sha256" in src


def test_no_installs_upgrades_or_ignored_failures():
    code = _code(_src())
    assert "pip install" not in code, "driver installs packages"
    assert "pip3 install" not in code
    assert not re.search(r"\bpip\b.*\s-U\b", code), \
        "driver upgrades packages"
    assert "uv sync" not in code and "uv pip" not in code
    assert "|| true" not in code, "driver ignores failures"
    assert "set -euo pipefail" in _src()


def test_no_mutable_image_default_anywhere_in_launch_path():
    """The known-incompatible mutable tag can never reappear as a launch
    environment, and the driver has NO image default: an unsealed run
    aborts instead."""
    src = _src()
    assert "pytorch/pytorch:" not in src
    assert not re.search(r"^\s*IMAGE\s*=", src, re.M)
    provisioner = (REPO / "latent_lab" / "bench" / "vast_provision.py"
                   ).read_text()
    assert "pytorch/pytorch:" not in provisioner
    assert not re.search(r"^IMAGE\s*=", provisioner, re.M)


# ---------------------------------------------------------------------------
# pre-spend gate ordering
# ---------------------------------------------------------------------------

def test_focused_tests_run_before_any_gpu_or_model_work():
    lines = _src().splitlines()
    pytest_idx = next(i for i, l in enumerate(lines)
                      if l.strip().startswith("python -m pytest"))
    gpu_markers = ("latent_run train", "latent_run eval",
                   "AutoModelForCausalLM", "model_info")
    first_gpu_idx = min(
        i for i, l in enumerate(lines)
        if any(m in l for m in gpu_markers))
    assert pytest_idx < first_gpu_idx, \
        "focused checks must precede ALL GPU/model contact"
    gate = "\n".join(lines[pytest_idx:pytest_idx + 5])
    for suite in ("test_latent_runtime_integrity.py",
                  "test_latent_run.py",
                  "test_artifact_contracts.py",
                  "test_paid_driver_sealed.py"):
        assert suite in gate or any(suite in l for l in lines[
            pytest_idx:first_gpu_idx]), f"{suite} not gated pre-spend"
    assert "torch.cuda.is_available" in "\n".join(lines[:pytest_idx]), \
        "CUDA availability is not verified before the spend"


# ---------------------------------------------------------------------------
# preregistration: no cherry-picking, bounded matrix
# ---------------------------------------------------------------------------

def test_preregistered_seed_and_no_result_based_selection():
    code = _code(_src())
    assert "BEST" not in code and "MEDIAN" not in code, \
        "result-based best/median selection reappeared"
    assert "best_val_acc" not in code, "validation results steer selection"
    assert re.search(r"^SEED=\d+$", code, re.M), \
        "seed is not a fixed constant"
    assert '--seed "$SEED"' in code, "training does not use the fixed seed"
    # exactly ONE training run; nothing loops over seeds/adapters
    assert len(re.findall(r"latent_run train", code)) == 1
    assert not re.search(r"for\s+\w+\s+in\s+.*\d+\.\.\d|seq\s+\d+", code)
    assert "DRIVER_MATRIX" in code, "matrix guard absent"


def test_paired_k_arms_use_same_adapter_and_bounded_evals():
    src = _src()
    ev_calls = re.findall(r'^ev "\$LABEL" (\S+) "\$K_\w+"$', src, re.M)
    assert ev_calls == ["test_id", "test_id", "test_ood", "test_ood"], \
        f"unexpected eval plan: {ev_calls}"
    # both arms come from the SAME single adapter ($LABEL / $RUN_DIR),
    # never from separately trained F adapters; only one eval invocation
    # site exists (inside ev) and it is driven by $K_POS/$K_ZERO
    assert "F4_k0" not in src and "runs/F4" not in src
    assert len(re.findall(r"latent_run eval", src)) == 1
    assert '"$K_POS"' in src and '"$K_ZERO"' in src


# ---------------------------------------------------------------------------
# sealed resume: full-contract validation, never existence
# ---------------------------------------------------------------------------

def test_resume_requires_full_expected_contract_validation():
    src = _src()
    # run validation carries the FULL canonical contract: model/rev/
    # suite/seed/label/k/steps + the canonical config-recipe digest
    expect_run_start = src.index("expect_run=(")
    expect_run_block = src[expect_run_start:src.index(")",
                                                      expect_run_start)]
    for flag in ("--expect-model", "--expect-rev", "--expect-suite",
                 "--expect-seed", "--expect-label", "--expect-k",
                 "--expect-steps", "--expect-config-sha256"):
        assert flag in expect_run_block, f"run contract flag {flag} missing"
    # eval validation carries model/rev/suite/digest/split/ablation/k/seed
    for flag in ("--expect-suite", "--expect-digest", "--expect-split",
                 "--expect-ablation"):
        assert flag in src, f"eval contract flag {flag} missing"
    assert src.count("--expect-k") >= 3  # run contract + both eval paths
    # every resume branch validates BEFORE reuse; quarantine on failure
    assert src.count("quarantine") >= 2
    assert 'valid_run "$RUN_DIR"' in src
    assert ".invalid.$(date +%s)" in src
