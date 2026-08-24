"""Sealed launch-environment regressions (repair cycle 2, blocker 5).

Proves, statically and behaviorally and WITHOUT any network/provider
call:
  * mutable/missing/mismatched image or environment inputs abort BEFORE
    any provider contact (provisioner) and BEFORE tests/training
    (driver);
  * a controlled EXACT fake contract passes the sealing boundary only;
  * no default image exists anywhere in the launch path.
"""

import json
import re
import sys
import types
from pathlib import Path

import pytest

from latent_lab.bench import sealed_env, vast_provision

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "latent_lab" / "bench" / "remote_driver_4b.sh"
PROVISIONER = REPO / "latent_lab" / "bench" / "vast_provision.py"
MUTABLE_TAG = "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime"
GOOD_IMAGE = "pytorch/pytorch@sha256:" + "a1" * 32


def _live_contract(path, image=GOOD_IMAGE, **over) -> Path:
    """A CONTROLLED EXACT contract built from this interpreter's real
    versions + the repo lockfile digest (no network)."""
    v = sealed_env.live_versions()
    d = {"image": image,
         "python": v["python"],
         "torch": v["torch"],
         "transformers": v["transformers"],
         "huggingface_hub": v["huggingface_hub"],
         "uvlock_sha256": sealed_env.lockfile_sha256(REPO / "uv.lock")}
    d.update(over)
    path.write_text(json.dumps(d))
    return path


# ---------------------------------------------------------------------------
# immutable-image discipline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    MUTABLE_TAG, "repo:latest", "repo", "", None,
    GOOD_IMAGE.upper(),
    "repo@sha256:" + "A1" * 32,
    "repo@sha256:" + "a1" * 31,
    "repo@sha256:" + "g1" * 32,
    "repo@sha512:" + "a1" * 32,
    "repo@@sha256:" + "a1" * 32,
])
def test_mutable_or_malformed_image_references_are_refused(bad):
    with pytest.raises(sealed_env.SealedEnvironmentError):
        sealed_env.require_immutable_image(bad)


def test_immutable_digest_reference_is_accepted():
    assert sealed_env.require_immutable_image(GOOD_IMAGE) == GOOD_IMAGE


# ---------------------------------------------------------------------------
# exact sealed-environment contracts
# ---------------------------------------------------------------------------

def test_contract_loads_only_when_exact(tmp_path):
    c = _live_contract(tmp_path / "c.json")
    loaded = sealed_env.load_contract(c)
    assert loaded["image"] == GOOD_IMAGE

    good = json.loads(c.read_text())
    for drop in list(good):
        p = tmp_path / f"drop_{drop}.json"
        p.write_text(json.dumps(
            {k: v for k, v in good.items() if k != drop}))
        with pytest.raises(sealed_env.SealedEnvironmentError):
            sealed_env.load_contract(p)
    p = tmp_path / "extra.json"
    p.write_text(json.dumps({**good, "cuda": "12.6"}))
    with pytest.raises(sealed_env.SealedEnvironmentError, match="unexpected"):
        sealed_env.load_contract(p)
    p = tmp_path / "mutable.json"
    p.write_text(json.dumps({**good, "image": MUTABLE_TAG}))
    with pytest.raises(sealed_env.SealedEnvironmentError, match="immutable"):
        sealed_env.load_contract(p)
    p = tmp_path / "lock.json"
    p.write_text(json.dumps({**good, "uvlock_sha256": "nope"}))
    with pytest.raises(sealed_env.SealedEnvironmentError):
        sealed_env.load_contract(p)
    p = tmp_path / "nan.json"
    p.write_text('{"image": NaN}')
    with pytest.raises(sealed_env.SealedEnvironmentError):
        sealed_env.load_contract(p)
    with pytest.raises(sealed_env.SealedEnvironmentError):
        sealed_env.load_contract(tmp_path / "missing.json")


def test_check_pins_cross_binds_contract_to_preregistration(tmp_path):
    c = sealed_env.load_contract(_live_contract(tmp_path / "c.json"))
    sealed_env.check_contract_pins(c, pins={"image": c["image"],
                                            "python": c["python"]})
    with pytest.raises(sealed_env.SealedEnvironmentError,
                       match="disagrees with the preregistered pin"):
        sealed_env.check_contract_pins(c, pins={"python": "3.12.7"})
    with pytest.raises(sealed_env.SealedEnvironmentError,
                       match="unknown pin keys"):
        sealed_env.check_contract_pins(c, pins={"flavor": "vanilla"})


def test_verify_live_accepts_exact_fake_contract_and_rejects_drift(
        tmp_path):
    ok = sealed_env.load_contract(_live_contract(tmp_path / "ok.json"))
    sealed_env.verify_live_environment(ok)          # reaches next boundary
    drift = sealed_env.load_contract(_live_contract(
        tmp_path / "drift.json", python="3.12.7"))
    with pytest.raises(sealed_env.SealedEnvironmentError,
                       match="violates the sealed contract"):
        sealed_env.verify_live_environment(drift)
    lock_drift = sealed_env.load_contract(_live_contract(
        tmp_path / "lock.json", uvlock_sha256="ab" * 32))
    with pytest.raises(sealed_env.SealedEnvironmentError, match="lockfile"):
        sealed_env.verify_live_environment(lock_drift,
                                           lockfile=REPO / "uv.lock")
    cuda_missing = sealed_env.load_contract(
        _live_contract(tmp_path / "cuda.json"))
    with pytest.raises(sealed_env.SealedEnvironmentError):
        sealed_env.verify_live_environment(cuda_missing,
                                           require_cuda=True)


def test_sealed_env_cli_exit_codes(tmp_path):
    ok = _live_contract(tmp_path / "ok.json")
    assert sealed_env.main(["verify-live", "--contract", str(ok)]) == 0
    bad = _live_contract(tmp_path / "bad.json", torch="0.0.0")
    assert sealed_env.main(["verify-live", "--contract", str(bad)]) == 1
    assert sealed_env.main(["require-image", "--image", MUTABLE_TAG]) == 1
    assert sealed_env.main(["require-image", "--image", GOOD_IMAGE]) == 0
    c = _live_contract(tmp_path / "c.json")
    assert sealed_env.main(["check-pins", "--contract", str(c),
                            "--pin", f"image={GOOD_IMAGE}",
                            "--pin", f"python={json.loads(c.read_text())['python']}"]) == 0


# ---------------------------------------------------------------------------
# provisioner behavior: unsealed inputs abort before ANY provider contact
# ---------------------------------------------------------------------------

class _ExplodingVastAI:
    instantiated = False

    def __init__(self, *a, **k):
        type(self).instantiated = True
        raise AssertionError("VastAI constructed despite unsealed inputs")


class _CountingVastAI:
    instantiated = False

    def __init__(self, *a, **k):
        type(self).instantiated = True

    def show_instances(self):
        return "[]"

    def show_volumes(self):
        return "[]"

    def search_offers(self, **kw):
        return "[]"


def _install_fake_vastai(monkeypatch, cls):
    cls.instantiated = False
    mod = types.ModuleType("vastai")
    mod.VastAI = cls
    monkeypatch.setitem(sys.modules, "vastai", mod)
    return cls


@pytest.fixture
def provision_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VAST_AI_API_KEY", "test-key")
    monkeypatch.setattr(vast_provision, "STATE",
                        tmp_path / ".rcc_work" / "vast_instance.json")
    monkeypatch.setattr(sys, "argv", ["vast_provision.py"])
    return tmp_path


@pytest.mark.parametrize("argv_extra", [
    ["--create"],
    ["--create", "--image", MUTABLE_TAG],
    ["--create", "--image", GOOD_IMAGE],
])
def test_provision_create_aborts_before_provider_on_unsealed_inputs(
        monkeypatch, provision_env, tmp_path, argv_extra):
    cls = _install_fake_vastai(monkeypatch, _ExplodingVastAI)
    argv = ["vast_provision.py", *argv_extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as ei:
        vast_provision.main()
    assert "sealed launch environment refused" in str(ei.value)
    assert not cls.instantiated, \
        "provider layer was reached on unsealed inputs"


def test_provision_create_rejects_contract_image_mismatch(
        monkeypatch, provision_env, tmp_path):
    cls = _install_fake_vastai(monkeypatch, _ExplodingVastAI)
    other_digest = "pytorch/pytorch@sha256:" + "b2" * 32
    contract = _live_contract(tmp_path / "c.json", image=other_digest)
    monkeypatch.setattr(sys, "argv", [
        "vast_provision.py", "--create", "--image", GOOD_IMAGE,
        "--env-contract", str(contract)])
    with pytest.raises(SystemExit) as ei:
        vast_provision.main()
    assert "preregistered pin" in str(ei.value)
    assert not cls.instantiated


def test_provision_sealed_inputs_reach_offer_boundary_only(
        monkeypatch, provision_env, tmp_path):
    """Controlled EXACT fake contract + immutable digest => provisioning
    passes the seal gate but stops honestly at the next preflight
    boundary (no eligible offer; nothing created)."""
    cls = _install_fake_vastai(monkeypatch, _CountingVastAI)
    contract = _live_contract(tmp_path / "c.json")
    monkeypatch.setattr(sys, "argv", [
        "vast_provision.py", "--create", "--image", GOOD_IMAGE,
        "--env-contract", str(contract)])
    with pytest.raises(SystemExit, match="no eligible offer"):
        vast_provision.main()
    assert cls.instantiated, "seal gate never passed?"
    assert not vast_provision.STATE.exists()


def test_provision_module_has_no_default_image_and_lazy_vastai():
    src = PROVISIONER.read_text()
    assert not re.search(r"^IMAGE\s*=", src, re.M), \
        "mutable module-level IMAGE default reappeared"
    assert MUTABLE_TAG not in src
    assert src.index("def sealed_launch_requirements") \
        < src.index("from vastai import VastAI"), \
        "vastai must be imported lazily AFTER sealing validation"
    assert "repository@sha256" in src


# ---------------------------------------------------------------------------
# driver: sealed gate precedes env verification/tests/GPU work
# ---------------------------------------------------------------------------

def _driver_lines():
    return DRIVER.read_text().splitlines()


def test_driver_requires_external_sealed_inputs_and_has_no_default():
    src = DRIVER.read_text()
    assert ': "${SEALED_IMAGE:?' in src
    assert ': "${SEALED_ENV_CONTRACT:?' in src
    assert MUTABLE_TAG not in src
    assert not re.search(r"^\s*IMAGE\s*=", src, re.M)
    assert 'sealed_env require-image --image "$SEALED_IMAGE"' in src
    assert "verify-contract" in src and "check-pins" in src
    assert '--pin "uvlock_sha256=$PIN_UVLOCK_SHA256"' in src, \
        "contract is not cross-bound to the driver's own pins"


def test_driver_seal_gate_precedes_environment_tests_and_gpu_work():
    lines = _driver_lines()
    seal_idx = min(i for i, l in enumerate(lines)
                   if "sealed_env require-image" in l)
    heredoc_idx = next(i for i, l in enumerate(lines)
                       if "require(sys.version.split()[0]" in l)
    pytest_idx = next(i for i, l in enumerate(lines)
                      if l.strip().startswith("python -m pytest"))
    gpu_markers = ("latent_run train", "latent_run eval",
                   "AutoModelForCausalLM", "model_info")
    gpu_idx = min(i for i, l in enumerate(lines)
                  if any(m in l for m in gpu_markers))
    assert seal_idx < heredoc_idx < pytest_idx < gpu_idx, (
        seal_idx, heredoc_idx, pytest_idx, gpu_idx)


def test_driver_resume_contract_is_the_full_canonical_recipe():
    src = DRIVER.read_text()
    expect_run_start = src.index("expect_run=(")
    expect_run_block = src[expect_run_start:
                           src.index(")", expect_run_start)]
    for flag in ("--expect-model", "--expect-rev", "--expect-suite",
                 "--expect-seed", "--expect-label", "--expect-k",
                 "--expect-steps", "--expect-config-sha256"):
        assert flag in expect_run_block, \
            f"run resume contract misses {flag}"
    assert "CONFIG_SHA256=$(python - <<PY" in src
    assert "train_recipe_digest" in src, \
        "driver must preregister via the shared canonical helper"
