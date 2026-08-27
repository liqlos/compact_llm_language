"""Regressions for the independently reproduced runtime contract failures.

Coverage map (each test fails on rejected candidate 6bf51fd):
  * Fix 1: device/dtype-bucketed bounded-sync reductions (CPU halves;
    CUDA/MPS halves live in test_latent_runtime_integrity.py).
  * Fix 2: uniform fatal wrapping of clipping/step/postcheck/inspection/
    helper faults; cmd_train atomically marks the run fatal and never
    leaves status 'running' or emits success artifacts.
  * Fix 3: exact canonical recipe identity (every semantic field binds).
  * Fix 4: strict run/eval evidence validation under explicit expected
    contracts; the two reproduced false accepts are pinned as negatives.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from latent_lab.bench import artifacts, latent_run
from latent_lab.train import checkpointing as ckpt
from latent_lab.train.checkpointing import (
    AdapterBundleIdentityError,
    AdapterBundleSchemaError,
    FatalRunInvalidError,
    GuardedStepFaultError,
    NonFiniteTrainingStateError,
    OptimizerStateInspectionError,
    guarded_optimizer_step,
    load_adapter_bundle,
    save_adapter_bundle,
    validate_optimizer_state_standard_and_finite,
)

from tests.artifact_fakes import (
    BASE_CFG,
    EVAL_IDENTITY,
    MODEL,
    REV_OK,
    SUITE_SHA,
    build_eval_payload,
    build_verified_run,
    cfg as fake_cfg,
    eval_record,
    run_contract,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _big_grad_model():
    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    loss = (m(x) ** 2).mean()
    loss.backward()
    for p in m.parameters():          # force a REAL clipping pass later
        p.grad.fill_(50.0)
    return m, opt, loss


def _run_dir_named(tmp_path, name):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Fix 2: uniform fatal wrapping (exact prior repro first)
# ---------------------------------------------------------------------------

def test_clip_kernel_fault_surfaces_as_fatal_preserving_cause(monkeypatch):
    """Prior repro: torch._foreach_mul_ raising RuntimeError('clip boom')
    during REAL clipping escaped as a raw exception. It must surface as
    FatalRunInvalidError with the original cause preserved."""
    m, opt, loss = _big_grad_model()
    boom = RuntimeError("clip boom")

    def exploding_mul_(tensors, scalar):
        raise boom

    monkeypatch.setattr(torch, "_foreach_mul_", exploding_mul_)
    with pytest.raises(FatalRunInvalidError) as ei:
        guarded_optimizer_step(opt, loss.detach(), list(m.parameters()), 0.5)
    assert "clip boom" in str(ei.value)
    assert ei.value.__cause__ is boom


def test_postcheck_helper_fault_surfaces_as_fatal_after_step(
        monkeypatch):
    """A fault in the POST-step parameter postcheck is fatal, preserves
    the cause, and happens AFTER optimizer.step really ran."""
    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    loss = (m(x) ** 2).mean()
    loss.backward()

    calls = {"n": 0}
    real = ckpt._all_finite_fused
    boom = RuntimeError("postcheck boom")

    def flaky(tensors):
        calls["n"] += 1
        if calls["n"] >= 3:           # 1: pre-params, 2: post-clip, 3: post
            raise boom
        return real(tensors)

    monkeypatch.setattr(ckpt, "_all_finite_fused", flaky)
    before = m.weight.detach().clone()
    with pytest.raises(GuardedStepFaultError) as ei:
        guarded_optimizer_step(opt, loss.detach(), list(m.parameters()), 0.5)
    assert ei.value.__cause__ is boom
    assert not torch.equal(m.weight.detach(), before), \
        "fault injected before the step ran; wrong phase"


def test_prestep_helper_fault_is_fatal_without_mutation(monkeypatch):
    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(6, 4)
    loss = (m(x) ** 2).mean()
    loss.backward()
    snap = {id(p): p.detach().clone() for p in m.parameters()}
    monkeypatch.setattr(
        ckpt, "_all_finite_fused",
        lambda tensors: (_ for _ in ()).throw(RuntimeError("precheck boom")))
    with pytest.raises(FatalRunInvalidError):
        guarded_optimizer_step(opt, loss.detach(), list(m.parameters()), 1.0)
    assert all(torch.equal(p.detach(), snap[id(p)]) for p in m.parameters())


def test_optimizer_state_inspection_fault_wrapped_fatal_preserving_cause():
    class _HostileStateOptimizer:
        @property
        def state(self):
            raise RuntimeError("state boom")

    with pytest.raises(OptimizerStateInspectionError) as ei:
        validate_optimizer_state_standard_and_finite(_HostileStateOptimizer())
    assert isinstance(ei.value, FatalRunInvalidError)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "state boom" in str(ei.value.__cause__)

    # the same fault through the guarded step is fatal, never raw
    class _Stepper(_HostileStateOptimizer):
        def __init__(self, inner):
            self.inner = inner

        def step(self, closure=None):
            self.inner.step(closure)

        @property
        def param_groups(self):
            return self.inner.param_groups

    torch.manual_seed(3)
    m = torch.nn.Linear(4, 4)
    opt = _Stepper(torch.optim.AdamW(m.parameters(), lr=0.1))
    x = torch.randn(6, 4)
    loss = (m(x) ** 2).mean()
    loss.backward()
    with pytest.raises(FatalRunInvalidError) as ei2:
        guarded_optimizer_step(opt, loss.detach(), list(m.parameters()), 1.0)
    assert isinstance(ei2.value.__cause__, RuntimeError)


def test_cmd_train_marks_run_fatal_atomically_on_any_exception(tmp_path,
                                                               monkeypatch):
    """cmd_train must atomically mark the run fatal for BOTH explicit
    fatal errors and unexpected crashes — never leave status 'running' —
    and must not emit report/checkpoint/manifest."""
    from latent_lab.train.checkpointing import (
        CHECKPOINT_FILE,
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
        read_run_status,
    )

    args = SimpleNamespace(revision=REV_OK, device="cpu", model=MODEL,
                           seed=0, out=str(tmp_path))

    for exc in (FatalRunInvalidError("fatal boom"),
                RuntimeError("unexpected boom")):
        def broken(args, out, device, revision, _exc=exc):
            raise _exc

        monkeypatch.setattr(latent_run, "_train_inner", broken)
        with pytest.raises(type(exc)):
            latent_run.cmd_train(args)
        st = read_run_status(tmp_path)
        assert st is not None and st["status"] == "fatal"
        assert st["error_type"] == type(exc).__name__
        assert exc.args[0].split()[0] in st["error"]
        assert not (tmp_path / TRAIN_REPORT_FILE).exists()
        assert not (tmp_path / CHECKPOINT_FILE).exists()
        assert not (tmp_path / RUN_MANIFEST_FILE).exists()


# ---------------------------------------------------------------------------
# Fix 1 (CPU halves): mixed groups/dtypes + bounded host decisions
# ---------------------------------------------------------------------------

def test_guarded_step_handles_mixed_param_groups_and_dtypes():
    torch.manual_seed(0)
    m32 = torch.nn.Linear(4, 4)
    m64 = torch.nn.Linear(4, 4).to(torch.float64)
    mbf = torch.nn.Linear(4, 4).to(torch.bfloat16)
    groups = [{"params": list(m32.parameters()), "lr": 1e-2},
              {"params": list(m64.parameters()), "lr": 1e-3},
              {"params": list(mbf.parameters()), "lr": 1e-1}]
    opt = torch.optim.AdamW(groups, lr=1e-2)
    x = torch.randn(6, 4)
    loss = ((m32(x) ** 2).mean()
            + (m64(x.to(torch.float64)) ** 2).mean()
            + (mbf(x.to(torch.bfloat16)).float() ** 2).mean())
    loss.backward()
    before = [p.detach().clone() for group in groups for p in group["params"]]
    guarded_optimizer_step(
        opt, loss.detach(),
        [p for group in groups for p in group["params"]],
        100.0)
    after = [p.detach() for group in groups for p in group["params"]]
    for b, a in zip(before, after):
        assert not torch.equal(b.to(a.dtype), a), \
            "a device/dtype bucket silently skipped its step"


def test_bucketed_checks_catch_poison_in_any_single_dtype_bucket():
    torch.manual_seed(0)
    m32 = torch.nn.Linear(4, 4)
    m64 = torch.nn.Linear(4, 4).to(torch.float64)
    groups = [{"params": list(m32.parameters()), "lr": 1e-2},
              {"params": list(m64.parameters()), "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, lr=1e-2)
    x32, x64 = torch.randn(6, 4), torch.randn(6, 4, dtype=torch.float64)
    loss = (m32(x32) ** 2).mean() + (m64(x64) ** 2).mean()
    loss.backward()
    snap = [p.detach().clone() for group in groups for p in group["params"]]
    with torch.no_grad():
        m64.bias.grad.fill_(float("nan"))     # poison ONLY the fp64 bucket
    with pytest.raises(NonFiniteTrainingStateError):
        guarded_optimizer_step(opt, loss.detach(),
                               [p for g in groups for p in g["params"]],
                               100.0)
    after = [p.detach() for group in groups for p in group["params"]]
    assert all(torch.equal(b, a) for b, a in zip(snap, after))


def test_host_decisions_are_bounded_never_per_gradient(monkeypatch):
    """Six trainables, real clipping: host reads stay O(#phases), never
    O(#gradients); device-tensor truthiness stays at the single loss
    check (the rejected design synced once per gradient post-clip)."""
    torch.manual_seed(1)
    model = torch.nn.Sequential(*[torch.nn.Linear(8, 8) for _ in range(3)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(4, 8)
    ((model(x) ** 2).mean()).backward()

    counts = {"item": 0, "bool": 0}
    real_item = torch.Tensor.item
    real_bool = torch.Tensor.__bool__

    def spy_item(self, *a, **kw):
        counts["item"] += 1
        return real_item(self, *a, **kw)

    def spy_bool(self):
        counts["bool"] += 1
        return real_bool(self)

    monkeypatch.setattr(torch.Tensor, "item", spy_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", spy_bool)
    guarded_optimizer_step(opt, (model(x) ** 2).mean(),
                           list(model.parameters()), 0.01)
    assert counts["item"] <= 6, \
        f"unbounded host reads ({counts}) for 6 trainables"
    assert counts["bool"] <= 2, \
        f"per-gradient/per-tensor host decisions: {counts}"


def test_no_device_unspecified_accumulator_is_created(monkeypatch):
    """Every accumulator allocated by the fused checks carries an EXPLICIT
    device equal to its bucket's device (fails on the CPU/device-
    unspecified accumulator of the rejected candidate)."""
    allocs = []
    real_zeros = torch.zeros

    def spy_zeros(*a, **kw):
        t = real_zeros(*a, **kw)
        allocs.append((str(t.device), t.dtype, kw.get("device", "<unset>")))
        return t

    monkeypatch.setattr(torch, "zeros", spy_zeros)
    torch.manual_seed(1)
    model = torch.nn.Sequential(*[torch.nn.Linear(8, 8) for _ in range(2)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(4, 8)
    ((model(x) ** 2).mean()).backward()
    allocs.clear()
    guarded_optimizer_step(opt, (model(x) ** 2).mean(),
                           list(model.parameters()), 0.01)
    assert allocs, "fused reductions allocated nothing (helper bypassed?)"
    for dev, dtype, dev_arg in allocs:
        assert dev_arg != "<unset>", \
            f"device-unspecified accumulator allocated: {allocs}"
        assert dev == "cpu", f"accumulator off-bucket: {allocs}"


# ---------------------------------------------------------------------------
# Fix 3: exact canonical recipe identity
# ---------------------------------------------------------------------------

_BASE_RECIPE_CFG = fake_cfg()


@pytest.mark.parametrize("field,over", [
    ("k", {"k": 8}),
    ("max_k", {"max_k": 8}),
    ("lora_r", {"lora_r": 16}),
    ("lora_alpha", {"lora_alpha": 32.0}),
    ("lr", {"lr": 5e-5}),
    ("steps", {"steps": 400}),
    ("seed", {"seed": 1}),
    ("optimizer", {"optimizer": "sgd"}),
    ("weight_decay", {"weight_decay": 0.0}),
    ("lr_schedule", {"lr_schedule": "cosine"}),
    ("warmup", {"warmup": 10}),
    ("clip", {"clip": 1.0}),
    ("detach_z0", {"detach_z0": True}),
    ("interval", {"interval": [6, 18]}),
    ("mode", {"mode": "D-full"}),
])
def test_every_semantic_field_changes_the_recipe_identity(field, over):
    base = ckpt.recipe_from_config(_BASE_RECIPE_CFG, SUITE_SHA)
    other = ckpt.recipe_from_config({**_BASE_RECIPE_CFG, **over}, SUITE_SHA)
    assert other["config_sha256"] != base["config_sha256"], \
        f"{field} change collapsed to the same identity"
    assert other != base


def test_suite_hash_is_bound_but_config_digest_is_stable_to_it():
    base = ckpt.recipe_from_config(_BASE_RECIPE_CFG, SUITE_SHA)
    other = ckpt.recipe_from_config(_BASE_RECIPE_CFG, "cd" * 32)
    assert other["suite_sha256"] == "cd" * 32
    assert other["config_sha256"] == base["config_sha256"]
    assert other != base


def test_recipe_rejects_missing_extra_and_invalid_fields():
    full = ckpt.recipe_from_config(_BASE_RECIPE_CFG, SUITE_SHA)
    for missing in ("k", "seed", "steps", "lr", "clip", "config_sha256"):
        bad = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(AdapterBundleSchemaError):
            ckpt.validate_recipe(bad)
    with pytest.raises(AdapterBundleSchemaError):
        ckpt.validate_recipe({**full, "extra": 1})
    for bad_over in ({"k": -1}, {"k": True}, {"steps": 0}, {"lr": 0.0},
                     {"clip": float("nan")}, {"warmup": -1},
                     {"lr_schedule": "linear"}, {"seed": False},
                     {"detach_z0": 1}, {"max_k": 0}, {"lora_r": 0},
                     {"weight_decay": -0.1}, {"optimizer": ""},
                     {"suite_sha256": "nope"}):
        bad = {**full, **bad_over}
        with pytest.raises(AdapterBundleSchemaError):
            ckpt.validate_recipe(bad)
    # an internally inconsistent config digest can NEVER pass as identity
    tampered = {**full, "config_sha256": "ab" * 32}
    with pytest.raises(AdapterBundleIdentityError):
        ckpt.validate_recipe(tampered)


def test_missing_semantic_config_fields_are_never_defaulted():
    for drop in ("lr", "seed", "warmup", "clip", "optimizer"):
        broken = {k: v for k, v in _BASE_RECIPE_CFG.items() if k != drop}
        with pytest.raises(AdapterBundleSchemaError):
            ckpt.recipe_from_config(broken, SUITE_SHA)


def test_removed_grad_checkpoint_is_false_only_migration_input():
    canonical = ckpt.recipe_from_config(_BASE_RECIPE_CFG, SUITE_SHA)
    migrated = ckpt.recipe_from_config(
        {**_BASE_RECIPE_CFG, "grad_checkpoint": False}, SUITE_SHA)
    assert migrated == canonical
    assert "grad_checkpoint" not in canonical
    with pytest.raises(AdapterBundleSchemaError, match="unsupported"):
        ckpt.recipe_from_config(
            {**_BASE_RECIPE_CFG, "grad_checkpoint": True}, SUITE_SHA)


@pytest.mark.parametrize("field,over", [
    ("k", {"k": 8}), ("lr", {"lr": 5e-5}), ("steps", {"steps": 400}),
    ("seed", {"seed": 2}), ("weight_decay", {"weight_decay": 0.05}),
])
def test_cross_load_fails_when_any_semantic_field_changed(tmp_path, field,
                                                          over):
    sd = {"lora.0.A": torch.randn(8, 4) * 0.01}
    path = tmp_path / "best_params.pt"
    save_adapter_bundle(path, sd, model_id="M", revision=REV_OK,
                        recipe=ckpt.recipe_from_config(_BASE_RECIPE_CFG,
                                                       SUITE_SHA))
    drifted = ckpt.recipe_from_config({**_BASE_RECIPE_CFG, **over},
                                      SUITE_SHA)
    with pytest.raises(AdapterBundleIdentityError):
        load_adapter_bundle(path, model_id="M", revision=REV_OK,
                            recipe=drifted)


def test_report_manifest_and_bundle_bind_one_canonical_recipe(tmp_path):
    build_verified_run(tmp_path)
    manifest = artifacts.validate_run(tmp_path)
    report = json.loads((tmp_path / "train_report.json").read_text())
    assert manifest["recipe"] == report["recipe"]
    assert (manifest["checkpoint_content_digest"]
            == report["checkpoint_content_digest"])
    assert (manifest["selected_adapter_state_sha256"]
            == report["selected_adapter_state_sha256"])
    # tampering with the REPORT config breaks recipe binding everywhere
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
        atomic_write_json,
    )
    report_path = tmp_path / "train_report.json"
    d = json.loads(report_path.read_text())
    d["config"]["lr"] = 9.9e-4
    atomic_write_json(report_path, d)
    # refresh manifest's report digest so ONLY the recipe binding can
    # reject (not mere byte coherence)
    m = json.loads((tmp_path / RUN_MANIFEST_FILE).read_text())
    from latent_lab.train.checkpointing import sha256_file
    m["report_sha256"] = sha256_file(report_path)
    atomic_write_json(tmp_path / RUN_MANIFEST_FILE, m)
    with pytest.raises(Exception) as ei:
        artifacts.validate_run(tmp_path)
    assert "canonical recipe" in str(ei.value)

    # tampering with the MANIFEST recipe likewise
    build_verified_run(tmp_path)
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    d = json.loads(manifest_path.read_text())
    d["recipe"]["seed"] = 7
    atomic_write_json(manifest_path, d)
    with pytest.raises(Exception):
        artifacts.validate_run(tmp_path)


def test_run_validator_recognizes_sealed_adapter_activation_metadata():
    assert {"training_objective", "recurrence_only_lora", "runtime_contract",
            "neutral_delta", "paired_delta", "trace_curriculum"} <= \
        artifacts._TRAIN_CONFIG_KNOWN_KEYS


def test_run_validator_rejects_unsealed_paired_delta(tmp_path):
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE, TRAIN_REPORT_FILE, atomic_write_json, sha256_file)

    build_verified_run(tmp_path)
    report_path = tmp_path / TRAIN_REPORT_FILE
    report = json.loads(report_path.read_text())
    report["config"]["paired_delta"] = True
    atomic_write_json(report_path, report)
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["report_sha256"] = sha256_file(report_path)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid paired-delta metadata"):
        artifacts.validate_run(tmp_path)


def test_run_validator_rejects_unsealed_trace_curriculum(tmp_path):
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE, TRAIN_REPORT_FILE, atomic_write_json, sha256_file)

    build_verified_run(tmp_path)
    report_path = tmp_path / TRAIN_REPORT_FILE
    report = json.loads(report_path.read_text())
    report["config"]["trace_curriculum"] = True
    atomic_write_json(report_path, report)
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["report_sha256"] = sha256_file(report_path)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid trace-curriculum metadata"):
        artifacts.validate_run(tmp_path)


def test_run_validator_rejects_unsealed_training_objective(tmp_path):
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE, TRAIN_REPORT_FILE, atomic_write_json, sha256_file)

    build_verified_run(tmp_path)
    report_path = tmp_path / TRAIN_REPORT_FILE
    report = json.loads(report_path.read_text())
    report["config"]["training_objective"] = "candidate_ce"
    atomic_write_json(report_path, report)
    manifest_path = tmp_path / RUN_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["report_sha256"] = sha256_file(report_path)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid training objective metadata"):
        artifacts.validate_run(tmp_path)


def test_valid_but_different_bundle_cannot_claim_selected_raw_history(
        tmp_path):
    """Rebinding all bundle/file digests cannot forge selected-step state."""
    from latent_lab.train.checkpointing import (
        CHECKPOINT_FILE, RUN_MANIFEST_FILE, TRAIN_REPORT_FILE,
        save_adapter_bundle, sha256_file, write_train_generation,
    )

    build_verified_run(tmp_path)
    report = json.loads((tmp_path / TRAIN_REPORT_FILE).read_text())
    manifest = json.loads((tmp_path / RUN_MANIFEST_FILE).read_text())
    original_selected = report["selected_adapter_state_sha256"]
    replacement = {"lora.0.A": torch.ones(8, 4),
                   "lora.0.B": torch.ones(4, 8)}
    bundle = save_adapter_bundle(
        tmp_path / CHECKPOINT_FILE, replacement,
        model_id=report["model"], revision=report["revision"],
        recipe=report["recipe"], metrics={"best_val_acc": 0.5})
    assert ckpt.adapter_state_sha256(replacement) != original_selected

    # Attacker coherently refreshes every legacy bundle/file digest while
    # retaining the genuine raw validation history and its selected hash.
    report["checkpoint_content_digest"] = bundle["content_digest"]
    report["checkpoint_sha256"] = sha256_file(tmp_path / CHECKPOINT_FILE)
    manifest["checkpoint_content_digest"] = bundle["content_digest"]
    write_train_generation(tmp_path, manifest=manifest, report=report)

    with pytest.raises(ValueError, match="selected checkpoint state"):
        artifacts.validate_run(tmp_path)


# ---------------------------------------------------------------------------
# Fix 4: exact negative pins for the two reproduced false accepts
# ---------------------------------------------------------------------------

def test_run_dir_name_is_never_evidence_k8_seed2_in_E4_k4_s0(tmp_path):
    """Reproduced false accept: a directory named E4_k4_s0 whose report
    was K8/seed2. Contents-coherent evidence still PASSES bare
    validation but MUST fail the driver's FULL preregistered contract."""
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d, config=fake_cfg(k=8, seed=2, steps=400))
    # coherent contents alone are accepted when nobody declares intent...
    assert artifacts.validate_run(d)["status"] == "complete"
    # ...but the driver's full expected contract rejects it outright
    with pytest.raises(Exception) as ei:
        artifacts.validate_run(d, expected=run_contract())
    msg = str(ei.value)
    assert "mismatch" in msg
    # and the CLI form used by drivers behaves identically
    rc = run_contract()
    assert artifacts.main(
        ["validate-run", str(d), "--expect-model", rc["model_id"],
         "--expect-rev", rc["revision"], "--expect-suite",
         rc["suite_sha256"], "--expect-seed", "0", "--expect-label",
         "E4_k4_s0", "--expect-k", "4", "--expect-steps", "800",
         "--expect-config-sha256", rc["config_sha256"]]) == 1


def test_two_file_hash_manifest_with_non_bundle_checkpoint_rejected(
        tmp_path):
    """Reproduced false accept: a two-file-hash manifest over a
    non-bundle best_params.pt. Even with hashes made coherent, the
    checkpoint is not a genuine identity-bound bundle -> reject."""
    from latent_lab.train.checkpointing import (
        CHECKPOINT_FILE,
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
        atomic_write_json,
        sha256_file,
    )

    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d)
    garbage = b"\xde\xad\xbe\xef" * 64
    (d / CHECKPOINT_FILE).write_bytes(garbage)
    sha = hashlib_sha256(garbage)
    rewrite_generation_hashes(d, checkpoint_sha=sha)
    with pytest.raises(Exception) as ei:
        artifacts.validate_run(d)
    msg = str(ei.value).lower()
    assert any(tok in msg for tok in ("digest", "bundle", "cannot read"))


def hashlib_sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def rewrite_generation_hashes(run_dir, *, checkpoint_sha: str) -> None:
    """Make report+manifest digests fully coherent with the CURRENT
    checkpoint bytes (so ONLY the bundle-identity check can reject what
    follows) and with each other."""
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
        atomic_write_json,
        sha256_file,
    )
    report = json.loads((run_dir / TRAIN_REPORT_FILE).read_text())
    manifest = json.loads((run_dir / RUN_MANIFEST_FILE).read_text())
    report["checkpoint_sha256"] = checkpoint_sha
    report["checkpoint_content_digest"] = checkpoint_sha
    atomic_write_json(run_dir / TRAIN_REPORT_FILE, report)
    manifest["checkpoint_sha256"] = checkpoint_sha
    manifest["checkpoint_content_digest"] = checkpoint_sha
    manifest["report_sha256"] = sha256_file(run_dir / TRAIN_REPORT_FILE)
    atomic_write_json(run_dir / RUN_MANIFEST_FILE, manifest)


def test_validate_eval_rejects_wrong_model_rev_or_suite(tmp_path):
    ep = tmp_path / "ev.json"

    payload = build_eval_payload()
    ep.write_text(json.dumps(payload))
    with pytest.raises(Exception) as ei:
        artifacts.validate_eval(ep, expected={"model_id": "WRONG-MODEL"})
    assert "WRONG-MODEL" in str(ei.value)

    # arbitrary/malformed revision is rejected even with NO expectation
    bad_rev = build_eval_payload(identity={
        **EVAL_IDENTITY, "revision": "main"})
    ep.write_text(json.dumps(bad_rev))
    with pytest.raises(ValueError):
        artifacts.validate_eval(ep)

    # valid-but-unexpected rev/suite pairs fail the driver's contract
    ep.write_text(json.dumps(build_eval_payload()))
    with pytest.raises(Exception):
        artifacts.validate_eval(ep, expected={"revision": "ff" * 20})
    with pytest.raises(Exception):
        artifacts.validate_eval(
            ep, expected={"suite_sha256": "ee" * 32})


def test_validate_eval_rejects_accuracy_only_and_lossless_violations(
        tmp_path):
    ep = tmp_path / "ev.json"

    accuracy_only = build_eval_payload()
    accuracy_only["results"]["clean"] = {"accuracy": 0.75, "n": 1,
                                         "k_steps": 4}
    ep.write_text(json.dumps(accuracy_only))
    with pytest.raises(ValueError) as ei:
        artifacts.validate_eval(ep)
    assert "raw records" in str(ei.value)

    payload = build_eval_payload()
    payload["results"]["clean"]["records"] = []          # n=1, records=0
    ep.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        artifacts.validate_eval(ep)

    payload = build_eval_payload()
    rec = payload["results"]["clean"]["records"][0]
    rec["scores_raw"][0] = float("nan")                  # non-finite score
    ep.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        artifacts.validate_eval(ep)

    # gold NOT top-ranked: recomputed accuracy is 0.0; claiming 1.0 must
    # fail independent rescoring
    payload = build_eval_payload(
        records=[eval_record(scores=(1.0, 2.0, 0.5))])
    payload["results"]["clean"]["accuracy"] = 1.0
    ep.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as ei:
        artifacts.validate_eval(ep)
    assert "recomputed" in str(ei.value)


def test_validate_eval_requires_full_identity_block(tmp_path):
    ep = tmp_path / "ev.json"
    for drop in ("checkpoint_content_digest", "split", "ablation",
                 "k_steps", "seed", "suite_sha256", "model_id",
                 "revision"):
        payload = build_eval_payload()
        del payload["identity"][drop]
        ep.write_text(json.dumps(payload))
        with pytest.raises(ValueError) as ei:
            artifacts.validate_eval(ep)
        assert drop in str(ei.value)


def _current_v3_eval_payload(**record_overrides):
    from latent_lab.bench.eval_v3 import aggregate_records, build_eval_record
    from latent_lab.bench.suite_v3 import build_suite

    suite = build_suite()
    example = suite.test_id[0]
    payload = build_eval_payload(identity={
        **EVAL_IDENTITY,
        "suite_sha256": suite.records_hash(),
    })
    ident = payload["identity"]
    metadata = {
        "run_id": "run-v3", "recipe_hash": "a" * 64,
        "model_id": ident["model_id"], "model_revision": ident["revision"],
        "adapter_id": payload["adapter"], "checkpoint_id": "best_params.pt",
        "checkpoint_content_hash": ident["checkpoint_content_digest"],
        "suite_id": "behavioral-v3", "suite_version": 3,
        "suite_hash": ident["suite_sha256"], "example_id": example.ex_id,
        "split": example.split, "family": example.family,
        "prompt": example.prompt, "candidates": example.candidates,
        "candidate_permutation_seed": example.candidate_permutation_seed,
        "candidate_permutation": example.candidate_permutation,
        "gold_answer": example.answer, "k": ident["k_steps"],
        "recurrence_config": {"interval": [12, 18],
                              "gradient_semantics": "truncated_cache"},
        "compute": {
            "prefill_layers": 12,
            "recurrence_interval_applications": 24,
            "k_loops": 4,
            "candidate_tail_layers": len(example.candidates) * 6,
            "lm_head_calls": len(example.candidates),
            "tokenizer_calls": len(example.candidates) + 1,
            "decode_calls": 0,
            "wall_seconds": 0.1,
            "peak_memory_bytes": None,
            "successful_task": True,
            "eval_ablation": {},
        },
    }
    metadata.update(record_overrides)
    scores = tuple(
        (-0.1,) if candidate == metadata["gold_answer"] else (-2.0,)
        for candidate in metadata["candidates"]
    )
    record = build_eval_record(per_token_logprobs=scores, **metadata)
    payload["results"]["clean"].update(
        n=1, metrics=aggregate_records([record]), records=[record])
    payload["results"]["clean"].pop("accuracy")
    return payload, record, example


def test_validate_eval_accepts_only_self_consistent_latent_eval_v3(tmp_path):
    payload, _, _ = _current_v3_eval_payload()
    ep = tmp_path / "eval-v3.json"
    ckpt.atomic_write_json(ep, payload)
    assert artifacts.validate_eval(ep) == payload

    payload["identity"]["checkpoint_content_digest"] = "0" * 64
    ckpt.atomic_write_json(ep, payload)
    with pytest.raises(ValueError, match="content_sha256"):
        artifacts.validate_eval(ep)


def test_validate_eval_rejects_relabelled_v3_ablation(tmp_path):
    payload, _, _ = _current_v3_eval_payload()
    result = payload["results"].pop("clean")
    payload["identity"]["ablation"] = "zero_state"
    result["ablate"] = {"zero_state": True}
    result["tag"] = "E-localized|test_id|zero_state|K=4"
    payload["results"]["zero_state"] = result
    ep = tmp_path / "relabelled-ablation.json"
    ckpt.atomic_write_json(ep, payload)

    with pytest.raises(ValueError, match="compute.eval_ablation"):
        artifacts.validate_eval(ep)


def test_validate_eval_binds_ablation_label_to_sealed_runtime_spec(tmp_path):
    _, clean_record, _ = _current_v3_eval_payload()
    noise_compute = {
        **clean_record["compute"], "eval_ablation": {"noise_state": True},
    }
    payload, _, _ = _current_v3_eval_payload(compute=noise_compute)
    result = payload["results"].pop("clean")
    payload["identity"]["ablation"] = "zero_state"
    result["ablate"] = {"noise_state": True}
    result["tag"] = "E-localized|test_id|zero_state|K=4"
    payload["results"]["zero_state"] = result
    ep = tmp_path / "wrong-ablation-label.json"
    ckpt.atomic_write_json(ep, payload)

    with pytest.raises(ValueError, match="requires.*zero_state"):
        artifacts.validate_eval(ep)


def test_validate_eval_rejects_schema_valid_fabricated_current_suite_record(
        tmp_path):
    _, _, example = _current_v3_eval_payload()
    fabricated_candidates = tuple(
        f"fabricated-{index}" for index in range(len(example.candidates)))
    payload, _, _ = _current_v3_eval_payload(
        family="fabricated-family",
        prompt="fabricated prompt",
        candidates=fabricated_candidates,
        candidate_permutation_seed=-777,
        candidate_permutation=tuple(reversed(range(len(fabricated_candidates)))),
        gold_answer=fabricated_candidates[
            (example.gold_index + 1) % len(fabricated_candidates)],
    )
    ep = tmp_path / "malicious-v3.json"
    ckpt.atomic_write_json(ep, payload)
    with pytest.raises(ValueError, match="canonical behavioral-v3 fields") \
            as exc:
        artifacts.validate_eval(ep)
    message = str(exc.value)
    for field in (
        "family", "prompt_hash", "candidates",
        "candidate_permutation_seed", "candidate_permutation",
        "gold_answer", "gold_index",
    ):
        assert field in message


def test_validate_eval_binds_expected_split_ablation_k_seed(tmp_path):
    ep = tmp_path / "ev.json"
    ep.write_text(json.dumps(build_eval_payload()))
    ok = {"split": "test_id", "ablation": "clean", "k": 4, "seed": 0}
    assert artifacts.validate_eval(ep, expected=ok) is not None
    for field, bad in (("split", "test_ood"), ("ablation", "zero_state"),
                       ("k", 0), ("seed", 1)):
        with pytest.raises(Exception) as ei:
            artifacts.validate_eval(ep, expected={**ok, field: bad})
        assert "mismatch" in str(ei.value)


def test_cmd_eval_identity_block_covers_the_required_key_set():
    """Drift guard: whatever validate_eval requires of identity.k_steps/
    split/ablation/..., the eval driver must actually persist."""
    from latent_lab.bench.artifacts import _EVAL_IDENTITY_KEYS

    src = Path(latent_run.__file__).read_text()
    payload_start = src.index("def build_v3_eval_payload")
    payload_end = src.index("def _dependency_versions")
    region = src[payload_start:payload_end]
    cmd_region = src[src.index("def cmd_eval"):]
    assert "build_v3_eval_payload(" in cmd_region
    start = region.index('"identity": {')
    depth = 0
    for i in range(start, len(region)):
        if region[i] == "{":
            depth += 1
        elif region[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = region[start:end]
    for key in _EVAL_IDENTITY_KEYS:
        assert key in block, \
            f"cmd_eval identity omits required field {key!r}"


# ---------------------------------------------------------------------------
# Repair cycle 2 — Blocker 1: fatal lifecycle can never leave
# complete-looking evidence at the active root
# ---------------------------------------------------------------------------

def _train_args(tmp_path):
    return SimpleNamespace(revision=REV_OK, device="cpu", model=MODEL,
                           seed=0, out=str(tmp_path / "run"))


def _late_writer(payload):
    def late(args, out, device, revision):
        build_verified_run(Path(out))
        raise RuntimeError(payload)
    return late


def test_cmd_train_late_failure_quarantines_complete_looking_evidence(
        tmp_path, monkeypatch):
    """Exact repro 1: a late exception from _train_inner AFTER it wrote
    train_report.json/best_params.pt/run_manifest.json must leave NO
    complete-looking artifact at the evidence root."""
    from latent_lab.train.checkpointing import (
        CHECKPOINT_FILE,
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
    )
    out = tmp_path / "run"

    monkeypatch.setattr(latent_run, "_train_inner",
                        _late_writer("late boom after generation"))
    with pytest.raises(RuntimeError, match="late boom"):
        latent_run.cmd_train(_train_args(tmp_path))

    st = ckpt.read_run_status(out)
    assert st is not None and st["status"] == "fatal"
    for name in (TRAIN_REPORT_FILE, CHECKPOINT_FILE, RUN_MANIFEST_FILE):
        assert not (out / name).exists(), name
    quarantined = sorted(p.name for p in out.glob("*"))
    assert sum(".invalid." in n for n in quarantined) >= 4, quarantined
    with pytest.raises(Exception):
        ckpt.verify_generation(out)
    with pytest.raises(Exception):
        artifacts.validate_run(out)


def test_cmd_train_final_complete_status_write_failure_fails_closed(
        tmp_path, monkeypatch):
    """Exact repro 2: the terminal complete-status write sits INSIDE the
    protected lifecycle; its failure must never leave 'running' beside
    completed evidence and must re-raise the ORIGINAL error."""
    out = tmp_path / "run"
    out.mkdir(parents=True)
    build_verified_run(out)
    real = ckpt.write_run_status

    def disk_full(out_dir, status, **kw):
        if status == "complete":
            raise OSError("disk full at terminal transition")
        return real(out_dir, status, **kw)

    monkeypatch.setattr(ckpt, "write_run_status", disk_full)
    monkeypatch.setattr(latent_run, "_train_inner",
                        lambda *a, **k: None)
    with pytest.raises(OSError, match="disk full"):
        latent_run.cmd_train(_train_args(tmp_path))

    st = ckpt.read_run_status(out)
    assert st is not None and st["status"] == "fatal"
    assert st["error_type"] == "OSError"
    for name in (ckpt.TRAIN_REPORT_FILE, ckpt.CHECKPOINT_FILE,
                 ckpt.RUN_MANIFEST_FILE):
        assert not (out / name).exists(), f"{name} survived"
        assert list(out.glob(f"{name}.invalid.*")), name
    with pytest.raises(Exception):
        artifacts.validate_run(out)


def test_cmd_train_cleanup_failure_escalates_to_whole_root_quarantine(
        tmp_path, monkeypatch):
    """Per-artifact purge failure => the ENTIRE root is moved aside and
    a fresh root carries only fatal status; the ORIGINAL exception is
    re-raised unchanged."""
    out = tmp_path / "run"
    monkeypatch.setattr(latent_run, "_train_inner",
                        _late_writer("original boom"))

    def broken_purge(root):
        raise OSError("purge boom")

    monkeypatch.setattr(ckpt, "quarantine_success_artifacts", broken_purge)
    with pytest.raises(RuntimeError, match="original boom"):
        latent_run.cmd_train(_train_args(tmp_path))

    moved = list(tmp_path.glob("run.invalid.*"))
    assert moved and (moved[0] / ckpt.RUN_STATUS_FILE).exists(), \
        "complete-looking generation stayed in an active root"
    st = ckpt.read_run_status(out)
    assert st is not None and st["status"] == "fatal"
    assert list(out.iterdir()) == [out / ckpt.RUN_STATUS_FILE]
    with pytest.raises(Exception):
        ckpt.verify_generation(out)


def test_cmd_train_double_cleanup_failure_escalates_preserving_original(
        tmp_path, monkeypatch):
    """When BOTH per-artifact quarantine AND whole-root escalation fail,
    fail closed with an explicit lifecycle error whose cause is the
    ORIGINAL exception — never a secondary cleanup error. The ACTIVE
    root must additionally no longer validate as a complete generation:
    the status mark is atomically poisoned BEFORE best-effort artifact
    handling, so verify_generation/validate_run reject even in this
    exact double-failure repro."""
    out = tmp_path / "run"
    monkeypatch.setattr(latent_run, "_train_inner",
                        _late_writer("original boom"))

    def broken_purge(root):
        raise OSError("purge boom")

    monkeypatch.setattr(ckpt, "quarantine_success_artifacts", broken_purge)
    real_replace = os.replace

    def broken_rename(a, b, *r, **kw):
        if "run.invalid." in str(b):
            raise OSError("rename refused")
        return real_replace(a, b, *r, **kw)

    monkeypatch.setattr(os, "replace", broken_rename)
    with pytest.raises(ckpt.EvidenceLifecycleError) as ei:
        latent_run.cmd_train(_train_args(tmp_path))
    cause = ei.value.__cause__
    assert isinstance(cause, RuntimeError) and "original boom" in str(cause)
    assert "failing closed" in str(ei.value)
    assert not list((tmp_path).glob("run.invalid.*")), \
        "escalation claimed success it did not have"

    # fail-stop invariant: the active root can NEVER validate as
    # complete after any caught late/final failure — not even when both
    # cleanup and fatal-status publication also failed
    st = ckpt.read_run_status(out)
    assert st is not None and st["status"] != "complete", \
        "active root still carries a completion mark"
    assert st["error_type"] == "RuntimeError"
    assert "original boom" in st["error"]
    with pytest.raises(Exception):
        ckpt.verify_generation(out)
    with pytest.raises(Exception):
        artifacts.validate_run(out)


def test_cmd_train_fatal_publication_failure_leaves_no_generation(
        tmp_path, monkeypatch):
    """Fatal-status write failure after quarantine => fail closed with
    the original exception preserved as cause."""
    out = tmp_path / "run"
    monkeypatch.setattr(latent_run, "_train_inner",
                        _late_writer("boom before fatal"))

    def broken_fatal(out_p, e):
        raise OSError("no space for fatal status")

    monkeypatch.setattr(latent_run, "_mark_run_fatal", broken_fatal)
    with pytest.raises(ckpt.EvidenceLifecycleError) as ei:
        latent_run.cmd_train(_train_args(tmp_path))
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert not (out / ckpt.RUN_STATUS_FILE).exists()
    with pytest.raises(Exception):
        ckpt.verify_generation(out)


def test_cmd_train_success_publishes_complete_validating_generation(
        tmp_path, monkeypatch):
    """Control: normal success publishes a generation that validates
    under the FULL preregistered contract."""
    out = tmp_path / "run"
    monkeypatch.setattr(latent_run, "_train_inner",
                        lambda args, o, device, revision:
                        build_verified_run(Path(o)))
    latent_run.cmd_train(_train_args(tmp_path))
    assert ckpt.require_complete_run(out)["status"] == "complete"
    manifest = artifacts.validate_run(out, expected=run_contract())
    assert manifest["status"] == "complete"


# ---------------------------------------------------------------------------
# Repair cycle 2 — Blocker 4: strict JSON everywhere in evidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_atomic_write_json_refuses_non_finite_values(tmp_path, bad):
    p = tmp_path / "x.json"
    for payload in ({"v": bad}, {"nested": [{"deep": {"v": bad}}]}):
        with pytest.raises(ValueError):
            ckpt.atomic_write_json(p, payload)
    assert not p.exists()


def test_strict_reader_rejects_nan_infinity_constants():
    for const in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="non-standard JSON constant"):
            ckpt.strict_json_loads('{"final_train_loss": %s}' % const)
        with pytest.raises(ValueError, match="non-standard JSON constant"):
            ckpt.strict_json_loads('{"wall_seconds": [%s]}' % const)


def test_decoded_object_scan_rejects_non_finite_anywhere():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(Exception, match="non-finite number"):
            ckpt.assert_json_numbers_finite(
                {"a": [1, {"b": {"c": bad}}], "d": [0.5]})


def test_validate_run_rejects_nan_and_infinity_in_report(tmp_path):
    """Reproduced accept: final_train_loss=NaN and wall_seconds=Infinity
    written as literal JSON constants must fail validation even when the
    manifest byte-hashes were refreshed around them."""
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
        sha256_file,
    )

    for field, bad_float in (("final_train_loss", float("nan")),
                             ("wall_seconds", float("inf"))):
        build_verified_run(tmp_path)
        rp = tmp_path / TRAIN_REPORT_FILE
        raw = json.loads(rp.read_text())
        # permissive dump writes the literal NaN/Infinity constant
        raw[field] = bad_float
        rp.write_text(json.dumps(raw))
        m = json.loads((tmp_path / RUN_MANIFEST_FILE).read_text())
        m["report_sha256"] = sha256_file(rp)     # keep ONLY finiteness able to reject
        ckpt.atomic_write_json(tmp_path / RUN_MANIFEST_FILE, m)
        with pytest.raises(ValueError) as ei:
            artifacts.validate_run(tmp_path)
        msg = str(ei.value)
        assert "non-standard JSON constant" in msg \
            or "non-finite number" in msg, msg
        assert ("NaN" in msg) == (field == "final_train_loss")


def test_read_run_status_treats_non_finite_status_as_unusable(tmp_path):
    p = tmp_path / ckpt.RUN_STATUS_FILE
    p.write_text(json.dumps({"status": "complete",
                             "wall_seconds": float("inf")}))
    assert ckpt.read_run_status(tmp_path) is None


def test_validate_eval_rejects_non_finite_numbers_nested(tmp_path):
    ep = tmp_path / "ev.json"

    def mutate(d):
        d["results"]["clean"]["seconds"] = float("nan")
        d["results"]["clean"]["records"][0]["second_best"] = -float("inf")

    payload = build_eval_payload()
    mutate(payload)
    ep.write_text(json.dumps(payload))            # literal NaN/Inf on disk
    with pytest.raises(ValueError):
        artifacts.validate_eval(ep)

    # decoded-object layer: the same tree rejected without any parse step
    d = build_eval_payload()
    mutate(d)
    from latent_lab.train.checkpointing import assert_json_numbers_finite
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_json_numbers_finite(d, where="payload")


# ---------------------------------------------------------------------------
# Repair cycle 2 — Blocker 2: resume bound to the FULL canonical recipe
# ---------------------------------------------------------------------------

def test_resume_rejects_old_partial_contract_structurally(tmp_path):
    """The old allowlist omitted LR/interval; partial intent must reject.

    The fixture stays on the current suite so this test reaches the expected-
    contract shape check instead of correctly failing earlier at suite
    evidence binding.
    """
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d, config=fake_cfg(lr=9e-4, interval=[6, 18]))
    stale_partial = {"model_id": MODEL, "revision": REV_OK, "seed": 0,
                     "label": "E4_k4_s0", "k": 4, "steps": 800}
    with pytest.raises(ValueError) as ei:
        artifacts.validate_run(d, expected=dict(stale_partial))
    assert "incomplete expected contract" in str(ei.value)
    assert "config_sha256" in str(ei.value)
    # the FULL preregistered contract rejects the drifted artifact
    with pytest.raises(ValueError) as ei2:
        artifacts.validate_run(d, expected=run_contract())
    assert "mismatch" in str(ei2.value)


@pytest.mark.parametrize("over", [
    ("suite", dict(suite_sha256="ee" * 32)),
    ("lr", dict(lr=5e-5)),
    ("interval", dict(interval=[10, 14])),
    ("lora_r", dict(lora_r=16)),
    ("lora_alpha", dict(lora_alpha=32.0)),
    ("optimizer", dict(optimizer="sgd")),
    ("weight_decay", dict(weight_decay=0.0)),
    ("lr_schedule", dict(lr_schedule="cosine")),
    ("warmup", dict(warmup=10)),
    ("clip", dict(clip=1.0)),
    ("detach_z0", dict(detach_z0=True)),
    ("k", dict(k=8)),
    ("max_k", dict(max_k=8)),
    ("steps", dict(steps=400)),
    ("seed", dict(seed=2)),
    ("mode", dict(mode="D-full")),
])
def test_resume_binds_every_recipe_category_via_canonical_digest(
        tmp_path, over):
    name, cfg_over = over
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d, config=fake_cfg(**cfg_over))
    with pytest.raises(Exception) as ei:
        artifacts.validate_run(d, expected=run_contract())
    msg = str(ei.value)
    if name == "suite":
        assert "canonical behavioral-v3" in msg
    else:
        assert "mismatch" in msg or "canonical training identity" in msg


def test_full_preregistered_contract_accepts_matching_artifact(tmp_path):
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d)
    manifest = artifacts.validate_run(d, expected=run_contract())
    digest = manifest["checkpoint_content_digest"]
    assert artifacts.validate_run(
        d, expected={**run_contract(),
                     "checkpoint_content_digest": digest}) is not None


def test_unexpected_expectation_keys_are_rejected(tmp_path):
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d)
    with pytest.raises(ValueError, match="unexpected expectation keys"):
        artifacts.validate_run(d, expected={**run_contract(),
                                            "flavor": "vanilla"})
    ep = tmp_path / "ev.json"
    ep.write_text(json.dumps(build_eval_payload()))
    ok = {"model_id": MODEL, "revision": REV_OK, "suite_sha256": SUITE_SHA,
          "checkpoint_content_digest": "cd" * 32, "split": "test_id",
          "ablation": "clean", "k": 4, "seed": 0}
    with pytest.raises(ValueError, match="unexpected expectation keys"):
        artifacts.validate_eval(ep, expected={**ok, "flavor": "x"})


def test_unexpected_config_metadata_is_rejected_not_ignored(tmp_path):
    from latent_lab.train.checkpointing import (
        RUN_MANIFEST_FILE,
        TRAIN_REPORT_FILE,
        sha256_file,
    )
    build_verified_run(tmp_path)
    rp = tmp_path / TRAIN_REPORT_FILE
    d = json.loads(rp.read_text())
    d["config"]["quantization"] = "none"          # unexpected alias/metadata
    ckpt.atomic_write_json(rp, d)
    m = json.loads((tmp_path / RUN_MANIFEST_FILE).read_text())
    m["report_sha256"] = sha256_file(rp)
    ckpt.atomic_write_json(tmp_path / RUN_MANIFEST_FILE, m)
    with pytest.raises(ValueError, match="unexpected config metadata"):
        artifacts.validate_run(tmp_path)


def test_cli_supports_expect_config_sha256_flag(tmp_path):
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d)
    rc = run_contract()
    flags = ["--expect-model", rc["model_id"], "--expect-rev",
             rc["revision"], "--expect-suite", rc["suite_sha256"],
             "--expect-seed", "0", "--expect-label", "E4_k4_s0",
             "--expect-k", "4", "--expect-steps", "800"]
    assert artifacts.main(["validate-run", str(d), *flags]) == 1  # incomplete
    assert artifacts.main(["validate-run", str(d), *flags,
                           "--expect-config-sha256",
                           rc["config_sha256"]]) == 0
    assert artifacts.main(["validate-run", str(d), *flags,
                           "--expect-config-sha256", "ab" * 32]) == 1


def test_train_recipe_digest_matches_trainer_recipe_binding():
    """Driver preregistration helper and the trainer's recipe agree; any
    semantic drift produces a different digest."""
    base = latent_run.train_recipe_digest(
        mode="E-localized", interval=[12, 18], k=4, max_k=16, lora_r=8,
        lora_alpha=16.0, lr=1e-4, steps=800, seed=0, optimizer="adamw",
        weight_decay=0.01, lr_schedule="constant", warmup=50, clip=0.5,
        detach_z0=False, suite_sha256=SUITE_SHA)
    assert base == run_contract()["config_sha256"]
    drift = latent_run.train_recipe_digest(
        mode="E-localized", interval=[12, 18], k=4, max_k=16, lora_r=8,
        lora_alpha=16.0, lr=3e-4, steps=800, seed=0, optimizer="adamw",
        weight_decay=0.01, lr_schedule="constant", warmup=50, clip=0.5,
        detach_z0=False, suite_sha256=SUITE_SHA)
    assert drift != base
    assert latent_run.mode_from_spec("full", 4) == "D-full"
    assert latent_run.mode_from_spec("mid", 0) == "F-control"
    assert latent_run.mode_from_spec("mid", 4) == "E-localized"


def test_in_memory_adapter_state_digest_binds_exact_tensor_bytes():
    from latent_lab.train.checkpointing import adapter_state_sha256

    left = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    same_different_order = {"a": torch.tensor([1.0]),
                            "b": torch.tensor([2.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([2.5])}
    assert adapter_state_sha256(left) == \
        adapter_state_sha256(same_different_order)
    assert adapter_state_sha256(left) != adapter_state_sha256(changed)


# ---------------------------------------------------------------------------
# Repair cycle 2 — Blocker 3: eval validation reconciles EVERY duplicated
# identity field (exact hostile repro + per-field regressions + controls)
# ---------------------------------------------------------------------------

_EVAL_OK_EXPECTED = {"model_id": MODEL, "revision": REV_OK,
                     "suite_sha256": SUITE_SHA,
                     "checkpoint_content_digest": "cd" * 32,
                     "split": "test_id", "ablation": "clean", "k": 4,
                     "seed": 0}


def _mutated_payload(mut):
    d = build_eval_payload()
    mut(d)
    return d


@pytest.mark.parametrize("name,mut", [
    ("top_model", lambda d: d.update(model="WRONG-MODEL")),
    ("top_revision_40_zeroes", lambda d: d.update(revision="0" * 40)),
    ("top_suite", lambda d: d.update(suite_sha256="12" * 32)),
    ("top_split", lambda d: d.update(split="test_ood")),
    ("top_seed", lambda d: d.update(seed=2)),
    ("config_model", lambda d: d.update(config={"model": "Other/Model"})),
    ("config_interval", lambda d: d.update(config={"interval": [6, 18]})),
    ("config_max_k", lambda d: d.update(config={"max_k": 8})),
    ("result_key_alias", lambda d: d.update(
        results={"zero_state": d["results"]["clean"]})),
    ("extra_result_alias", lambda d: d["results"].update(
        {"bypass_interval": dict(d["results"]["clean"])})),
    ("result_k_steps", lambda d:
        d["results"]["clean"].update(k_steps=8)),
    ("tag_garbage_ffn_K0", lambda d:
        d["results"]["clean"].update(tag="ffn/K0")),
    ("tag_wrong_split", lambda d: d["results"]["clean"].update(
        tag="E-localized|test_ood|clean|K=4")),
    ("tag_wrong_k", lambda d: d["results"]["clean"].update(
        tag="E-localized|test_id|clean|K=8")),
    ("clean_carries_ablate", lambda d: d["results"]["clean"].update(
        ablate={"zero_state": True})),
])
def test_validate_eval_reconciles_every_duplicated_field(tmp_path, name,
                                                          mut):
    ep = tmp_path / "ev.json"
    ep.write_text(json.dumps(_mutated_payload(mut)))
    with pytest.raises(ValueError) as ei:
        artifacts.validate_eval(ep)
    msg = str(ei.value).lower()
    assert any(tok in msg for tok in
               ("contradict", "unexpected selected results",
                "no matching results entry", "not the canonical")), msg


def test_validate_eval_distinguishes_training_k_seed_from_eval_arm(tmp_path):
    payload, _, _ = _current_v3_eval_payload()
    payload["config"] = {"k": 1, "seed": 99}
    ep = tmp_path / "different-training-k-seed.json"
    ckpt.atomic_write_json(ep, payload)

    assert artifacts.validate_eval(ep) == payload


def test_validate_eval_exact_hostile_contradictory_payload_rejected(
        tmp_path):
    """The independently reproduced payload: nested identity matches the
    expected Qwen/test_id/clean/K4/seed0/digest while EVERY top-level
    duplicate contradicts it."""
    ep = tmp_path / "ev.json"
    hostile = build_eval_payload()
    hostile.update({
        "model": "WRONG-MODEL",
        "revision": "0" * 40,
        "suite_sha256": "12" * 32,
        "split": "test_ood",
        "seed": 2,
        "config": {"k": 0},
    })
    res = hostile["results"]["clean"]
    res["tag"] = "ffn/K0"
    res["k_steps"] = 0
    ep.write_text(json.dumps(hostile))
    with pytest.raises(ValueError):
        artifacts.validate_eval(ep)
    # even with NO expectation supplied the contradictions reject
    with pytest.raises(ValueError):
        artifacts.validate_eval(ep, expected=None)


def test_validate_eval_normal_valid_controls_pass(tmp_path):
    ep = tmp_path / "ev.json"
    ep.write_text(json.dumps(build_eval_payload()))
    d = artifacts.validate_eval(ep)
    assert d["status"] == "complete"
    assert artifacts.validate_eval(ep, expected=_EVAL_OK_EXPECTED) \
        is not None


def test_validate_eval_extended_identity_fields_are_well_formed(tmp_path):
    ep = tmp_path / "ev.json"
    bad_interval = build_eval_payload(identity={
        **EVAL_IDENTITY, "interval": [18]})
    ep.write_text(json.dumps(bad_interval))
    with pytest.raises(ValueError, match="identity.interval"):
        artifacts.validate_eval(ep)
    bad_max_k = build_eval_payload(identity={**EVAL_IDENTITY, "max_k": 0})
    ep.write_text(json.dumps(bad_max_k))
    with pytest.raises(ValueError, match="identity.max_k"):
        artifacts.validate_eval(ep)


# ---------------------------------------------------------------------------
# Regression: artifact validation enforces candidates[gold] == answer —
# rescoring derives gold identity from answer/candidates and rejects a
# missing, ambiguous/duplicated or substituted gold instead of trusting
# the supplied index.
# ---------------------------------------------------------------------------

def _stealthy_substituted_gold_record():
    """A record whose index names the top-scored rival while rank_of_gold/
    correct/accuracy are rewritten to stay self-consistent: index-trusting
    scoring accepts this wholesale."""
    rec = eval_record()                # candidates c0..c2, answer c0, gold 0
    assert rec["answer"] == "c0"
    rec["gold_candidate_index"] = 1    # substitute
    rec["rank_of_gold"] = 0            # keep derived fields consistent...
    rec["correct"] = 1.0               # ...so ONLY content binding catches it
    return rec


def test_validate_eval_rejects_substituted_gold_identity(tmp_path):
    ep = tmp_path / "ev.json"
    ep.write_text(json.dumps(build_eval_payload(
        records=[_stealthy_substituted_gold_record()])))
    with pytest.raises(ValueError) as ei:
        artifacts.validate_eval(ep)
    assert "gold_candidate_index" in str(ei.value)


def test_validate_eval_rejects_missing_and_duplicated_gold(tmp_path):
    missing = eval_record()
    missing["answer"] = "zzz"          # absent from candidates
    ep = tmp_path / "ev_missing.json"
    ep.write_text(json.dumps(build_eval_payload(records=[missing])))
    with pytest.raises(ValueError, match="missing from"):
        artifacts.validate_eval(ep)

    dup = eval_record(scores=(2.0, 1.0))
    dup["candidates"] = ["c0", "c0"]   # gold present twice: ambiguous
    dup["score_order"] = [0, 1]
    ep = tmp_path / "ev_dup.json"
    ep.write_text(json.dumps(build_eval_payload(records=[dup])))
    with pytest.raises(ValueError, match="duplicated"):
        artifacts.validate_eval(ep)


# ---------------------------------------------------------------------------
# Regression: unknown ablations such as 'bogus' are rejected by the
# artifact validator EXACTLY as by the CLI via ONE shared whitelist/
# normalizer (normalize_ablation).
# ---------------------------------------------------------------------------

def test_validate_eval_rejects_unknown_ablation_like_cli(tmp_path):
    ablation = "bogus"
    payload = build_eval_payload(identity={
        **EVAL_IDENTITY, "ablation": ablation})
    # results are already keyed/tagged self-consistently with 'bogus':
    # ONLY the shared whitelist can reject this payload
    assert list(payload["results"]) == [ablation]
    payload["results"][ablation]["tag"] = \
        f"E-localized|test_id|{ablation}|K=4"
    ep = tmp_path / "ev.json"
    ep.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as ei:
        artifacts.validate_eval(ep)
    assert "unknown ablation 'bogus'" in str(ei.value)
    # identical rejection by the CLI parser for the same name
    with pytest.raises(ValueError) as cli_ei:
        latent_run.parse_ablation_cli(ablation, 4)
    assert str(cli_ei.value) == str(ei.value).split(
        "identity.ablation rejected: ", 1)[-1]


@pytest.mark.parametrize("name,k,spec", [
    (None, 4, None),
    ("clean", 4, None),
    ("zero_state", 4, {"zero_state": True}),
    ("bypass_interval", 4, {"bypass_interval": True}),
    ("swap_state", 4, {"swap_state": True}),
    ("noise_state", 4, {"noise_state": True}),
    ("clocks_off", 4, {"clocks": "off"}),
    ("reverse_clocks", 4, {"clocks": "reverse"}),
    ("shuffle_clocks:2,0,1", 3, {"clocks": "shuffle_perm:2,0,1"}),
    ("truncate_half", 5, {"truncate_k": 2}),
    ("readout_reset_to_z0", 4, {"reset_state": True}),
    ("cache_reset_to_prompt", 4, {"reset_cache": True}),
    ("full_state_reset_to_z0", 4,
     {"reset_state": True, "reset_cache": True}),
])
def test_shared_normalizer_whitelist_used_by_cli_and_validator(name, k, spec):
    assert latent_run.normalize_ablation(name, k) == spec
    assert latent_run.parse_ablation_cli(name, k) == spec


@pytest.mark.parametrize("name", ["make_it_better", "shuffle:0,1", "",
                                  "truncate_half "])
def test_shared_normalizer_rejects_unknown_names_identically(name):
    import re

    with pytest.raises(ValueError) as norm_ei:
        latent_run.normalize_ablation(name, 4)
    with pytest.raises(ValueError) as cli_ei:
        latent_run.parse_ablation_cli(name, 4)
    assert str(norm_ei.value) == str(cli_ei.value)
    assert re.search(r"unknown ablation", str(norm_ei.value))


def test_validate_eval_accepts_every_whitelisted_ablation_name(tmp_path):
    for name, ablate in (("zero_state", {"zero_state": True}),
                         ("clocks_off", {"clocks": "off"}),
                         ("truncate_half", {"truncate_k": 2})):
        payload = build_eval_payload(identity={
            **EVAL_IDENTITY, "ablation": name})
        res = payload["results"][name]
        res["ablate"] = ablate
        res["tag"] = f"E-localized|test_id|{name}|K=4"
        ep = tmp_path / f"ev_{name}.json"
        ep.write_text(json.dumps(payload))
        assert artifacts.validate_eval(ep) is not None
