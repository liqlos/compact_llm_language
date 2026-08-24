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
    ("grad_checkpoint", {"grad_checkpoint": False}),
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
    for drop in ("lr", "seed", "warmup", "clip", "optimizer",
                 "grad_checkpoint"):
        broken = {k: v for k, v in _BASE_RECIPE_CFG.items() if k != drop}
        with pytest.raises(AdapterBundleSchemaError):
            ckpt.recipe_from_config(broken, SUITE_SHA)


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


# ---------------------------------------------------------------------------
# Fix 4: exact negative pins for the two reproduced false accepts
# ---------------------------------------------------------------------------

def test_run_dir_name_is_never_evidence_k8_seed2_in_E4_k4_s0(tmp_path):
    """Reproduced false accept: a directory named E4_k4_s0 whose report
    was K8/seed2. Contents-coherent evidence still PASSES bare
    validation but MUST fail the driver's preregistered contract."""
    d = _run_dir_named(tmp_path, "E4_k4_s0")
    build_verified_run(d, config=fake_cfg(k=8, seed=2, steps=400))
    # coherent contents alone are accepted when nobody declares intent...
    assert artifacts.validate_run(d)["status"] == "complete"
    # ...but the driver's expected contract rejects it outright
    with pytest.raises(Exception) as ei:
        artifacts.validate_run(d, expected={"label": "E4_k4_s0", "seed": 0,
                                            "k": 4})
    msg = str(ei.value)
    assert "mismatch" in msg
    # and the CLI form used by drivers behaves identically
    assert artifacts.main(["validate-run", str(d), "--expect-label",
                           "E4_k4_s0", "--expect-seed", "0",
                           "--expect-k", "4"]) == 1


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
    region = src[src.index("def cmd_eval"):]
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
