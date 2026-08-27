"""Static/mock contract tests for the unexecuted R1 experiment driver."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "latent_lab" / "bench" / "r1_experiment_driver.sh"
SUITE_HASH = "5cf5cbf397510ba597b59f7ccf0839cf344e6fb795a5cb29d031f39dac218254"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DRIVER_SHA256 = hashlib.sha256(DRIVER.read_bytes()).hexdigest()
TEXT_PRODUCER = ROOT / "latent_lab" / "bench" / "text_baselines.py"
TEXT_PRODUCER_SHA256 = hashlib.sha256(TEXT_PRODUCER.read_bytes()).hexdigest()
RUNTIME_SOURCE_SHA256 = {
    "artifact_validator": hashlib.sha256(
        (ROOT / "latent_lab" / "bench" / "artifacts.py").read_bytes()).hexdigest(),
    "eval_v3": hashlib.sha256(
        (ROOT / "latent_lab" / "bench" / "eval_v3.py").read_bytes()).hexdigest(),
    "latent_run": hashlib.sha256(
        (ROOT / "latent_lab" / "bench" / "latent_run.py").read_bytes()).hexdigest(),
    "localized_runtime": hashlib.sha256(
        (ROOT / "latent_lab" / "backends" / "localized.py").read_bytes()).hexdigest(),
}
SPLITS = (
    "validation", "test_id", "test_ood_length", "test_ood_semantic",
    "final_test",
)


def _driver_namespace():
    source = DRIVER.read_text(encoding="utf-8")
    body = source.split("# BEGIN_R1_EMBEDDED_DRIVER", 1)[1].split(
        "# END_R1_EMBEDDED_DRIVER", 1)[0]
    namespace = {"__name__": "r1_driver_contract_test"}
    exec(compile(body, str(DRIVER), "exec"), namespace)
    return namespace


def _build_plan(tmp_path):
    namespace = _driver_namespace()
    return namespace, namespace["build_plan"](
        seeds=(0, 1, 2),
        device="cuda",
        out=tmp_path / "future-output",
        suite_hash=SUITE_HASH,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        python_bin="/frozen/python",
        driver_source_sha256=DRIVER_SHA256,
        text_producer_source_sha256=TEXT_PRODUCER_SHA256,
        runtime_source_hashes=RUNTIME_SOURCE_SHA256,
    )


def _option(argv, flag):
    return argv[argv.index(flag) + 1]


def test_driver_is_shell_syntax_valid_and_refuses_without_authority(tmp_path):
    syntax = subprocess.run(
        ["bash", "-n", str(DRIVER)], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    out = tmp_path / "must-not-exist"
    env = os.environ.copy()
    env.pop("RCC_PAID_SPEND_AUTHORIZED", None)
    env.pop("RCC_R1_PREREG_ACK", None)
    env["PYTHON"] = "/definitely/missing/r1-python"
    refused = subprocess.run(
        [
            "bash", str(DRIVER), "--seeds", "0,1,2", "--device", "cuda",
            "--out", str(out),
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert refused.returncode == 78
    assert "PAID_SPEND_NOT_AUTHORIZED" in refused.stderr
    assert not out.exists()


def test_prereg_ack_check_precedes_python_or_any_output(tmp_path):
    out = tmp_path / "must-not-exist"
    env = os.environ.copy()
    # This is an isolated negative-policy test: the deliberately wrong ACK and
    # impossible interpreter make execution/model loading unreachable.
    env["RCC_PAID_SPEND_AUTHORIZED"] = "1"
    env["RCC_R1_PREREG_ACK"] = "wrong"
    env["PYTHON"] = "/definitely/missing/r1-python"
    refused = subprocess.run(
        [
            "bash", str(DRIVER), "--seeds", "0,1,2", "--device", "cuda",
            "--out", str(out),
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert refused.returncode == 78
    assert "PREREG_NOT_ACKNOWLEDGED" in refused.stderr
    assert not out.exists()


def test_plan_is_fixed_complete_and_hash_bound(tmp_path):
    namespace, plan = _build_plan(tmp_path)
    assert plan["schema_version"] == "rcc.r1_experiment_plan.v1"
    assert plan["status"] == "PLANNED_NOT_STARTED"
    assert plan["suite_hash"] == SUITE_HASH
    assert plan["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    assert plan["driver_source_sha256"] == DRIVER_SHA256
    assert plan["text_producer_source_sha256"] == TEXT_PRODUCER_SHA256
    assert plan["runtime_source_sha256"] == RUNTIME_SOURCE_SHA256
    assert plan["seeds"] == [0, 1, 2]
    assert plan["same_adapter_k_values"] == [0, 1, 2, 4, 8]
    assert plan["runtime_default_assumptions"] == {
        "noise_seed": 1234,
        "swap_policy": "fixed_next_example_in_suite_order",
        "binding": (
            "localized/latent-run source hashes; current CLI has no "
            "independent seed/swap-policy override"
        ),
    }
    assert plan["checkpoint_selection_split"] == "validation"
    assert plan["checkpoint_selection_uses_test"] is False
    assert plan["offline_model_loading_required"] is True
    assert plan["budget"] == {
        "max_wall_hours_per_seed": 8,
        "max_seeds": 3,
        "max_total_gpu_hours": 24,
        "max_rate_usd_per_hour": 0.45,
        "max_compute_cost_usd": 10.8,
        "hard_total_cost_cap_usd": 12.0,
        "rate_source": "preregistered_cap_not_live_price_claim",
        "provisioning_permitted": False,
        "enforcement_scope": (
            "cumulative local command wall; provider billing lifetime/rate "
            "requires an external authorized watchdog"
        ),
    }
    unhashed = copy.deepcopy(plan)
    digest = unhashed.pop("plan_hash")
    assert digest == hashlib.sha256(namespace["canonical_json"](unhashed)).hexdigest()


def test_training_and_eval_matrix_use_fixed_adapters_splits_and_controls(tmp_path):
    _namespace, plan = _build_plan(tmp_path)
    commands = plan["commands"]
    training = [command for command in commands if command["kind"] == "training"]
    assert len(training) == 9
    for seed in (0, 1, 2):
        seed_training = [command for command in training if command["seed"] == seed]
        assert {command["arm"] for command in seed_training} == {
            "train_mid_k4", "train_mid_k0", "train_full_k4",
        }
        expected = {
            "train_mid_k4": ("mid", "4"),
            "train_mid_k0": ("mid", "0"),
            "train_full_k4": ("full", "4"),
        }
        for command in seed_training:
            interval, trained_k = expected[command["arm"]]
            argv = command["argv"]
            assert _option(argv, "--interval") == interval
            assert _option(argv, "--k") == trained_k
            assert _option(argv, "--steps") == "800"
            assert _option(argv, "--eval-every") == "100"
            assert "--val-examples" not in argv
            assert len(command["evidence_paths"]) == 3

    evaluations = [
        command for command in commands if command["kind"] == "evaluation"
    ]
    fixed_arms = set(plan["fixed_arm_ids"])
    for seed in (0, 1, 2):
        for split in SPLITS:
            selected = [
                command for command in evaluations
                if command["seed"] == seed and command["split"] == split
            ]
            assert {command["arm"] for command in selected} == fixed_arms
            assert len(selected) == len(fixed_arms)

    same_adapter = [
        command for command in evaluations
        if command["seed"] == 0 and command["split"] == "validation"
        and command["arm"].startswith("same_adapter_k")
    ]
    assert {_option(command["argv"], "--k") for command in same_adapter} == {
        "0", "1", "2", "4", "8",
    }
    assert len({_option(command["argv"], "--adapter")
                for command in same_adapter}) == 1
    assert all("mid-k4" in _option(command["argv"], "--adapter")
               for command in same_adapter)

    latent_validation = {
        command["arm"]: command for command in evaluations
        if command["seed"] == 0 and command["split"] == "validation"
        and any("latent_run" in value for value in command["argv"])
    }
    assert "mid-k0-secondary" in _option(
        latent_validation["separately_trained_k0_secondary"]["argv"],
        "--adapter",
    )
    assert "full-k4" in _option(
        latent_validation["full_decoder_k4"]["argv"], "--adapter")
    expected_ablations = {
        "zero_state": "zero_state",
        "noise_state_seed_1234": "noise_state",
        "swap_state_fixed_next_example": "swap_state",
        "bypass_interval": "bypass_interval",
        "compute_matched_clock_permutation": "reverse_clocks",
    }
    for arm, ablation in expected_ablations.items():
        assert _option(latent_validation[arm]["argv"], "--ablate") == ablation

    text = [
        command for command in evaluations
        if any("text_baselines" in value for value in command["argv"])
    ]
    assert {_option(command["argv"], "--mode") for command in text} == {"A", "C"}
    assert all(_option(command["argv"], "--max-new-tokens") == "64"
               for command in text)


def test_every_artifact_is_validated_and_final_test_is_strictly_last(tmp_path):
    _namespace, plan = _build_plan(tmp_path)
    commands = plan["commands"]
    first_final = plan["final_test_policy"]["first_command_index"]
    assert all(command["split"] != "final_test"
               for command in commands[:first_final])
    assert all(command["split"] == "final_test"
               for command in commands[first_final:])
    assert all(command["phase"] != "training"
               for command in commands[first_final:])

    evaluations = [command for command in commands if command["kind"] == "evaluation"]
    for seed in (0, 1, 2):
        for arm in plan["fixed_arm_ids"]:
            assert sum(
                command["seed"] == seed and command["arm"] == arm
                and command["split"] == "final_test"
                for command in evaluations
            ) == 1

    validation_commands = [
        command for command in commands
        if command["kind"] == "offline_rescore"
    ]
    assert any("validate-run" in command["argv"]
               for command in validation_commands)
    assert any("validate-eval" in command["argv"]
               for command in validation_commands)
    assert all(
        "--require-r1-preregistered" in command["argv"]
        for command in validation_commands
        if any("text_baselines" in value for value in command["argv"])
    )
    for evaluation in evaluations:
        assert evaluation["evidence_paths"]
        later = commands[evaluation["index"] + 1:]
        assert any(
            command["seed"] == evaluation["seed"]
            and command["arm"] == evaluation["arm"]
            and command["split"] == evaluation["split"]
            and command["kind"] == "offline_rescore"
            for command in later
        )


@pytest.mark.parametrize(
    "override",
    [
        {"seeds": (0, 1)},
        {"device": "cpu"},
        {"max_wall_hours_per_seed": 7},
        {"max_total_gpu_hours": 23},
        {"max_rate_usd_per_hour": 0.46},
        {"max_compute_cost_usd": 10.81},
        {"hard_total_cost_cap_usd": 11.99},
        {"driver_source_sha256": "bad"},
        {"text_producer_source_sha256": "bad"},
        {"runtime_source_hashes": {"latent_run": "a" * 64}},
    ],
)
def test_plan_builder_rejects_preregistration_or_budget_drift(tmp_path, override):
    namespace = _driver_namespace()
    kwargs = {
        "seeds": (0, 1, 2), "device": "cuda",
        "out": tmp_path / "future-output", "suite_hash": SUITE_HASH,
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "python_bin": "/frozen/python",
        "driver_source_sha256": DRIVER_SHA256,
        "text_producer_source_sha256": TEXT_PRODUCER_SHA256,
        "runtime_source_hashes": RUNTIME_SOURCE_SHA256,
    }
    kwargs.update(override)
    with pytest.raises(ValueError):
        namespace["build_plan"](**kwargs)


def _exercise_budget(monkeypatch, tmp_path, budget):
    namespace = _driver_namespace()
    monotonic_values = iter((0.0, 10.0, 10.0, 30.0, 30.0, 35.0))
    wall_values = iter((100.0, 110.0, 120.0, 140.0, 150.0, 155.0))
    timeouts = []

    monkeypatch.setattr(namespace["time"], "monotonic",
                        lambda: next(monotonic_values))
    monkeypatch.setattr(namespace["time"], "time", lambda: next(wall_values))

    def fake_run(argv, *, check, timeout, env):
        del argv, check, env
        timeouts.append(timeout)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    commands = []
    for seed in (0, 1, 0):
        namespace["add_command"](
            commands, seed=seed, phase="mock", kind="offline_rescore",
            arm="mock", split="validation", argv=["true"],
        )
    plan = {
        "plan_hash": "a" * 64,
        "driver_source_sha256": "b" * 64,
        "text_producer_source_sha256": "c" * 64,
        "seeds": [0, 1],
        "commands": commands,
        "budget": budget,
    }
    out = tmp_path / "mock-execution"
    out.mkdir()
    summary = namespace["execute_plan"](plan, out)
    return timeouts, summary


def test_per_seed_budget_counts_only_active_commands(monkeypatch, tmp_path):
    timeouts, summary = _exercise_budget(monkeypatch, tmp_path, {
        "max_wall_hours_per_seed": 25 / 3600,
        "max_total_gpu_hours": 100 / 3600,
        "max_rate_usd_per_hour": 1.0,
        "max_compute_cost_usd": 100 / 3600,
        "hard_total_cost_cap_usd": 1.0,
    })
    assert timeouts == pytest.approx([25.0, 25.0, 15.0])
    assert summary["total_cumulative_command_wall_seconds"] == 35.0


@pytest.mark.parametrize("limiter", ["gpu_hours", "compute_cost"])
def test_global_gpu_hour_and_compute_cost_caps_use_cumulative_command_wall(
        monkeypatch, tmp_path, limiter):
    budget = {
        "max_wall_hours_per_seed": 100 / 3600,
        "max_total_gpu_hours": (40 if limiter == "gpu_hours" else 100) / 3600,
        "max_rate_usd_per_hour": 2.0,
        "max_compute_cost_usd": (
            200 / 3600 if limiter == "gpu_hours" else 80 / 3600),
        "hard_total_cost_cap_usd": 1.0,
    }
    timeouts, summary = _exercise_budget(monkeypatch, tmp_path, budget)
    assert timeouts == pytest.approx([40.0, 30.0, 10.0])
    assert summary["estimated_compute_cost_usd_at_cap_rate"] == pytest.approx(
        35 / 3600 * 2.0)


def test_plan_is_written_before_execution_and_driver_cannot_provision_or_download():
    source = DRIVER.read_text(encoding="utf-8")
    assert source.index('atomic_write_json(out / "execution_plan.json"') < source.index(
        "execution_summary = execute_plan(plan, out)")
    for forbidden in (
        "vastai", "runpod", "curl ", "wget ", "pip install", "git clone",
        "from_pretrained",
    ):
        assert forbidden not in source.lower()
    for offline_setting in (
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
    ):
        assert offline_setting in source
    assert "plan_hash" in source
    assert "argv_sha256" in source
    assert "hash_evidence_file" in source
    assert "RCC_R1_TEXT_PRODUCER_SOURCE_SHA256" in source
    for expected_flag in (
        "--expect-producer-command-index", "--expect-mode", "--expect-seed",
        "--expect-split", "--expect-run-id",
    ):
        assert expected_flag in source
