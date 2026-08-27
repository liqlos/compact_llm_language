#!/usr/bin/env bash
set -euo pipefail

SUITE_HASH="5cf5cbf397510ba597b59f7ccf0839cf344e6fb795a5cb29d031f39dac218254"
MODEL_ID="Qwen/Qwen3.5-2B"
MODEL_REVISION="15852e8c16360a2fea060d615a32b45270f8a8fc"
MAX_WALL_HOURS_PER_SEED="8"
MAX_TOTAL_GPU_HOURS="24"
MAX_RATE_USD_PER_HOUR="0.45"
MAX_COMPUTE_COST_USD="10.80"
HARD_TOTAL_COST_CAP_USD="12.00"

SEEDS=""
DEVICE=""
OUT=""

usage() {
  echo "usage: $0 --seeds 0,1,2 --device cuda --out PATH" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      [[ $# -ge 2 ]] || { usage; exit 64; }
      SEEDS="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || { usage; exit 64; }
      DEVICE="$2"
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || { usage; exit 64; }
      OUT="$2"
      shift 2
      ;;
    *)
      usage
      exit 64
      ;;
  esac
done

[[ "$SEEDS" == "0,1,2" ]] || {
  echo "refusing non-preregistered seeds: expected 0,1,2" >&2
  exit 65
}
[[ "$DEVICE" == "cuda" ]] || {
  echo "refusing non-preregistered device: expected cuda" >&2
  exit 65
}
[[ -n "$OUT" ]] || { usage; exit 64; }

if [[ "${RCC_PAID_SPEND_AUTHORIZED:-}" != "1" ]]; then
  echo "PAID_SPEND_NOT_AUTHORIZED: RCC_PAID_SPEND_AUTHORIZED must equal 1" >&2
  exit 78
fi
if [[ "${RCC_R1_PREREG_ACK:-}" != "$SUITE_HASH" ]]; then
  echo "PREREG_NOT_ACKNOWLEDGED: RCC_R1_PREREG_ACK must equal suite hash" >&2
  exit 78
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "python interpreter unavailable: $PYTHON_BIN" >&2
  exit 69
}
DRIVER_PATH="$SCRIPT_DIR/r1_experiment_driver.sh"
TEXT_PRODUCER_PATH="$SCRIPT_DIR/text_baselines.py"
LATENT_RUN_PATH="$SCRIPT_DIR/latent_run.py"
EVAL_V3_PATH="$SCRIPT_DIR/eval_v3.py"
ARTIFACT_VALIDATOR_PATH="$SCRIPT_DIR/artifacts.py"
LOCALIZED_RUNTIME_PATH="$SCRIPT_DIR/../backends/localized.py"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "SHA-256 utility unavailable" >&2
    return 69
  fi
}
DRIVER_SOURCE_SHA256="$(sha256_file "$DRIVER_PATH")" || exit $?
TEXT_PRODUCER_SOURCE_SHA256="$(sha256_file "$TEXT_PRODUCER_PATH")" || exit $?
LATENT_RUN_SOURCE_SHA256="$(sha256_file "$LATENT_RUN_PATH")" || exit $?
EVAL_V3_SOURCE_SHA256="$(sha256_file "$EVAL_V3_PATH")" || exit $?
ARTIFACT_VALIDATOR_SOURCE_SHA256="$(sha256_file "$ARTIFACT_VALIDATOR_PATH")" || exit $?
LOCALIZED_RUNTIME_SOURCE_SHA256="$(sha256_file "$LOCALIZED_RUNTIME_PATH")" || exit $?

# The authorized host must already contain the pinned model/tokenizer and
# frozen environment. These settings make an accidental weight fetch fail.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

export RCC_R1_DRIVER_SEEDS="$SEEDS"
export RCC_R1_DRIVER_DEVICE="$DEVICE"
export RCC_R1_DRIVER_OUT="$OUT"
export RCC_R1_DRIVER_SUITE_HASH="$SUITE_HASH"
export RCC_R1_DRIVER_MODEL_ID="$MODEL_ID"
export RCC_R1_DRIVER_MODEL_REVISION="$MODEL_REVISION"
export RCC_R1_DRIVER_MAX_WALL_HOURS_PER_SEED="$MAX_WALL_HOURS_PER_SEED"
export RCC_R1_DRIVER_MAX_TOTAL_GPU_HOURS="$MAX_TOTAL_GPU_HOURS"
export RCC_R1_DRIVER_MAX_RATE_USD_PER_HOUR="$MAX_RATE_USD_PER_HOUR"
export RCC_R1_DRIVER_MAX_COMPUTE_COST_USD="$MAX_COMPUTE_COST_USD"
export RCC_R1_DRIVER_HARD_TOTAL_COST_CAP_USD="$HARD_TOTAL_COST_CAP_USD"
export RCC_R1_DRIVER_SOURCE_SHA256="$DRIVER_SOURCE_SHA256"
export RCC_R1_TEXT_PRODUCER_SOURCE_SHA256="$TEXT_PRODUCER_SOURCE_SHA256"
export RCC_R1_LATENT_RUN_SOURCE_SHA256="$LATENT_RUN_SOURCE_SHA256"
export RCC_R1_EVAL_V3_SOURCE_SHA256="$EVAL_V3_SOURCE_SHA256"
export RCC_R1_ARTIFACT_VALIDATOR_SOURCE_SHA256="$ARTIFACT_VALIDATOR_SOURCE_SHA256"
export RCC_R1_LOCALIZED_RUNTIME_SOURCE_SHA256="$LOCALIZED_RUNTIME_SOURCE_SHA256"

"$PYTHON_BIN" - <<'PY'
# BEGIN_R1_EMBEDDED_DRIVER
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


LATENT_ARMS = (
    ("same_adapter_k0", "mid_k4", 0, None),
    ("same_adapter_k1", "mid_k4", 1, None),
    ("same_adapter_k2", "mid_k4", 2, None),
    ("same_adapter_k4", "mid_k4", 4, None),
    ("same_adapter_k8", "mid_k4", 8, None),
    ("separately_trained_k0_secondary", "mid_k0", 0, None),
    ("full_decoder_k4", "full_k4", 4, None),
    ("zero_state", "mid_k4", 4, "zero_state"),
    ("noise_state_seed_1234", "mid_k4", 4, "noise_state"),
    ("swap_state_fixed_next_example", "mid_k4", 4, "swap_state"),
    ("bypass_interval", "mid_k4", 4, "bypass_interval"),
    ("compute_matched_clock_permutation", "mid_k4", 4, "reverse_clocks"),
)
TEXT_ARMS = (
    ("direct_no_thinking_textual", "A"),
    ("visible_scratch_max_64", "C"),
)
EVAL_SPLITS = (
    "validation",
    "test_id",
    "test_ood_length",
    "test_ood_semantic",
    "final_test",
)


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def add_command(
        commands, *, seed, phase, kind, arm, split, argv,
        evidence_paths=()):
    command = {
        "index": len(commands),
        "seed": seed,
        "phase": phase,
        "kind": kind,
        "arm": arm,
        "split": split,
        "accesses_final_test": split == "final_test" and kind == "evaluation",
        "argv": [str(value) for value in argv],
        "evidence_paths": [str(Path(value).resolve()) for value in evidence_paths],
    }
    commands.append(command)
    return command


def train_command(python_bin, *, seed, device, model_id, revision, out,
                  interval, trained_k, label):
    return [
        python_bin, "-m", "latent_lab.bench.latent_run", "train",
        "--k", str(trained_k), "--interval", interval,
        "--steps", "800", "--lr", "0.0001", "--warmup", "50",
        "--optimizer", "adamw", "--weight-decay", "0.01",
        "--clip", "0.5", "--lr-schedule", "constant",
        "--lora-r", "8", "--lora-alpha", "16.0", "--max-k", "8",
        "--seed", str(seed), "--eval-every", "100", "--device", device,
        "--model", model_id, "--revision", revision, "--label", label,
        "--out", str(out),
    ]


def build_plan(*, seeds, device, out, suite_hash, model_id, revision,
               python_bin, driver_source_sha256, text_producer_source_sha256,
               runtime_source_hashes,
               max_wall_hours_per_seed=8,
               max_total_gpu_hours=24, max_rate_usd_per_hour=0.45,
               max_compute_cost_usd=10.80, hard_total_cost_cap_usd=12.00):
    seeds = tuple(seeds)
    if seeds != (0, 1, 2):
        raise ValueError("preregistered seeds must be exactly (0, 1, 2)")
    if device != "cuda":
        raise ValueError("preregistered device must be cuda")
    if max_wall_hours_per_seed != 8 or max_total_gpu_hours != 24:
        raise ValueError("preregistered wall/GPU-hour caps changed")
    if not math.isclose(max_rate_usd_per_hour, 0.45):
        raise ValueError("preregistered hourly rate cap changed")
    if not math.isclose(max_compute_cost_usd, 10.80):
        raise ValueError("preregistered compute-cost cap changed")
    if not math.isclose(hard_total_cost_cap_usd, 12.00):
        raise ValueError("preregistered hard total cost cap changed")
    if not math.isclose(
            max_total_gpu_hours,
            max_wall_hours_per_seed * len(seeds)):
        raise ValueError("GPU-hour cap must cover exactly the per-seed caps")
    if max_compute_cost_usd > max_total_gpu_hours * max_rate_usd_per_hour:
        raise ValueError("compute-cost cap exceeds rate-times-hours cap")
    if hard_total_cost_cap_usd < max_compute_cost_usd:
        raise ValueError("hard total cost cap is below compute-cost cap")
    if (not isinstance(driver_source_sha256, str)
            or len(driver_source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in driver_source_sha256)):
        raise ValueError("driver source SHA-256 is invalid")
    if (not isinstance(text_producer_source_sha256, str)
            or len(text_producer_source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in text_producer_source_sha256)):
        raise ValueError("text producer source SHA-256 is invalid")
    expected_runtime_sources = {
        "artifact_validator", "eval_v3", "latent_run", "localized_runtime",
    }
    if (not isinstance(runtime_source_hashes, dict)
            or set(runtime_source_hashes) != expected_runtime_sources
            or any(
                not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)
                for value in runtime_source_hashes.values()
            )):
        raise ValueError("runtime source SHA-256 manifest is invalid")
    out = Path(out).resolve()
    commands = []
    fixed_arm_ids = [arm for arm, _mode in TEXT_ARMS]
    fixed_arm_ids.extend(arm for arm, _adapter, _k, _ablation in LATENT_ARMS)

    adapters_by_seed = {}
    for seed in seeds:
        seed_root = out / f"seed-{seed}"
        adapters_by_seed[seed] = {
            "mid_k4": seed_root / "train" / "mid-k4",
            "mid_k0": seed_root / "train" / "mid-k0-secondary",
            "full_k4": seed_root / "train" / "full-k4",
        }
        adapters = adapters_by_seed[seed]
        for adapter_key, interval, trained_k in (
            ("mid_k4", "mid", 4),
            ("mid_k0", "mid", 0),
            ("full_k4", "full", 4),
        ):
            arm = f"train_{adapter_key}"
            adapter_path = adapters[adapter_key]
            label = f"r1-{adapter_key}-seed-{seed}"
            add_command(
                commands, seed=seed, phase="training", kind="training",
                arm=arm, split="validation",
                argv=train_command(
                    python_bin, seed=seed, device=device, model_id=model_id,
                    revision=revision, out=adapters[adapter_key],
                    interval=interval, trained_k=trained_k,
                    label=label,
                ),
                evidence_paths=(
                    adapter_path / "best_params.pt",
                    adapter_path / "train_report.json",
                    adapter_path / "run_manifest.json",
                ),
            )
            add_command(
                commands, seed=seed, phase="training_artifact_validation",
                kind="offline_rescore", arm=arm, split="validation",
                argv=[
                    python_bin, "-m", "latent_lab.bench.artifacts",
                    "validate-run", str(adapter_path),
                    "--expect-model", model_id,
                    "--expect-rev", revision,
                    "--expect-suite", suite_hash,
                    "--expect-seed", str(seed),
                    "--expect-label", label,
                    "--expect-k", str(trained_k),
                    "--expect-steps", "800",
                ],
            )

    # Every checkpoint is selected on validation. All fixed, non-final
    # evaluations finish before the untouched final_test is opened.
    for split in EVAL_SPLITS:
        for seed in seeds:
            seed_root = out / f"seed-{seed}"
            adapters = adapters_by_seed[seed]
            phase = (
                "validation_evaluation" if split == "validation"
                else "final_evaluation" if split == "final_test"
                else "fixed_heldout_evaluation"
            )
            eval_root = seed_root / "eval" / split
            for arm, mode in TEXT_ARMS:
                evidence_path = eval_root / f"{arm}.json"
                run_id = f"r1-{arm}-seed-{seed}-{split}"
                producer_command = add_command(
                    commands, seed=seed, phase=phase, kind="evaluation",
                    arm=arm, split=split,
                    argv=[
                        python_bin, "-m", "latent_lab.bench.text_baselines",
                        "--mode", mode, "--split", split,
                        "--max-new-tokens", "64", "--batch", "1",
                        "--seed", str(seed), "--device", device,
                        "--model", model_id, "--revision", revision,
                        "--run-id", run_id,
                        "--out", str(evidence_path),
                    ],
                    evidence_paths=(evidence_path,),
                )
                rescore_path = eval_root / f"{arm}.rescore.json"
                add_command(
                    commands, seed=seed, phase=phase, kind="offline_rescore",
                    arm=arm, split=split,
                    argv=[
                        python_bin, "-m", "latent_lab.bench.text_baselines",
                        "--rescore", str(evidence_path),
                        "--require-r1-preregistered",
                        "--expect-producer-command-index",
                        str(producer_command["index"]),
                        "--expect-mode", mode,
                        "--expect-seed", str(seed),
                        "--expect-split", split,
                        "--expect-run-id", run_id,
                        "--out", str(rescore_path),
                    ],
                    evidence_paths=(rescore_path,),
                )

            for arm, adapter_key, k_value, ablation in LATENT_ARMS:
                evidence_path = eval_root / f"{arm}.json"
                argv = [
                    python_bin, "-m", "latent_lab.bench.latent_run", "eval",
                    "--adapter", str(adapters[adapter_key]),
                    "--split", split, "--k", str(k_value),
                    "--seed", str(seed), "--device", device,
                    "--out", str(evidence_path),
                ]
                if ablation is not None:
                    argv.extend(["--ablate", ablation])
                add_command(
                    commands, seed=seed, phase=phase, kind="evaluation",
                    arm=arm, split=split, argv=argv,
                    evidence_paths=(evidence_path,),
                )
                add_command(
                    commands, seed=seed, phase=phase, kind="offline_rescore",
                    arm=arm, split=split,
                    argv=[
                        python_bin, "-m", "latent_lab.bench.artifacts",
                        "validate-eval", str(evidence_path),
                        "--expect-model", model_id,
                        "--expect-rev", revision,
                        "--expect-suite", suite_hash,
                        "--expect-seed", str(seed),
                        "--expect-k", str(k_value),
                        "--expect-split", split,
                        "--expect-ablation", ablation or "clean",
                    ],
                )

    first_final = next(
        command["index"] for command in commands
        if command["accesses_final_test"]
    )
    if any(command["phase"] == "training" for command in commands[first_final:]):
        raise RuntimeError("training appears after final_test was opened")
    if any(command["split"] != "final_test"
           for command in commands[first_final:]):
        raise RuntimeError("a non-final command appears after final_test was opened")
    for seed in seeds:
        for arm in fixed_arm_ids:
            count = sum(
                command["seed"] == seed
                and command["kind"] == "evaluation"
                and command["arm"] == arm
                and command["split"] == "final_test"
                for command in commands
            )
            if count != 1:
                raise RuntimeError(
                    f"final_test must occur exactly once for {seed=} {arm=}: {count}"
                )

    plan = {
        "schema_version": "rcc.r1_experiment_plan.v1",
        "status": "PLANNED_NOT_STARTED",
        "suite_identity": "behavioral-v3",
        "suite_hash": suite_hash,
        "model": {"id": model_id, "revision": revision},
        "driver_source_sha256": driver_source_sha256,
        "text_producer_source_sha256": text_producer_source_sha256,
        "runtime_source_sha256": dict(sorted(runtime_source_hashes.items())),
        "seeds": list(seeds),
        "device": device,
        "paid_spend_authorization_verified_by_wrapper": True,
        "offline_model_loading_required": True,
        "checkpoint_selection_split": "validation",
        "checkpoint_selection_uses_test": False,
        "fixed_arm_ids": fixed_arm_ids,
        "same_adapter_k_values": [0, 1, 2, 4, 8],
        "runtime_default_assumptions": {
            "noise_seed": 1234,
            "swap_policy": "fixed_next_example_in_suite_order",
            "binding": (
                "localized/latent-run source hashes; current CLI has no "
                "independent seed/swap-policy override"
            ),
        },
        "clock_permutation": {
            "runtime_ablation": "reverse_clocks",
            "claim": "recurrence-compute-matched clock-order control",
            "wall_clock_equality_claimed": False,
        },
        "final_test_policy": {
            "access": "once_per_fixed_arm_per_seed",
            "first_command_index": first_final,
            "result_dependent_arm_selection": False,
        },
        "budget": {
            "max_wall_hours_per_seed": max_wall_hours_per_seed,
            "max_seeds": len(seeds),
            "max_total_gpu_hours": max_total_gpu_hours,
            "max_rate_usd_per_hour": max_rate_usd_per_hour,
            "max_compute_cost_usd": max_compute_cost_usd,
            "hard_total_cost_cap_usd": hard_total_cost_cap_usd,
            "rate_source": "preregistered_cap_not_live_price_claim",
            "provisioning_permitted": False,
            "enforcement_scope": (
                "cumulative local command wall; provider billing lifetime/rate "
                "requires an external authorized watchdog"
            ),
        },
        "commands": commands,
    }
    plan["plan_hash"] = hashlib.sha256(canonical_json(plan)).hexdigest()
    return plan


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json(payload) + b"\n")
    os.replace(temporary, path)


def hash_evidence_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"expected evidence file is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def execute_plan(plan, out):
    receipts_path = out / "command_receipts.jsonl"
    seed_used_seconds = {seed: 0.0 for seed in plan["seeds"]}
    cap_seconds = float(plan["budget"]["max_wall_hours_per_seed"]) * 3600
    total_cap_seconds = float(plan["budget"]["max_total_gpu_hours"]) * 3600
    hourly_rate = float(plan["budget"]["max_rate_usd_per_hour"])
    compute_cost_cap = float(plan["budget"]["max_compute_cost_usd"])
    hard_total_cap = float(plan["budget"]["hard_total_cost_cap_usd"])
    cost_limited_seconds = min(compute_cost_cap, hard_total_cap) / hourly_rate * 3600
    total_used_seconds = 0.0
    evidence_files = {}
    receipt_count = 0
    for command in plan["commands"]:
        seed = command["seed"]
        remaining = min(
            cap_seconds - seed_used_seconds[seed],
            total_cap_seconds - total_used_seconds,
            cost_limited_seconds - total_used_seconds,
        )
        if remaining <= 0:
            raise TimeoutError(
                f"seed/global experiment budget exhausted before command "
                f"{command['index']}"
            )
        monotonic_started = time.monotonic()
        started = time.time()
        receipt = {
            "schema_version": "rcc.r1_command_receipt.v1",
            "command_index": command["index"],
            "seed": seed,
            "arm": command["arm"],
            "split": command["split"],
            "started_unix_seconds": started,
            "plan_hash": plan["plan_hash"],
            "argv_sha256": hashlib.sha256(
                canonical_json(command["argv"])
            ).hexdigest(),
        }
        try:
            child_env = os.environ.copy()
            child_env["RCC_R1_ACTIVE_PLAN_HASH"] = plan["plan_hash"]
            child_env["RCC_R1_ACTIVE_COMMAND_INDEX"] = str(command["index"])
            child_env["RCC_R1_DRIVER_SOURCE_SHA256"] = plan[
                "driver_source_sha256"]
            child_env["RCC_R1_TEXT_PRODUCER_SOURCE_SHA256"] = plan[
                "text_producer_source_sha256"]
            completed = subprocess.run(
                command["argv"], check=False, timeout=remaining,
                env=child_env,
            )
            receipt["returncode"] = completed.returncode
            receipt["wall_seconds"] = time.time() - started
            receipt["status"] = "PASS" if completed.returncode == 0 else "FAIL"
            if receipt["status"] == "PASS":
                try:
                    command_evidence = [
                        hash_evidence_file(path)
                        for path in command["evidence_paths"]
                    ]
                except (OSError, ValueError) as error:
                    receipt["status"] = "MISSING_OR_UNREADABLE_EVIDENCE"
                    receipt["evidence_error"] = str(error)
                    command_evidence = []
                receipt["evidence_files"] = command_evidence
                for item in command_evidence:
                    item["producer_command_index"] = command["index"]
                    item["plan_hash"] = plan["plan_hash"]
                    item["driver_source_sha256"] = plan[
                        "driver_source_sha256"]
                    evidence_files[item["path"]] = item
        except subprocess.TimeoutExpired:
            receipt["returncode"] = None
            receipt["wall_seconds"] = time.time() - started
            receipt["status"] = "WALL_CAP_EXCEEDED"
        command_wall_seconds = time.monotonic() - monotonic_started
        seed_used_seconds[seed] += command_wall_seconds
        total_used_seconds += command_wall_seconds
        receipt["seed_cumulative_command_wall_seconds"] = seed_used_seconds[seed]
        receipt["total_cumulative_command_wall_seconds"] = total_used_seconds
        receipt["estimated_compute_cost_usd_at_cap_rate"] = (
            total_used_seconds / 3600 * hourly_rate
        )
        with receipts_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(receipt).decode("utf-8") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt_count += 1
        if receipt["status"] != "PASS":
            raise RuntimeError(
                f"command {command['index']} failed closed: {receipt['status']}"
            )
    receipts = hash_evidence_file(receipts_path)
    return {
        "receipt_count": receipt_count,
        "receipts": receipts,
        "evidence_files": [evidence_files[path]
                           for path in sorted(evidence_files)],
        "total_cumulative_command_wall_seconds": total_used_seconds,
        "estimated_compute_cost_usd_at_cap_rate": (
            total_used_seconds / 3600 * hourly_rate
        ),
    }


def main():
    seeds = tuple(int(value) for value in os.environ["RCC_R1_DRIVER_SEEDS"].split(","))
    out = Path(os.environ["RCC_R1_DRIVER_OUT"]).resolve()
    if out.exists():
        raise RuntimeError(f"fresh output root required; already exists: {out}")
    plan = build_plan(
        seeds=seeds,
        device=os.environ["RCC_R1_DRIVER_DEVICE"],
        out=out,
        suite_hash=os.environ["RCC_R1_DRIVER_SUITE_HASH"],
        model_id=os.environ["RCC_R1_DRIVER_MODEL_ID"],
        revision=os.environ["RCC_R1_DRIVER_MODEL_REVISION"],
        python_bin=sys.executable,
        driver_source_sha256=os.environ["RCC_R1_DRIVER_SOURCE_SHA256"],
        text_producer_source_sha256=os.environ[
            "RCC_R1_TEXT_PRODUCER_SOURCE_SHA256"],
        runtime_source_hashes={
            "latent_run": os.environ["RCC_R1_LATENT_RUN_SOURCE_SHA256"],
            "eval_v3": os.environ["RCC_R1_EVAL_V3_SOURCE_SHA256"],
            "artifact_validator": os.environ[
                "RCC_R1_ARTIFACT_VALIDATOR_SOURCE_SHA256"],
            "localized_runtime": os.environ[
                "RCC_R1_LOCALIZED_RUNTIME_SOURCE_SHA256"],
        },
        max_wall_hours_per_seed=int(
            os.environ["RCC_R1_DRIVER_MAX_WALL_HOURS_PER_SEED"]),
        max_total_gpu_hours=int(
            os.environ["RCC_R1_DRIVER_MAX_TOTAL_GPU_HOURS"]),
        max_rate_usd_per_hour=float(
            os.environ["RCC_R1_DRIVER_MAX_RATE_USD_PER_HOUR"]),
        max_compute_cost_usd=float(
            os.environ["RCC_R1_DRIVER_MAX_COMPUTE_COST_USD"]),
        hard_total_cost_cap_usd=float(
            os.environ["RCC_R1_DRIVER_HARD_TOTAL_COST_CAP_USD"]),
    )
    out.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out / "execution_plan.json", plan)
    execution_summary = execute_plan(plan, out)
    atomic_write_json(out / "execution_complete.json", {
        "schema_version": "rcc.r1_execution_complete.v1",
        "plan_hash": plan["plan_hash"],
        "status": "COMPLETE_REQUIRES_INDEPENDENT_EVIDENCE_REVIEW",
        "execution_summary": execution_summary,
    })


if __name__ == "__main__":
    main()
# END_R1_EMBEDDED_DRIVER
PY
