#!/usr/bin/env bash
# Sealed PREREGISTERED paid canary — Qwen/Qwen3.5-4B.
#
# Contract (fail closed):
#   * Focused unit/integrity tests MUST pass BEFORE any GPU/model spend;
#     any failure aborts the driver.
#   * EXACT dependency identity (equality against pinned versions,
#     including Python, torch+cu126 variant, torch.version.cuda,
#     transformers, huggingface_hub AND the project uv.lock content
#     hash): no ranges, NO generic installs/upgrades ("pip -U" and
#     "uv sync" are banned). The ONLY network bootstrap is the exact
#     sha256-pinned uv wheel plus the sha256-pinned frozen lock export
#     (--require-hashes) into a FRESH system-site venv; torch/CUDA stay
#     owned by the verified base image. NO ignored failures
#     (set -euo pipefail).
#   * ONE preregistered seed/recipe fixed BELOW, before any GPU second:
#     nothing in this driver ever inspects validation results to choose
#     a seed, an adapter or a recipe (no BEST/MEDIAN cherry-picking).
#   * Paired causal arm: K>0 and K=0 are evaluated on the SAME trained
#     adapter/seed/suite (only the eval-time K differs). Separate F
#     adapters do NOT satisfy this contract and are not used.
#   * Bounded by construction: exactly ONE training run + paired K=4/K=0
#     evaluations on each preregistered behavioral-v3 evidence split. Any
#     matrix expansion requires its own separate
#     pre-spend authorization and is refused here.
#   * Resume NEVER trusts bare existence: a directory/payload counts as
#     done ONLY when it re-validates under the FULL expected contract;
#     anything else is quarantined (*.invalid.<ts>) and recomputed.
#   * SEALED LAUNCH ENVIRONMENT: no launch happens unless an explicitly
#     immutable image reference (repository@sha256:<64-hex>) and an exact
#     sealed-environment contract are supplied from outside; provisioning
#     and this driver are bound to the SAME digest/contract. There is NO
#     default mutable image: unsealed inputs abort BEFORE any work.
set -euo pipefail
cd /root/rcc

if [ -n "${DRIVER_MATRIX:-}" ]; then
  echo "FATAL: DRIVER_MATRIX is set; matrix expansion is NOT part of this" \
       "canary and requires separate pre-spend authorization." >&2
  exit 5
fi

### --- preregistered experiment constants (fixed BEFORE any GPU work) ---
MODEL=Qwen/Qwen3.5-4B
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
SEED=0
K_POS=4
K_ZERO=0
INTERVAL=mid
STEPS=800
LR=1e-4
WARMUP=50
CLIP=0.5
OPTIMIZER=adamw
WEIGHT_DECAY=0.01
LR_SCHEDULE=constant
MAX_K=16
LORA_R=8
LORA_ALPHA=16.0
DETACH_Z0=false
LABEL=E4_k${K_POS}_s${SEED}
RUN_DIR=runs/${LABEL}

# Shell 'true'/'false' MUST become real Python booleans before they can
# enter any Python heredoc (a bare lowercase false/true is a NameError
# that would abort AFTER paid setup). ONE validated source feeds BOTH
# the recipe digest and the trainer flag binding, so they can never
# disagree.
py_bool () {
  case "$1" in
    true) printf 'True' ;;
    false) printf 'False' ;;
    *) echo "FATAL: expected shell boolean exactly 'true' or 'false'; got '$1'" >&2
       exit 5 ;;
  esac
}
PY_DETACH_Z0=$(py_bool "$DETACH_Z0")
# THE verified immutable base image (Python 3.12.3, torch
# 2.13.0+cu126/CUDA 12.6, transformers/huggingface_hub absent): the
# externally supplied SEALED_IMAGE must equal this digest exactly.
PIN_IMAGE="pytorch/pytorch@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
PIN_PYTHON="3.12.3"
PIN_TORCH="2.13.0+cu126"
PIN_TORCH_CUDA="12.6"
PIN_TRANSFORMERS="5.15.1"
PIN_HUGGINGFACE_HUB="1.28.0"
PIN_UVLOCK_SHA256="62187a854931549a8cd927537a3cf393759fd56b79152c5f400447b9c3de035f"
# Hash-pinned bootstrap artifacts: exact uv wheel + the frozen lock
# export bytes it must produce. NOTHING else is ever fetched/installed.
PIN_UV_VERSION="0.11.28"
PIN_UV_WHEEL_URL="https://files.pythonhosted.org/packages/75/2e/62273ee6c9fbebccd8248c153b44870f81ebf5267c31edf4c095d78537fb/uv-0.11.28-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
PIN_UV_WHEEL_SHA256="49fe42df9f42056037473f3876adec1615709b57d3470ed39178ff420f3afb9f"
PIN_REQUIREMENTS_SHA256="15d25f1ca4eec6bf8ea59c4a224a8d3268897963d958f93516a4262526d947fd"

### --- sealed launch environment gate (aborts BEFORE tests/training) ---
: "${SEALED_IMAGE:?FATAL: SEALED_IMAGE must be preregistered as an immutable repository@sha256:<64-hex> reference; no default exists}"
: "${SEALED_ENV_CONTRACT:?FATAL: SEALED_ENV_CONTRACT must point at the preregistered sealed-environment contract JSON; no default exists}"
[ "$SEALED_IMAGE" = "$PIN_IMAGE" ] \
  || { echo "FATAL: SEALED_IMAGE $SEALED_IMAGE != preregistered verified image $PIN_IMAGE" >&2
       exit 5; }
python -m latent_lab.bench.sealed_env require-image --image "$SEALED_IMAGE"
python -m latent_lab.bench.sealed_env verify-contract \
  --contract "$SEALED_ENV_CONTRACT" --image "$SEALED_IMAGE"
# cross-bind the contract to THESE preregistered pins so driver and
# provisioner can never drift onto different environments
python -m latent_lab.bench.sealed_env check-pins \
  --contract "$SEALED_ENV_CONTRACT" \
  --pin "image=$PIN_IMAGE" \
  --pin "python=$PIN_PYTHON" \
  --pin "torch=$PIN_TORCH" \
  --pin "transformers=$PIN_TRANSFORMERS" \
  --pin "huggingface_hub=$PIN_HUGGINGFACE_HUB" \
  --pin "uvlock_sha256=$PIN_UVLOCK_SHA256"

mkdir -p runs results
exec 9>results/.driver4b.lock
if ! flock -n 9; then
  echo "FATAL: another driver holds results/.driver4b.lock" >&2
  exit 3
fi

echo "=== hash-pinned bootstrap: exact uv wheel -> frozen export -> fresh venv ==="

# Lock identity is re-proved BEFORE anything is fetched or exported.
ACTUAL_LOCK_SHA=$(sha256sum uv.lock | cut -d ' ' -f1)
if [ "$ACTUAL_LOCK_SHA" != "$PIN_UVLOCK_SHA256" ]; then
  echo "FATAL: project lock drift: uv.lock sha256 $ACTUAL_LOCK_SHA != pinned $PIN_UVLOCK_SHA256" >&2
  exit 2
fi

BOOTSTRAP_DIR=.bootstrap4b
mkdir -p "$BOOTSTRAP_DIR"
EXPECTED_WHEEL_NAME="uv-${PIN_UV_VERSION}-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
[ "${PIN_UV_WHEEL_URL##*/}" = "$EXPECTED_WHEEL_NAME" ] \
  || { echo "FATAL: uv wheel URL does not name the exact pinned artifact $EXPECTED_WHEEL_NAME" >&2
       exit 2; }
UV_WHEEL="$BOOTSTRAP_DIR/$EXPECTED_WHEEL_NAME"
UV_WHEEL_URL="$PIN_UV_WHEEL_URL" UV_WHEEL_OUT="$UV_WHEEL" python - <<'PY'
import os
import shutil
import time
import urllib.request

url = os.environ["UV_WHEEL_URL"]
out = os.environ["UV_WHEEL_OUT"]
partial = out + ".partial"
for attempt in range(3):
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                open(partial, "wb") as target:
            shutil.copyfileobj(response, target)
        os.replace(partial, out)
        break
    except Exception:
        try:
            os.unlink(partial)
        except FileNotFoundError:
            pass
        if attempt == 2:
            raise
        time.sleep(1)
PY
WHEEL_SHA=$(sha256sum "$UV_WHEEL" | cut -d ' ' -f1)
if [ "$WHEEL_SHA" != "$PIN_UV_WHEEL_SHA256" ]; then
  echo "FATAL: downloaded uv wheel sha256 $WHEEL_SHA != pinned $PIN_UV_WHEEL_SHA256" >&2
  exit 2
fi

# Debian system interpreter receives ONLY this exact digest-verified
# wheel (--break-system-packages); every lab package lands in the fresh
# sealed venv below, never in the system environment.
python -m pip install --break-system-packages --no-deps "$UV_WHEEL"
UV_BIN=$(command -v uv) \
  || { echo "FATAL: uv executable not found on PATH after wheel install" >&2
       exit 2; }
export PATH="$(dirname "$UV_BIN"):$PATH"
UV_REPORTED=$(uv --version)
case "$UV_REPORTED" in
  "uv $PIN_UV_VERSION "*) : ;;
  *) echo "FATAL: uv version drift: got '$UV_REPORTED', need uv $PIN_UV_VERSION.*" >&2
     exit 2 ;;
esac

# Frozen export of EXACTLY the preregistered command; its bytes are
# digest-pinned below. uv records this invocation in the file header,
# so even an argv drift would break the required sha256.
REQ="$BOOTSTRAP_DIR/requirements.sealed.txt"
uv export --frozen --group lab --group dev --no-emit-project --prune torch > "$REQ"
REQ_SHA=$(sha256sum "$REQ" | cut -d ' ' -f1)
if [ "$REQ_SHA" != "$PIN_REQUIREMENTS_SHA256" ]; then
  echo "FATAL: exported requirements sha256 $REQ_SHA != pinned $PIN_REQUIREMENTS_SHA256" >&2
  exit 2
fi
# Defense in depth: no actual requirement may name torch/triton/cuda*/
# nvidia-* (comments do not count); torch+CUDA stay owned by the image.
FORBIDDEN_RE='^(torch|triton|nvidia[-_a-z0-9.]*|cuda[-_a-z0-9.]*)[=\[@ ;]'
if grep -Ev '^[[:space:]]*(#|$)' "$REQ" | grep -Eiq "$FORBIDDEN_RE"; then
  echo "FATAL: exported requirements name a torch/triton/cuda/nvidia package;" \
       "the verified base image owns torch and CUDA" >&2
  exit 2
fi

# Fresh DETERMINISTIC venv bound to THIS image interpreter (explicit
# interpreter + --no-managed-python forbid any uv-managed download);
# --system-site-packages keeps the image torch/CUDA stack authoritative.
SEALED_VENV="$PWD/.venv-sealed-4b"
rm -rf "$SEALED_VENV"
PYBIN=$(command -v python3)
SYSTEM_PY_VER=$("$PYBIN" -c 'import platform; print(platform.python_version())')
if [ "$SYSTEM_PY_VER" != "$PIN_PYTHON" ]; then
  echo "FATAL: image interpreter $PYBIN reports Python $SYSTEM_PY_VER != pinned $PIN_PYTHON" >&2
  exit 2
fi
uv venv --system-site-packages --no-managed-python --python "$PYBIN" "$SEALED_VENV"
uv pip install --python "$SEALED_VENV/bin/python" --no-deps --require-hashes -r "$REQ"

# From here to the end of the driver, ONLY the venv python is used.
export PATH="$SEALED_VENV/bin:$PATH"
[ "$(command -v python)" = "$SEALED_VENV/bin/python" ] \
  || { echo "FATAL: python does not resolve into the sealed venv" >&2
       exit 2; }

echo "=== sealed live-environment verification AFTER bootstrap ==="
python -m latent_lab.bench.sealed_env verify-live \
  --contract "$SEALED_ENV_CONTRACT" --lockfile uv.lock --require-cuda

echo "=== sealed environment identity (EXACT pins incl. provenance) ==="
SEALED_VENV_DIR="$SEALED_VENV" python - <<PY
import glob
import os
import sys
import huggingface_hub
import torch
import transformers

def require(actual, pinned, what):
    assert actual == pinned, f"{what} {actual!r} != pinned {pinned!r}"

require(sys.version.split()[0], "${PIN_PYTHON}", "python")
require(torch.__version__, "${PIN_TORCH}", "torch")
require(str(torch.version.cuda), "${PIN_TORCH_CUDA}", "torch.version.cuda")
require(transformers.__version__, "${PIN_TRANSFORMERS}", "transformers")
require(huggingface_hub.__version__, "${PIN_HUGGINGFACE_HUB}",
        "huggingface_hub")
# provenance: torch must come from the IMAGE system site-packages, never
# from anything installed into the fresh sealed venv
venv_root = os.path.realpath(os.environ["SEALED_VENV_DIR"])
assert not os.path.realpath(str(torch.__file__)).startswith(
    venv_root + os.sep), (
    "torch resolved INSIDE the sealed venv; it must come from the image "
    "system site-packages")
own_sites = [p for p in sys.path
             if str(p).startswith(venv_root + os.sep)]
invading = [os.path.basename(hit) for p in own_sites
            for hit in glob.glob(os.path.join(p, "*"))
            if os.path.basename(hit).lower().startswith("torch")]
assert not invading, f"torch artifacts installed into the venv: {invading}"
assert torch.cuda.is_available(), "CUDA unavailable"
print("ENV_OK", sys.version.split()[0], torch.__version__,
      transformers.__version__, huggingface_hub.__version__,
      torch.cuda.get_device_name(0))
PY

echo "=== pre-spend gate: focused checks MUST pass before ANY GPU work ==="
python -m pytest -q \
  tests/test_latent_runtime_integrity.py \
  tests/test_latent_run.py \
  tests/test_artifact_contracts.py \
  tests/test_paid_driver_sealed.py

SUITE_SHA=$(python -c "from latent_lab.bench.suite_v3 import build_suite; print(build_suite().manifest()['suite_hash'])")

# Preregister the EXACT canonical recipe digest from the SAME constants
# the trainer will use (shared helper — no hand-maintained field list).
# Resume validation binds this digest, so a wrong suite/LR/interval/LoRA/
# optimizer/schedule/warmup/clip/detach artifact can never be
# resumed or reused.
N_LAYERS=$(python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained(\"$MODEL\", revision=\"$REV\").num_hidden_layers)")
CONFIG_SHA256=$(python - <<PY
from latent_lab.bench.latent_run import (
    interval_from_spec, mode_from_spec, train_recipe_digest)
iv = interval_from_spec("${INTERVAL}", ${N_LAYERS})
print(train_recipe_digest(
    mode=mode_from_spec("${INTERVAL}", ${K_POS}),
    interval=list(iv), k=${K_POS}, max_k=${MAX_K},
    lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA},
    lr=${LR}, steps=${STEPS}, seed=${SEED},
    optimizer="${OPTIMIZER}", weight_decay=${WEIGHT_DECAY},
    lr_schedule="${LR_SCHEDULE}", warmup=${WARMUP},
    clip=${CLIP}, detach_z0=${PY_DETACH_Z0},
    suite_sha256="${SUITE_SHA}"))
PY
)
echo "PREREGISTERED_RECIPE config_sha256=$CONFIG_SHA256 suite=$SUITE_SHA"

MODEL_IDENTITY_MODEL="$MODEL" MODEL_IDENTITY_REV="$REV" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info(os.environ["MODEL_IDENTITY_MODEL"])
commit = info.sha
expected = os.environ["MODEL_IDENTITY_REV"]
assert commit == expected, f"hub resolved {commit!r}, expected pinned {expected}"
print("MODEL_IDENTITY_OK", commit)
PY

COMMON=(--model "$MODEL" --revision "$REV")

quarantine () {
  local P=$1
  if [ -e "$P" ]; then
    mv "$P" "$P.invalid.$(date +%s)" && echo "QUARANTINED $P"
  fi
}

expect_run=(--expect-model "$MODEL" --expect-rev "$REV"
            --expect-suite "$SUITE_SHA"
            --expect-seed "$SEED" --expect-label "$LABEL"
            --expect-k "$K_POS" --expect-steps "$STEPS"
            --expect-config-sha256 "$CONFIG_SHA256")
valid_run () {
  python3 -m latent_lab.bench.artifacts validate-run "$1" "${expect_run[@]}"
}

echo "=== single preregistered training run: label=$LABEL seed=$SEED k=$K_POS ==="
if valid_run "$RUN_DIR" >/dev/null 2>&1; then
  echo "RESUME $RUN_DIR (generation verified under expected contract)"
else
  quarantine "$RUN_DIR"
  TRAIN_ARGS=(--k "$K_POS" --interval "$INTERVAL"
    --steps "$STEPS" --lr "$LR" --seed "$SEED" --warmup "$WARMUP"
    --clip "$CLIP" --optimizer "$OPTIMIZER" --weight-decay "$WEIGHT_DECAY"
    --lr-schedule "$LR_SCHEDULE" --max-k "$MAX_K" --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA")
  # shellcheck disable=SC2181
  [ "$DETACH_Z0" = "true" ] && TRAIN_ARGS+=(--detach-z0)
  python -m latent_lab.bench.latent_run train "${TRAIN_ARGS[@]}" \
    --eval-every 100 \
    --label "$LABEL" --device cuda "${COMMON[@]}" --out "$RUN_DIR"
  valid_run "$RUN_DIR" \
    || { echo "FATAL: freshly written $RUN_DIR failed validation" >&2; exit 4; }
fi

ADAPTER_DIGEST=$(python3 -c "import json; print(json.load(open('$RUN_DIR/run_manifest.json'))['checkpoint_content_digest'])")

echo "=== paired SAME-adapter evals: only eval-time K differs ==="
ev () { AD=$1; SP=$2; K=$3;
  F=results/ev4b_${AD}_${SP}_K${K}.json
  if python3 -m latent_lab.bench.artifacts validate-eval "$F" \
      --expect-model "$MODEL" --expect-rev "$REV" --expect-suite "$SUITE_SHA" \
      --expect-digest "$ADAPTER_DIGEST" --expect-split "$SP" \
      --expect-ablation clean --expect-k "$K" --expect-seed "$SEED" \
      >/dev/null 2>&1; then
    echo "RESUME $F (payload verified under expected contract)"
    return
  fi
  quarantine "$F"
  python -m latent_lab.bench.latent_run eval --adapter "runs/$AD" \
    --split "$SP" --k "$K" --seed "$SEED" --device cuda --out "$F"
  python3 -m latent_lab.bench.artifacts validate-eval "$F" \
      --expect-model "$MODEL" --expect-rev "$REV" --expect-suite "$SUITE_SHA" \
      --expect-digest "$ADAPTER_DIGEST" --expect-split "$SP" \
      --expect-ablation clean --expect-k "$K" --expect-seed "$SEED" >/dev/null \
    || { echo "FATAL: freshly written $F failed validation" >&2; exit 4; }
  python3 -c "import json;d=json.load(open('$F'));r=d['results'][list(d['results'])[-1]];print('EVAL $AD $SP K=$K micro_accuracy', r['metrics']['micro_accuracy'], 'n', r['n'])"
}

ev "$LABEL" test_id "$K_POS"
ev "$LABEL" test_id "$K_ZERO"
ev "$LABEL" test_ood_length "$K_POS"
ev "$LABEL" test_ood_length "$K_ZERO"
ev "$LABEL" test_ood_semantic "$K_POS"
ev "$LABEL" test_ood_semantic "$K_ZERO"
ev "$LABEL" final_test "$K_POS"
ev "$LABEL" final_test "$K_ZERO"

echo ALL_DONE_4B
