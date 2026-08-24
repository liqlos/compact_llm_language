#!/usr/bin/env bash
# Sealed PREREGISTERED paid canary — Qwen/Qwen3.5-4B.
#
# Contract (fail closed):
#   * Focused unit/integrity tests MUST pass BEFORE any GPU/model spend;
#     any failure aborts the driver.
#   * EXACT dependency identity (equality against pinned versions,
#     including Python, torch, transformers, huggingface_hub AND the
#     project uv.lock content hash): no ranges, NO installs/upgrades
#     ("pip -U" is banned), NO ignored failures (set -euo pipefail).
#   * ONE preregistered seed/recipe fixed BELOW, before any GPU second:
#     nothing in this driver ever inspects validation results to choose
#     a seed, an adapter or a recipe (no BEST/MEDIAN cherry-picking).
#   * Paired causal arm: K>0 and K=0 are evaluated on the SAME trained
#     adapter/seed/suite (only the eval-time K differs). Separate F
#     adapters do NOT satisfy this contract and are not used.
#   * Bounded by construction: exactly ONE training run + four paired
#     evaluations. Any matrix expansion requires its own separate
#     pre-spend authorization and is refused here.
#   * Resume NEVER trusts bare existence: a directory/payload counts as
#     done ONLY when it re-validates under the FULL expected contract;
#     anything else is quarantined (*.invalid.<ts>) and recomputed.
set -euo pipefail
cd /root/rcc

if [ -n "${DRIVER_MATRIX:-}" ]; then
  echo "FATAL: DRIVER_MATRIX is set; matrix expansion is NOT part of this" \
       "canary and requires separate pre-spend authorization." >&2
  exit 5
fi

mkdir -p runs results
exec 9>results/.driver4b.lock
if ! flock -n 9; then
  echo "FATAL: another driver holds results/.driver4b.lock" >&2
  exit 3
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
LABEL=E4_k${K_POS}_s${SEED}
RUN_DIR=runs/${LABEL}

echo "=== sealed environment verification (EXACT pins; no installs) ==="
PIN_PYTHON="3.14.0"
PIN_TORCH="2.13.0"
PIN_TRANSFORMERS="5.15.1"
PIN_HUGGINGFACE_HUB="1.28.0"
PIN_UVLOCK_SHA256="62187a854931549a8cd927537a3cf393759fd56b79152c5f400447b9c3de035f"

python - <<PY
import sys
import huggingface_hub
import torch
import transformers

def require(actual, pinned, what):
    assert actual == pinned, f"{what} {actual!r} != pinned {pinned!r}"

require(sys.version.split()[0], "${PIN_PYTHON}", "python")
require(torch.__version__, "${PIN_TORCH}", "torch")
require(transformers.__version__, "${PIN_TRANSFORMERS}", "transformers")
require(huggingface_hub.__version__, "${PIN_HUGGINGFACE_HUB}",
        "huggingface_hub")
assert torch.cuda.is_available(), "CUDA unavailable"
print("ENV_OK", sys.version.split()[0], torch.__version__,
      transformers.__version__, huggingface_hub.__version__,
      torch.cuda.get_device_name(0))
PY

ACTUAL_LOCK_SHA=$(sha256sum uv.lock | cut -d ' ' -f1)
if [ "$ACTUAL_LOCK_SHA" != "$PIN_UVLOCK_SHA256" ]; then
  echo "FATAL: project lock drift: uv.lock sha256 $ACTUAL_LOCK_SHA != pinned $PIN_UVLOCK_SHA256" >&2
  exit 2
fi

echo "=== pre-spend gate: focused checks MUST pass before ANY GPU work ==="
python -m pytest -q \
  tests/test_latent_runtime_integrity.py \
  tests/test_latent_run.py \
  tests/test_artifact_contracts.py \
  tests/test_paid_driver_sealed.py

SUITE_SHA=$(python -c "from latent_lab.bench.suite import build_suite; print(build_suite().manifest()['sha256'])")

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
            --expect-seed "$SEED" --expect-label "$LABEL"
            --expect-k "$K_POS" --expect-steps "$STEPS")
valid_run () {
  python3 -m latent_lab.bench.artifacts validate-run "$1" "${expect_run[@]}"
}

echo "=== single preregistered training run: label=$LABEL seed=$SEED k=$K_POS ==="
if valid_run "$RUN_DIR" >/dev/null 2>&1; then
  echo "RESUME $RUN_DIR (generation verified under expected contract)"
else
  quarantine "$RUN_DIR"
  python -m latent_lab.bench.latent_run train --k "$K_POS" --interval "$INTERVAL" \
    --steps "$STEPS" --lr "$LR" --seed "$SEED" --warmup "$WARMUP" --clip "$CLIP" \
    --optimizer "$OPTIMIZER" --weight-decay "$WEIGHT_DECAY" \
    --lr-schedule "$LR_SCHEDULE" \
    --eval-every 100 --val-examples 28 \
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
  python3 -c "import json;d=json.load(open('$F'));r=d['results'][list(d['results'])[-1]];print('EVAL $AD $SP K=$K acc', r['accuracy'], 'n', r['n'])"
}

ev "$LABEL" test_id "$K_POS"
ev "$LABEL" test_id "$K_ZERO"
ev "$LABEL" test_ood "$K_POS"
ev "$LABEL" test_ood "$K_ZERO"

echo ALL_DONE_4B
