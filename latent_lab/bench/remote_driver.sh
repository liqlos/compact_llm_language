#!/usr/bin/env bash
# Remote one-shot experiment driver for the RCC behavioral gate.
# Runs on a vast.ai RTX 3090/4090 instance (CUDA). Sequential only.
#
# Sealed contract (fail closed):
#   * unit-test failures ABORT the driver (no `|| true`);
#   * resume NEVER trusts bare file existence: runs/evals count as done
#     only when their status/digest/identity validation passes;
#     unverified leftovers are quarantined (*.invalid.<ts>) and redone.
set -euo pipefail
cd /root/rcc
mkdir -p runs results
LOG=results/driver.log
exec > >(tee -a $LOG) 2>&1

echo "=== env ==="
python - <<'PY'
import torch, transformers, platform
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "-")
print(platform.platform())
PY

echo "=== sanity: unit tests (cpu) — failures are fatal ==="
python -m pytest tests/test_suite.py tests/test_localized.py \
    tests/test_latent_runtime_integrity.py -q

quarantine () {
  local P=$1
  if [ -e "$P" ]; then
    mv "$P" "$P.invalid.$(date +%s)" && echo "QUARANTINED $P"
  fi
}
valid_run () { python3 -m latent_lab.bench.artifacts validate-run "$1"; }
valid_eval () { python3 -m latent_lab.bench.artifacts validate-eval "$1"; }

echo "=== textual baselines ==="
for MODE in A C B; do
  for SPLIT in test_id test_ood; do
    F=results/text_${MODE}_${SPLIT}.json
    if valid_eval "$F" >/dev/null 2>&1; then
      echo "RESUME $F"; continue
    fi
    quarantine "$F"
    python -m latent_lab.bench.text_baselines --mode $MODE --split $SPLIT \
      --batch 16 --device cuda --out "$F"
  done
done

echo "=== latent training ==="
train () { # label k interval steps seed lr
  L=$1; K=$2; IV=$3; ST=$4; SD=$5; LR=${6:-1e-4}
  if valid_run "runs/$L" >/dev/null 2>&1; then
    echo "RESUME runs/$L (generation verified)"
    return
  fi
  quarantine "runs/$L"
  python -m latent_lab.bench.latent_run train --k $K --interval $IV \
    --steps $ST --lr $LR --seed $SD --eval-every 100 --val-examples 28 \
    --device cuda --out runs/$L
}
train E_k4_s0 4 mid 800 0
train F_k0_s0 0 mid 800 0
train E_k1_s0 1 mid 400 0
train E_k2_s0 2 mid 400 0
train E_k8_s0 8 mid 400 0
train E_k16_s0 16 mid 400 0
train D_full_k4_s0 4 full 400 0
train E_k4_s1 4 mid 800 1
train E_k4_s2 4 mid 800 2

echo "=== evals: clean + ablations on test_id; clean on test_ood ==="
ev () { # adapter split ablate out extra
  AD=$1; SP=$2; AB=${3:-clean}; OUT=$4; EX=${5:-}
  F=results/ev_${AD}_${SP}_${AB}${EX}.json
  if valid_eval "$F" >/dev/null 2>&1; then
    echo "RESUME $F (payload verified)"
    return
  fi
  quarantine "$F"
  if [ "$AB" = "clean" ]; then ABFLAG=""; else ABFLAG="--ablate $AB"; fi
  python -m latent_lab.bench.latent_run eval --adapter runs/$AD \
    --split $SP $ABFLAG --device cuda --out "$F" $EX
}
for AD in E_k4_s0 F_k0_s0 E_k1_s0 E_k2_s0 E_k8_s0 E_k16_s0 D_full_k4_s0 E_k4_s1 E_k4_s2; do
  ev $AD test_id clean "" ""
  ev $AD test_ood clean "" ""
done
AD=E_k4_s0
for AB in zero_state bypass_interval clocks_off reverse_clocks truncate_half \
           swap_state noise_state; do
  ev $AD test_id $AB "" ""
done

echo "=== K-generalization of E_k4_s0 ==="
for K in 1 2 8 16; do
  F=results/ev_E_k4_s0_test_id_clean_K$K.json
  if valid_eval "$F" >/dev/null 2>&1; then
    echo "RESUME $F (payload verified)"
    continue
  fi
  quarantine "$F"
  python -m latent_lab.bench.latent_run eval --adapter runs/E_k4_s0 \
    --split test_id --k $K --device cuda --out "$F"
done

echo "DRIVER_DONE"
