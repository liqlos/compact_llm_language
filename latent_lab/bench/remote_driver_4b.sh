#!/usr/bin/env bash
# Sealed remote behavioral gate for Qwen3.5-4B (same suite/sha as the 2B gate).
#
# Contract (fail closed):
#   * NO package installs/upgrades: the image ships pinned deps; env is
#     verified and any mismatch aborts the driver.
#   * All runs serialize under an exclusive lock (dead-holder recovery).
#   * Resume is NEVER based on bare file existence: a training directory or
#     eval payload counts as done only when its digests/status/identity
#     validate; anything else is quarantined (renamed *.invalid.<ts>) and
#     recomputed.
set -euo pipefail
cd /root/rcc

LOCK=results/.driver4b.lock
mkdir -p runs results

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "FATAL: another driver holds $LOCK" >&2
  exit 3
fi

MODEL=Qwen/Qwen3.5-4B
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
COMMON="--model $MODEL --revision $REV"

echo "=== sealed environment verification (no installs) ==="
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "CUDA unavailable"
def mm(v): return ".".join(v.split(".")[:2])
assert mm(torch.__version__) == "2.13", f"torch {torch.__version__} != pinned 2.13.x"
assert int(transformers.__version__.split(".")[0]) >= 5, \
    f"transformers {transformers.__version__} too old"
print("ENV_OK", torch.__version__, transformers.__version__,
      torch.cuda.get_device_name(0))
PY
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

quarantine () {
  local P=$1
  if [ -e "$P" ]; then
    mv "$P" "$P.invalid.$(date +%s)" && echo "QUARANTINED $P"
  fi
}

valid_run () { python3 -m latent_lab.bench.artifacts validate-run "$1"; }

train () { L=$1; K=$2; IV=$3; ST=$4; SD=$5;
  if valid_run "runs/$L" >/dev/null 2>&1; then
    echo "RESUME runs/$L (generation verified)"
    return
  fi
  quarantine "runs/$L"
  python -m latent_lab.bench.latent_run train --k $K --interval $IV \
    --steps $ST --lr 1e-4 --seed $SD --warmup 50 --clip 0.5 \
    --eval-every 100 --val-examples 28 \
    --device cuda $COMMON --out runs/$L
}
train E4_k4_s0 4 mid 800 0
train F4_k0_s0 0 mid 800 0
train E4_k4_s1 4 mid 800 1
train F4_k0_s1 0 mid 800 1
train E4_k4_s2 4 mid 800 2
train F4_k0_s2 0 mid 800 2
train E4_k1_s0 1 mid 400 0
train E4_k8_s0 8 mid 400 0

echo "=== select best/median E seeds ==="
read BEST MEDIAN <<< "$(python - <<'PY'
import json, glob, subprocess, sys
rows = []
for p in sorted(glob.glob("runs/E4_k4_s*/train_report.json")):
    d = p.rsplit("/", 1)[0]
    r = subprocess.run([sys.executable, "-m",
                        "latent_lab.bench.artifacts", "validate-run", d])
    if r.returncode != 0:
        raise SystemExit(f"unverified run dir: {d}")
    rows.append((json.load(open(p))["best_val_acc"], d))
if len(rows) < 2:
    raise SystemExit("need >=2 verified E runs for best/median selection")
rows.sort()
strip = lambda s: s.replace("E4_k4_", "").replace("s", "")
print(strip(rows[-1][1]), strip(rows[len(rows)//2][1]))
PY
)"
echo "BEST=$BEST MEDIAN=$MEDIAN"

echo "=== evals ==="
ev () { AD=$1; SP=$2; AB=${3:-clean}
  F=results/ev4b_${AD}_${SP}_${AB}.json
  if python3 -m latent_lab.bench.artifacts validate-eval "$F" >/dev/null 2>&1; then
    echo "RESUME $F (payload verified)"
    return
  fi
  quarantine "$F"
  if [ "$AB" = "clean" ]; then ABFLAG=""; else ABFLAG="--ablate $AB"; fi
  python -m latent_lab.bench.latent_run eval --adapter runs/$AD \
    --split $SP $ABFLAG --device cuda --out $F
  python3 -m latent_lab.bench.artifacts validate-eval "$F" >/dev/null \
    || { echo "FATAL: freshly written $F failed validation" >&2; exit 4; }
  python -c "import json;d=json.load(open('$F'));r=d['results'][list(d['results'])[-1]];print('EVAL $AD $SP $AB acc', r['accuracy'], 'n', r['n'])"
}
for S in 0 1 2; do
  ev "E4_k4_s$S" test_id clean ""
  ev "E4_k4_s$S" test_ood clean ""
  ev "F4_k0_s$S" test_id clean ""
  ev "F4_k0_s$S" test_ood clean ""
done
ev E4_k1_s0 test_id clean ""; ev E4_k1_s0 test_ood clean ""
ev E4_k8_s0 test_id clean ""; ev E4_k8_s0 test_ood clean ""
for S in $BEST $MEDIAN; do
  for AB in zero_state bypass_interval clocks_off reverse_clocks truncate_half swap_state noise_state; do
    ev "E4_k4_s$S" test_id $AB
  done
done
for KK in 1 2 8 16; do
  F=results/ev4b_E4_k4_s${BEST}_test_id_cleanK${KK}.json
  if python3 -m latent_lab.bench.artifacts validate-eval "$F" >/dev/null 2>&1; then
    echo "RESUME $F (payload verified)"
  else
    quarantine "$F"
    python -m latent_lab.bench.latent_run eval --adapter runs/E4_k4_s$BEST \
      --split test_id --k $KK --device cuda --out $F
    python3 -m latent_lab.bench.artifacts validate-eval "$F" >/dev/null \
      || { echo "FATAL: freshly written $F failed validation" >&2; exit 4; }
  fi
done
echo ALL_DONE_4B
