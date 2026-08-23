#!/usr/bin/env bash
# Remote behavioral gate for Qwen3.5-4B (same suite/sha as the 2B gate).
set -euo pipefail
cd /root/rcc
mkdir -p runs results
LOG=results/driver4b.log
exec > >(tee -a $LOG) 2>&1

MODEL=Qwen/Qwen3.5-4B
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
COMMON="--model $MODEL --revision $REV"

echo "=== setup ==="
pip install --no-cache-dir -q -U torch transformers packaging pytest huggingface_hub || true
pip uninstall -y -q torchaudio torchvision 2>/dev/null || true
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq build-essential >/dev/null 2>&1 || true
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "CUDA unavailable"
print("ENV_OK", torch.__version__, transformers.__version__,
      torch.cuda.get_device_name(0))
PY
python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV')); print('MODEL_OK')"

echo "=== textual baselines ==="
for MODE in A C B; do
  for SPLIT in test_id test_ood; do
    F=results/text4b_${MODE}_${SPLIT}.json
    if [ ! -f "$F" ]; then
      python -m latent_lab.bench.text_baselines --mode $MODE --split $SPLIT \
        --batch 16 --device cuda --out $F $COMMON
    fi
  done
done
python - <<'PY'
import json
for sp in ("test_id", "test_ood"):
    d = json.load(open(f"results/text4b_B_{sp}.json"))
    nt = d["non_termination_count"] / d["n_examples"]
    print(f"B nonterm rate {sp}: {nt:.2f}")
PY

echo "=== trainings ==="
train () { L=$1; K=$2; IV=$3; ST=$4; SD=$5;
  if [ ! -f runs/$L/train_report.json ]; then
    python -m latent_lab.bench.latent_run train --k $K --interval $IV \
      --steps $ST --lr 1e-4 --seed $SD --warmup 50 --clip 0.5 \
      --eval-every 100 --val-examples 28 \
      --device cuda $COMMON --out runs/$L
  fi }
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
import json, glob
rows=[]
for p in glob.glob("runs/E4_k4_s*/train_report.json"):
    r=json.load(open(p))
    rows.append((r["best_val_acc"], p.split("/")[1]))
rows.sort()
print(rows[-1][1].replace("E4_k4_","").replace("s",""), rows[len(rows)//2][1].replace("E4_k4_","").replace("s",""))
PY
)"
echo "BEST=$BEST MEDIAN=$MEDIAN"

echo "=== evals ==="
ev () { AD=$1; SP=$2; AB=${3:-clean}
  F=results/ev4b_${AD}_${SP}_${AB}.json
  if [ ! -f "$F" ]; then
    if [ "$AB" = "clean" ]; then ABFLAG=""; else ABFLAG="--ablate $AB"; fi
    python -m latent_lab.bench.latent_run eval --adapter runs/$AD \
      --split $SP $ABFLAG --device cuda --out $F
    python -c "import json;d=json.load(open('$F'));r=d['results'][list(d['results'])[-1]];print('EVAL $AD $SP $AB acc', r['accuracy'], 'n', r['n'])"
  fi }
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
  if [ ! -f "$F" ]; then
    python -m latent_lab.bench.latent_run eval --adapter runs/E4_k4_s$BEST \
      --split test_id --k $KK --device cuda --out $F
  fi
done
echo ALL_DONE_4B
