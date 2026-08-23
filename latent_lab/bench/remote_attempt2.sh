#!/usr/bin/env bash
# Attempt #2: stabilized schedules for the decisive pair E_k4 vs F_k0.
cd /root/rcc
LOG=results/attempt2.log
exec > >(tee -a $LOG) 2>&1
train () { L=$1; K=$2; IV=$3; ST=$4; SD=$5; LR=$6;
  if [ ! -f runs/$L/train_report.json ]; then
    python -m latent_lab.bench.latent_run train --k $K --interval $IV \
      --steps $ST --lr $LR --seed $SD --lr-schedule cosine --warmup 50 \
      --clip 0.25 --eval-every 150 --val-examples 56 \
      --device cuda --out runs/$L
  fi }
train E_k4c_s0 4 mid 1200 0 5e-5
train F_k0c_s0 0 mid 1200 0 5e-5
train E_k4c_s1 4 mid 1200 1 5e-5
train F_k0c_s1 0 mid 1200 1 5e-5
for AD in E_k4c_s0 F_k0c_s0 E_k4c_s1 F_k0c_s1; do
  python -m latent_lab.bench.latent_run eval --adapter runs/$AD \
    --split test_id --device cuda --out results/ev_${AD}_test_id_clean.json
  python -m latent_lab.bench.latent_run eval --adapter runs/$AD \
    --split test_ood --device cuda --out results/ev_${AD}_test_ood_clean.json
done
echo ATTEMPT2_DONE
