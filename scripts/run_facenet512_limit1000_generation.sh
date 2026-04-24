#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[$(date '+%F %T')] starting Facenet512 limit1000 generation"

gen_threads="${GEN_THREADS:-32}"

echo "[$(date '+%F %T')] generation threads=$gen_threads"

python3 attack_plus.py \
  --input-csv input2400.csv \
  --base-path dataset_extractedfaces \
  --thresholds-json thresholds.json \
  --attackers Facenet512 \
  --attacks PGD_SM,MI_FGSM_SM,TI_FGSM_SM,SI_NI_FGSM_SM,MI_ADMIX_DI_TI_SM \
  --threads "$gen_threads" \
  --tf-threads 1 \
  --batch-size 1 \
  --print-every-batches 20 \
  --num-iter 12 \
  --source-lambda 0.20 \
  --limit 1000 \
  --adv-root results/Facenet512/adv_images_all12_lambda020_limit1000 \
  --output-csv results/Facenet512/transfer_adv_paths_all12_lambda020_limit1000_sm.csv

echo "[$(date '+%F %T')] Facenet512 generation complete"
