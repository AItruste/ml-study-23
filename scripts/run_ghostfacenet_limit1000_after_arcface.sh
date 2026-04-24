#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARC_SYNC="sync_attack_performance.py --input-csv results/ARCFACE/transfer_adv_paths_all12_lambda020_limit1000.csv"
while pgrep -f "$ARC_SYNC" >/dev/null 2>&1; do
  echo "[$(date '+%F %T')] waiting for ArcFace rescoring to finish..."
  sleep 30
done

echo "[$(date '+%F %T')] starting GhostFaceNet limit1000 run"

python3 attack_plus.py \
  --input-csv input2400.csv \
  --base-path dataset_extractedfaces \
  --thresholds-json thresholds.json \
  --attackers GhostFaceNet \
  --attacks PGD_SM,MI_FGSM_SM,TI_FGSM_SM,SI_NI_FGSM_SM,MI_ADMIX_DI_TI_SM \
  --threads 30 \
  --tf-threads 1 \
  --batch-size 1 \
  --print-every-batches 20 \
  --num-iter 12 \
  --source-lambda 0.20 \
  --limit 1000 \
  --adv-root results/Ghostfacenet/adv_images_all12_lambda020_limit1000 \
  --output-csv results/Ghostfacenet/transfer_adv_paths_all12_lambda020_limit1000_sm.csv

python3 - <<'PY'
import pandas as pd
old_csv = 'results/Ghostfacenet/transfer_adv_paths_all12.csv'
new_csv = 'results/Ghostfacenet/transfer_adv_paths_all12_lambda020_limit1000_sm.csv'
out_csv = 'results/Ghostfacenet/transfer_adv_paths_all12_lambda020_limit1000.csv'
key_cols = ['row_id', 'attacker_model']
sm_cols = ['pgd_sm_path','mi_fgsm_sm_path','ti_fgsm_sm_path','si_ni_fgsm_sm_path','mi_admix_di_ti_sm_path']
old_df = pd.read_csv(old_csv)
new_df = pd.read_csv(new_csv)
old_df['row_id'] = pd.to_numeric(old_df['row_id'], errors='raise').astype(int)
new_df['row_id'] = pd.to_numeric(new_df['row_id'], errors='raise').astype(int)
old_df['attacker_model'] = old_df['attacker_model'].astype(str)
new_df['attacker_model'] = new_df['attacker_model'].astype(str)
new_keys_df = new_df[key_cols].drop_duplicates().copy()
old_subset = old_df.merge(new_keys_df, on=key_cols, how='inner', validate='many_to_one')
merged = old_subset.merge(new_df[key_cols + sm_cols], on=key_cols, how='left', suffixes=('', '__new'), validate='one_to_one')
for col in sm_cols:
    new_col = f'{col}__new'
    merged[col] = merged[new_col].fillna('').astype(str)
    merged.drop(columns=[new_col], inplace=True)
merged = merged.sort_values(['row_id', 'attacker_model'], kind='stable').reset_index(drop=True)
merged.to_csv(out_csv, index=False)
print(f'wrote {out_csv} rows={len(merged)}')
PY

python3 sync_attack_performance.py \
  --input-csv results/Ghostfacenet/transfer_adv_paths_all12_lambda020_limit1000.csv \
  --output-csv results/Ghostfacenet/transfer_attack_performance_all12_lambda020_limit1000.csv \
  --similarity-csv results/Ghostfacenet/transfer_attack_similarity_scores_all12_lambda020_limit1000.csv \
  --workers 24 \
  --tf-threads 1 \
  --progress-every 20 \
  --checkpoint-every 20

echo "[$(date '+%F %T')] GhostFaceNet limit1000 run complete"
