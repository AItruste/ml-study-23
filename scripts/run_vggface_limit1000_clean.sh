#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[$(date '+%F %T')] starting VGG-Face limit1000 clean run"

gen_threads="${GEN_THREADS:-32}"
sync_workers="${SYNC_WORKERS:-24}"

echo "[$(date '+%F %T')] generation threads=$gen_threads sync_workers=$sync_workers"

python3 attack_plus.py \
  --input-csv input2400.csv \
  --base-path dataset_extractedfaces \
  --thresholds-json thresholds.json \
  --attackers VGG-Face \
  --attacks PGD_SM,MI_FGSM_SM,TI_FGSM_SM,SI_NI_FGSM_SM,MI_ADMIX_DI_TI_SM \
  --threads "$gen_threads" \
  --tf-threads 1 \
  --batch-size 1 \
  --print-every-batches 20 \
  --num-iter 12 \
  --source-lambda 0.20 \
  --limit 1000 \
  --adv-root results/VGG-Face/adv_images_all12_lambda020_limit1000 \
  --output-csv results/VGG-Face/transfer_adv_paths_all12_lambda020_limit1000_sm.csv

python3 - <<'PY'
import pandas as pd
old_csv = 'results/VGG-Face/transfer_adv_paths_all12.csv'
new_csv = 'results/VGG-Face/transfer_adv_paths_all12_lambda020_limit1000_sm.csv'
out_csv = 'results/VGG-Face/transfer_adv_paths_all12_lambda020_limit1000.csv'
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
  --input-csv results/VGG-Face/transfer_adv_paths_all12_lambda020_limit1000_sm.csv \
  --output-csv results/VGG-Face/transfer_attack_performance_all12_lambda020_limit1000_sm_only.csv \
  --similarity-csv results/VGG-Face/transfer_attack_similarity_scores_all12_lambda020_limit1000.csv \
  --attacks PGD_SM,MI_FGSM_SM,TI_FGSM_SM,SI_NI_FGSM_SM,MI_ADMIX_DI_TI_SM \
  --workers "$sync_workers" \
  --tf-threads 1 \
  --progress-every 50 \
  --checkpoint-every 20 \
  --skip-charts \
  --disable-faceapi

python3 - <<'PY'
from pathlib import Path
import pandas as pd
base = pd.read_csv('results/VGG-Face/transfer_attack_performance_all12.csv')
new = pd.read_csv('results/VGG-Face/transfer_attack_performance_all12_lambda020_limit1000_sm_only.csv')
limit_df = pd.read_csv('results/VGG-Face/transfer_adv_paths_all12_lambda020_limit1000_sm.csv', usecols=['row_id'])
limit_ids = set(limit_df['row_id'].astype(int).tolist())
base = base[(base['attacker_model'].astype(str) == 'VGG-Face') & (base['row_id'].astype(int).isin(limit_ids))].copy()
key_cols = ['row_id','attacker_model','victim_model','dataset','attack_type']
sm_cols = [c for c in new.columns if c.endswith('_sm_adv_similarity') or c.endswith('_sm_breach') or c.endswith('_sm_impact') or c.endswith('_sm_adv_image_path')]
overlay = new[key_cols + sm_cols].copy()
merged = base.merge(overlay, on=key_cols, how='left', suffixes=('', '__new'))
for col in sm_cols:
    newcol = f'{col}__new'
    if newcol in merged.columns:
        if col in merged.columns:
            merged[col] = merged[newcol].combine_first(merged[col])
        else:
            merged[col] = merged[newcol]
        merged.drop(columns=[newcol], inplace=True)
out = Path('results/VGG-Face/transfer_attack_performance_all12_lambda020_limit1000.csv')
merged.to_csv(out, index=False)
print('wrote', out, 'rows', len(merged), 'cols', len(merged.columns))
PY

echo "[$(date '+%F %T')] VGG-Face limit1000 clean run complete"
