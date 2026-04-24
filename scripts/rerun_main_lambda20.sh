#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SM_ATTACKS="PGD_SM,MI_FGSM_SM,TI_FGSM_SM,SI_NI_FGSM_SM,MI_ADMIX_DI_TI_SM"
THREADS="${THREADS:-30}"
SYNC_WORKERS="${SYNC_WORKERS:-24}"
TF_THREADS="${TF_THREADS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_ITER="${NUM_ITER:-12}"
PRINT_EVERY_BATCHES="${PRINT_EVERY_BATCHES:-10}"

run_one() {
  local attacker="$1"
  local result_dir="$2"
  local old_adv_csv="$3"
  local combined_adv_csv="$4"
  local sm_adv_csv="$5"
  local perf_csv="$6"
  local sim_csv="$7"
  local adv_root="$8"

  mkdir -p "$result_dir"

  echo "=================================================================="
  echo "[lambda20] attacker=$attacker"
  echo "[lambda20] result_dir=$result_dir"
  echo "[lambda20] step 1/3 generate SM adversarial images"
  python3 attack_plus.py \
    --input-csv input2400.csv \
    --base-path dataset_extractedfaces \
    --thresholds-json thresholds.json \
    --attackers "$attacker" \
    --attacks "$SM_ATTACKS" \
    --threads "$THREADS" \
    --tf-threads "$TF_THREADS" \
    --batch-size "$BATCH_SIZE" \
    --print-every-batches "$PRINT_EVERY_BATCHES" \
    --num-iter "$NUM_ITER" \
    --source-lambda 0.20 \
    --adv-root "$adv_root" \
    --output-csv "$sm_adv_csv"

  echo "[lambda20] step 2/3 merge new SM paths into combined adv csv"
  python3 - "$old_adv_csv" "$sm_adv_csv" "$combined_adv_csv" <<'PY'
import sys
import pandas as pd

old_csv, new_csv, out_csv = sys.argv[1:4]
key_cols = ["row_id", "attacker_model"]
sm_cols = [
    "pgd_sm_path",
    "mi_fgsm_sm_path",
    "ti_fgsm_sm_path",
    "si_ni_fgsm_sm_path",
    "mi_admix_di_ti_sm_path",
]

old_df = pd.read_csv(old_csv)
new_df = pd.read_csv(new_csv)

for col in old_df.columns:
    if col not in new_df.columns:
        new_df[col] = ""

old_df["row_id"] = pd.to_numeric(old_df["row_id"], errors="raise").astype(int)
new_df["row_id"] = pd.to_numeric(new_df["row_id"], errors="raise").astype(int)
old_df["attacker_model"] = old_df["attacker_model"].astype(str)
new_df["attacker_model"] = new_df["attacker_model"].astype(str)

old_keys = set(map(tuple, old_df[key_cols].to_records(index=False)))
new_keys = set(map(tuple, new_df[key_cols].to_records(index=False)))
if old_keys != new_keys:
    missing = sorted(old_keys - new_keys)[:10]
    extra = sorted(new_keys - old_keys)[:10]
    raise SystemExit(
        f"Key mismatch between old and new adv CSVs. "
        f"missing_in_new={len(old_keys - new_keys)} sample={missing} "
        f"extra_in_new={len(new_keys - old_keys)} sample={extra}"
    )

merged = old_df.merge(
    new_df[key_cols + sm_cols],
    on=key_cols,
    how="left",
    suffixes=("", "__new"),
    validate="one_to_one",
)

for col in sm_cols:
    new_col = f"{col}__new"
    merged[col] = merged[new_col].fillna("").astype(str)
    merged.drop(columns=[new_col], inplace=True)

merged.to_csv(out_csv, index=False)
print(f"wrote {out_csv}")
PY

  echo "[lambda20] step 3/3 rescore all attacks with combined adv csv"
  python3 sync_attack_performance.py \
    --input-csv "$combined_adv_csv" \
    --output-csv "$perf_csv" \
    --similarity-csv "$sim_csv" \
    --workers "$SYNC_WORKERS" \
    --tf-threads "$TF_THREADS" \
    --progress-every 20 \
    --checkpoint-every 20
}

run_one \
  "ArcFace" \
  "results/ARCFACE" \
  "results/ARCFACE/transfer_adv_paths_all12.csv" \
  "results/ARCFACE/transfer_adv_paths_all12_lambda020.csv" \
  "results/ARCFACE/transfer_adv_paths_all12_lambda020_sm.csv" \
  "results/ARCFACE/transfer_attack_performance_all12_lambda020.csv" \
  "results/ARCFACE/transfer_attack_similarity_scores_all12_lambda020.csv" \
  "results/ARCFACE/adv_images_all12_lambda020"

run_one \
  "Facenet512" \
  "results/Facenet512" \
  "results/Facenet512/transfer_adv_paths_all12_baseline_legacy.csv" \
  "results/Facenet512/transfer_adv_paths_all12_lambda020.csv" \
  "results/Facenet512/transfer_adv_paths_all12_lambda020_sm.csv" \
  "results/Facenet512/transfer_attack_performance_all12_lambda020.csv" \
  "results/Facenet512/transfer_attack_similarity_scores_all12_lambda020.csv" \
  "results/Facenet512/adv_images_all12_lambda020"

run_one \
  "GhostFaceNet" \
  "results/Ghostfacenet" \
  "results/Ghostfacenet/transfer_adv_paths_all12.csv" \
  "results/Ghostfacenet/transfer_adv_paths_all12_lambda020.csv" \
  "results/Ghostfacenet/transfer_adv_paths_all12_lambda020_sm.csv" \
  "results/Ghostfacenet/transfer_attack_performance_all12_lambda020.csv" \
  "results/Ghostfacenet/transfer_attack_similarity_scores_all12_lambda020.csv" \
  "results/Ghostfacenet/adv_images_all12_lambda020"

run_one \
  "VGG-Face" \
  "results/VGG-Face" \
  "results/VGG-Face/transfer_adv_paths_all12.csv" \
  "results/VGG-Face/transfer_adv_paths_all12_lambda020.csv" \
  "results/VGG-Face/transfer_adv_paths_all12_lambda020_sm.csv" \
  "results/VGG-Face/transfer_attack_performance_all12_lambda020.csv" \
  "results/VGG-Face/transfer_attack_similarity_scores_all12_lambda020.csv" \
  "results/VGG-Face/adv_images_all12_lambda020"

echo "[lambda20] main rerun completed for all four attackers."
