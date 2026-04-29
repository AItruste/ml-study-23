from pathlib import Path

import pandas as pd


def _normalize_path(value: str) -> Path:
    p = Path(str(value).strip())
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def collect_referenced_adv_paths(output_csv, attack_cols, selected_models=None, selected_attacks=None):
    csv_path = Path(output_csv)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    df = pd.read_csv(csv_path)
    needed_cols = ["row_id", "attacker_model"] + list(attack_cols.values())
    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    df["attacker_model"] = df["attacker_model"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["row_id", "attacker_model"], keep="last")

    selected_model_set = set(selected_models or [])
    selected_attack_set = set(selected_attacks or attack_cols.keys())

    keep = set()
    for _, rec in df.iterrows():
        attacker = str(rec["attacker_model"]).strip()
        if selected_model_set and attacker not in selected_model_set:
            continue
        for attack_name, col in attack_cols.items():
            if attack_name not in selected_attack_set:
                continue
            value = str(rec.get(col, "")).strip()
            if not value:
                continue
            keep.add(_normalize_path(value))
    return keep


def cleanup_orphan_adv_images(output_csv, adv_root, attack_cols, selected_models=None, selected_attacks=None, dry_run=False):
    adv_root_path = Path(adv_root).resolve()
    selected_model_list = list(selected_models or [])
    selected_attack_list = list(selected_attacks or attack_cols.keys())
    keep = collect_referenced_adv_paths(
        output_csv=output_csv,
        attack_cols=attack_cols,
        selected_models=selected_model_list,
        selected_attacks=selected_attack_list,
    )

    scanned_files = 0
    removed_files = 0
    removed_dirs = 0
    removed_bytes = 0

    for model_name in selected_model_list:
        for attack_name in selected_attack_list:
            attack_dir = adv_root_path / model_name / attack_name
            if not attack_dir.exists():
                continue
            for path in attack_dir.rglob("*"):
                if not path.is_file():
                    continue
                scanned_files += 1
                resolved = path.resolve()
                if resolved in keep:
                    continue
                removed_files += 1
                try:
                    removed_bytes += path.stat().st_size
                except OSError:
                    pass
                if not dry_run:
                    path.unlink(missing_ok=True)

            # Remove empty directories bottom-up after orphan files are deleted.
            if not dry_run:
                for subdir in sorted([p for p in attack_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
                    try:
                        subdir.rmdir()
                        removed_dirs += 1
                    except OSError:
                        pass
                try:
                    attack_dir.rmdir()
                    removed_dirs += 1
                except OSError:
                    pass
                try:
                    model_dir = adv_root_path / model_name
                    model_dir.rmdir()
                    removed_dirs += 1
                except OSError:
                    pass

    return {
        "scanned_files": scanned_files,
        "kept_referenced_files": len(keep),
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "removed_bytes": removed_bytes,
        "dry_run": bool(dry_run),
    }
