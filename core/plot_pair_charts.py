#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attack_plus import ATTACK_PAIRS, ALL_ATTACKS, perf_metric_columns


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if {"attack_name", "breach", "impact"}.issubset(set(df.columns)):
        return (
            df.groupby("attack_name", dropna=False)
            .agg(
                breach_rate=("breach", "mean"),
                impact_mean=("impact", "mean"),
                n=("attack_name", "count"),
            )
            .reset_index()
        )

    rows = []
    for attack_name in ALL_ATTACKS:
        metrics = perf_metric_columns(attack_name)
        breach_col = metrics["breach"]
        impact_col = metrics["impact"]
        if breach_col not in df.columns or impact_col not in df.columns:
            continue
        breach_vals = pd.to_numeric(df[breach_col], errors="coerce").dropna()
        impact_vals = pd.to_numeric(df[impact_col], errors="coerce").dropna()
        n = int(min(len(breach_vals), len(impact_vals)))
        if n == 0:
            continue
        rows.append(
            {
                "attack_name": attack_name,
                "breach_rate": float(breach_vals.mean()),
                "impact_mean": float(impact_vals.mean()),
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def value_for(summary: pd.DataFrame, attack_name: str, col: str) -> float:
    row = summary.loc[summary["attack_name"] == attack_name, col]
    if row.empty:
        return 0.0
    return float(row.iloc[0])


def plot_paired_bars(labels, vanilla_vals, plus_vals, ylabel, title, out_path: Path, pct_labels=None) -> None:
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, vanilla_vals, width, label="Vanilla", color="#c0392b")
    plus_bars = ax.bar(x + width / 2, plus_vals, width, label="Plus (SM)", color="#27ae60")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    if pct_labels is not None:
        for bar, txt in zip(plus_bars, pct_labels):
            y = bar.get_height()
            dy = 0.01 if y >= 0 else -0.01
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y + dy, txt, ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pct_improvement(vanilla: float, plus: float) -> float:
    if abs(vanilla) < 1e-12:
        return float("inf") if plus > 0 else (0.0 if abs(plus) < 1e-12 else float("-inf"))
    return ((plus - vanilla) / abs(vanilla)) * 100.0


def create_pair_charts(perf_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(perf_csv)
    summary = summarize(df)
    if summary.empty:
        raise SystemExit(f"No usable attack summary rows found in {perf_csv}")

    labels = []
    impact_vanilla = []
    impact_plus = []
    breach_vanilla = []
    breach_plus = []

    for van, plus in ATTACK_PAIRS:
        if van not in set(summary["attack_name"]) and plus not in set(summary["attack_name"]):
            continue
        labels.append(van)
        impact_vanilla.append(value_for(summary, van, "impact_mean"))
        impact_plus.append(value_for(summary, plus, "impact_mean"))
        breach_vanilla.append(value_for(summary, van, "breach_rate"))
        breach_plus.append(value_for(summary, plus, "breach_rate"))

    if not labels:
        raise SystemExit("No matching attack pairs found in performance CSV.")

    breach_improvements = [pct_improvement(v, p) for v, p in zip(breach_vanilla, breach_plus)]
    impact_improvements = [pct_improvement(v, p) for v, p in zip(impact_vanilla, impact_plus)]
    breach_pct_labels = [f"{v:+.2f}%" for v in breach_improvements]
    impact_pct_labels = [f"{v:+.2f}%" for v in impact_improvements]

    plot_paired_bars(
        labels=labels,
        vanilla_vals=impact_vanilla,
        plus_vals=impact_plus,
        ylabel="Impact Mean",
        title="Impact: Vanilla vs Plus (SM)",
        out_path=out_dir / "impact_pair_chart.png",
        pct_labels=impact_pct_labels,
    )
    plot_paired_bars(
        labels=labels,
        vanilla_vals=breach_vanilla,
        plus_vals=breach_plus,
        ylabel="Breach Rate",
        title="Breach Gain/Rate: Vanilla vs Plus (SM)",
        out_path=out_dir / "gain_pair_chart.png",
        pct_labels=breach_pct_labels,
    )

    print(f"Saved: {out_dir / 'impact_pair_chart.png'}")
    print(f"Saved: {out_dir / 'gain_pair_chart.png'}")
    print("\nPercentage improvement (Plus vs Vanilla):")
    for i, label in enumerate(labels):
        van_br = float(breach_vanilla[i])
        sm_br = float(breach_plus[i])
        van_im = float(impact_vanilla[i])
        sm_im = float(impact_plus[i])
        br_pct = pct_improvement(van_br, sm_br)
        im_pct = pct_improvement(van_im, sm_im)
        br_delta_pp = (sm_br - van_br) * 100.0
        print(
            f"{label}: breach_delta={br_delta_pp:+.2f} pp, "
            f"breach_improvement={br_pct:+.2f}% | impact_improvement={im_pct:+.2f}%"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Create paired vanilla-vs-plus bar charts from performance CSV.")
    p.add_argument("--perf-csv", default="transfer_attack_performance_all12.csv")
    p.add_argument("--out-dir", default="charts")
    args = p.parse_args()

    perf_csv = Path(args.perf_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    create_pair_charts(perf_csv=perf_csv, out_dir=out_dir)


if __name__ == "__main__":
    main()
