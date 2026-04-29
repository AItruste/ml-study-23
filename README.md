# Anonymous Study Repository

Anonymous review repository for the FaceSM study on transferable adversarial attacks against face verification systems.

This repository contains the core implementation, experiment scripts, verified summary CSV files, and paper figures used to support the manuscript results. It is designed as a lightweight review package rather than a full training workspace.

## Repository Layout

- `core/`
  - Main attack and evaluation code, including the attack implementation, rescoring utilities, plotting helpers, and threshold configuration.
- `experiments/`
  - Scripts for rebuilding benchmark summaries, ablation analysis, lambda sensitivity analysis, and RobFR validation.
- `scripts/`
  - Shell wrappers used during experiment refresh and result preparation.
- `robfr_patch/`
  - Local RobFR modifications used for the external validation experiments.
- `results_summary/`
  - Compact CSV summaries used for manuscript tables and figures.
- `charts/`
  - Rendered figure files derived from the verified summaries.
- `docs/`
  - Release and usage notes.

## What Is Included

- Core implementation used in the study
- Experiment scripts for:
  - main benchmark summary rebuilding
  - ablation analysis
  - source-separation sensitivity analysis
  - RobFR external validation including LGC extension
- Verified summary CSV files used for manuscript tables and charts
- Generated chart files used in the manuscript

## What Is Not Included

To keep the repository lightweight and reviewer-safe, the following are intentionally excluded:

- large pretrained model weights
- raw benchmark datasets
- generated adversarial image dumps
- the complete external evaluation workspace

## Reproducibility Notes

The original experiments were run in a larger workspace that included public face datasets, pretrained weights, and local evaluation checkouts. To reproduce the full pipeline, users need to:

- obtain the required datasets from their official sources
- place pretrained weights in the expected locations
- provide the benchmark pair definitions and any external framework dependencies

Raw LFW, CelebA, and VGGFace2 images are not redistributed here.

## Key Result Files

The main manuscript numbers are backed by the following summary files:

- `results_summary/pairwise_split_summary.csv`
- `results_summary/modelwise_summary.csv`
- `results_summary/datasetwise_summary.csv`
- `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`
- `results_summary/ablation_overall_summary.csv`
- `results_summary/lambda_sweep_summary.csv`
- `results_summary/robfr_external_validation_all_attacks.csv`

## Notes on Evaluation Scope

Two different aggregation scopes are present in the repository:

- `results_summary/modelwise_summary.csv` reports surrogate-wise averages on the main benchmark common victim pool.
- `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv` reports broader attacker-victim transfer pairs and excludes the `Facenet512 -> Facenet` same-family case.

Because these scopes differ, some aggregate numbers for the same surrogate can differ between sections of the manuscript.

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested Reviewer Entry Points

- Main benchmark summaries: `results_summary/`
- Paper figure files: `charts/`
- Summary rebuild script: `experiments/build_paper_results_lambda20_limit1000.py`
- RobFR extension script: `experiments/run_robfr_lgc_extension.py`
