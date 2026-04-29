# FaceSM Anonymous Review Repository

This repository is the anonymous review package for the FaceSM study on transferable adversarial attacks against face verification systems. It is organized to make the paper easy to audit: the manuscript source is included, the figure assets used in the submission are present, and the main reported numbers are backed by compact CSV summaries.

## Start Here

If you are reviewing the paper and want the quickest path through the materials:

1. Open `paper/facesm-dc.tex` for the manuscript source.
2. Check `paper/` for the exact figure PDFs referenced in the submission.
3. Check `results_summary/` for the verified CSV files behind the main tables and figures.
4. Use `docs/manuscript_result_map.md` to trace manuscript claims to repository files.

## Repository Layout

- `paper/`
  - Anonymous LaTeX submission source and exact figure PDFs used in the manuscript.
- `results_summary/`
  - Compact CSV files backing the paper tables and figure values.
- `core/`
  - Main attack and evaluation code, plus utilities used during result consolidation.
- `experiments/`
  - Scripts and notebooks used to rebuild summary tables, ablations, sensitivity studies, and RobFR validation.
- `robfr_patch/`
  - Local RobFR-side modifications used for the external validation experiments.
- `docs/`
  - Data-availability notes and a manuscript-to-results map.

## What Is Included

- anonymous manuscript source
- exact manuscript figure PDFs
- verified result summary CSV files
- core implementation and experiment scripts

## What Is Not Included

To keep the review package lightweight and safe to share, the repository intentionally excludes:

- raw face dataset images
- large pretrained model weights
- full generated adversarial image dumps
- the larger private workspace used during experiment execution

## Main Result Files

The primary manuscript numbers are backed by the following summary files:

- `results_summary/pairwise_split_summary.csv`
- `results_summary/modelwise_summary.csv`
- `results_summary/datasetwise_summary.csv`
- `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`
- `results_summary/ablation_overall_summary.csv`
- `results_summary/lambda_sweep_summary.csv`
- `results_summary/robfr_external_validation_all_attacks.csv`

## Scope Note for Aggregate Numbers

Two different aggregation scopes appear in the repository:

- `results_summary/modelwise_summary.csv` reports surrogate-wise averages on the main benchmark common victim pool.
- `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv` reports a broader cross-model protocol and excludes the `Facenet512 -> Facenet` same-family case.

Because the victim pools differ, some aggregate numbers for the same surrogate can differ between manuscript sections. This is expected and reflects the evaluation scope rather than a data inconsistency.

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested Reviewer Entry Points

- manuscript source: `paper/facesm-dc.tex`
- figure assets: `paper/`
- verified summaries: `results_summary/`
- summary rebuild script: `experiments/build_paper_results_lambda20_limit1000.py`
- RobFR extension script: `experiments/run_robfr_lgc_extension.py`
