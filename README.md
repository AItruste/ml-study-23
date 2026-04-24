# Anonymous Study Repository

This repository contains code and scripts used for experimental evaluation of transferability and robustness in machine learning models.

It provides implementations, evaluation pipelines, and reproducibility utilities used in the study, along with compact summaries of experimental results.

## Repository Layout

- `core/`
  - Main attack and evaluation code, including core implementation, evaluation utilities, and plotting scripts.
- `experiments/`
  - Scripts for benchmark reconstruction, ablation analysis, and parameter sensitivity studies.
- `scripts/`
  - Shell utilities used to run and organize experiments.
- `robfr_patch/`
  - Local modifications used for extended evaluation setups.
- `results_summary/`
  - Compact CSV summaries and chart images used for reporting results.
- `docs/`
  - Notes related to release and usage.

## What Is Included

- Core implementation used in the study  
- Experiment scripts for:
  - benchmark reconstruction  
  - ablation analysis  
  - parameter sensitivity studies  
  - extended evaluation setups  
- Summary CSV files used for tables and figures  
- Generated visualizations used in the manuscript  

## What Is Not Included

To keep the repository lightweight:

- large pretrained model weights  
- datasets  
- generated adversarial samples  
- full external evaluation frameworks  

## Reproducibility Notes

The original experiments were conducted in a larger workspace that included datasets, pretrained models, and additional evaluation components.

To fully reproduce results, users need to:
- provide appropriate datasets  
- place pretrained models in expected locations  
- adapt paths if needed  

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
