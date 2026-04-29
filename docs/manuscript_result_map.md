# Manuscript to Result Map

This note links the main manuscript tables and figures to the repository files that support them.

## Manuscript source

- `paper/facesm-dc.tex`

## Main tables

- Attack-wise comparison table:
  - `results_summary/pairwise_split_summary.csv`
- Cross-model transferability table:
  - `results_summary/cross_model_transferability_analysis_with_attacker_victim_pairs.csv`
- RobFR external validation table:
  - `results_summary/robfr_external_validation_all_attacks.csv`
- Ablation table:
  - `results_summary/ablation_overall_summary.csv`

## Main figures

- Figure 1 overview flowchart:
  - `paper/fig1_facesm-flowchart.pdf`
- Figure 2 surrogate-wise and dataset-wise breach:
  - `paper/fig2a_surrogate_breach.pdf`
  - `paper/fig2b_dataset_breach.pdf`
  - numerical source: `results_summary/modelwise_summary.csv`, `results_summary/datasetwise_summary.csv`
- Figure 3 lambda sensitivity:
  - `paper/fig3_lambda_sweep_vertical.pdf`
  - numerical source: `results_summary/lambda_sweep_summary.csv`
- Figure 4 embedding trajectory:
  - `paper/fig4_embedding_trajectory.pdf`
- Figure 5 gradient alignment:
  - `paper/fig5a_sini_alignment.pdf`
  - `paper/fig5b_alignment_gain.pdf`

