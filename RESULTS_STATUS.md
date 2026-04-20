# Results Status

This document separates what is already verified from what is already
implemented in the repository and what naturally extends from the same shared
interfaces.

## Verified now

### Ohio daily benchmark

| Setting | RMSE | NSE | PBIAS | KGE | Source |
|---|---:|---:|---:|---:|---|
| Lumped | 1.084 | 0.734 | -0.594 | 0.787 | `results/ohio_daily/run_summary_table.csv` |
| Semi-distributed | 1.222 | 0.670 | 3.859 | 0.693 | `results/ohio_daily/run_summary_table.csv` |

This is the cleanest current same-pipeline cross-tier benchmark.

### CAMELS-US daily subset (`HUC 01 + 05 + 15`)

| Setting | RMSE | NSE | PBIAS | KGE | Source |
|---|---:|---:|---:|---:|---|
| Lumped | 0.953 | 0.771 | 9.085 | 0.765 | `results/us_daily_subset/run_summary_table.csv` |
| Semi-distributed | 1.214 | 0.635 | 23.033 | 0.567 | `results/us_daily_subset/run_summary_table.csv` |

This provides a broader daily follow-up on top of the Ohio lock.

### Ohio ablation signal

- Daily `era5_core + hydro_skill` is the strongest verified Ohio setting for
  both lumped and semi-distributed runs.
- Monthly variants are implemented and summarized in `results/ohio_daily/`, but
  the strongest rebuttal-facing evidence remains daily.
- Context-vs-skill diagnostics were generated for both Ohio and the US subset,
  including reservoir-count and human-use overlays.

## Implemented and included in this repo

- Refactored curation interfaces under `src/stream/`.
- Full Ohio experiment code and tests under `experiments/ohio_daily/`.
- Full CAMELS-US daily subset code under `experiments/us_daily_subset/`.
- Figure-generation and report-writing scripts for both workspaces.
- Metric tables and selected figure bundles under `results/`.

## Natural next extensions

- larger-scale CAMELS-US runs beyond the current subset,
- CAMELS-IND expansion using the same experiment interfaces,
- additional distributed baselines and extended topology controls.

## Intended public narrative

1. STREAM standardizes cross-scale hydrologic curation.
2. The Ohio daily package is the most polished verified cross-tier benchmark.
3. The CAMELS-US daily subset confirms the same code path on a broader basin
   family.
4. Larger CAMELS-US / CAMELS-IND runs extend the same interfaces rather than
   redefining the pipeline.
