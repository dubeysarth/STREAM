# Experiment Workspaces

This repository includes two concrete benchmark workspaces in addition to the
top-level `stream` package.

## `ohio_daily/`

The Ohio pilot is the verified end-to-end benchmark package. It contains:

- source-audit and registry scripts,
- tensor builders for lumped / semi-distributed / distributed regimes,
- human-use and reservoir-context extraction,
- lumped, semi-distributed, and distributed training scripts,
- ablation entrypoints,
- graph-batch caching,
- evaluation, plotting, and summary writers,
- tests for metrics, splits, models, and graph batching.

## `us_daily_subset/`

The US daily subset workspace mirrors the Ohio design on a broader set of
basins. The current verified subset uses `HUC 01 + 05 + 15` and keeps the Ohio
daily lock:

- `era5_core`,
- `hydro_skill`,
- `history_length=365`,
- lumped `batch_size=256`,
- semi-distributed `graph_batch_size=256`.

## Why keep these as explicit workspaces?

The workspaces make it obvious which code paths were used for verified runs. The
top-level `stream` package provides the shared repo interface; the workspace
folders preserve concrete benchmark packages with actual configs, metrics,
plots, and run manifests.
