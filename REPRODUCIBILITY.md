# Reproducibility

## Execution model

The public release uses a clone-and-run workflow rather than package
installation.

```bash
git clone <REPO_URL>
cd STREAM
conda env create -f environment.yml
conda activate stream
```

## Reproducibility principles

1. The same basin should be reproducible across lumped, semi-distributed, and
   distributed settings.
2. Snapping, maskmaps, graph extraction, forcings, targets, and human-context
   layers must be saved as inspectable artifacts.
3. Every training run should emit a resolved config, plain-text log, metric
   table, prediction artifact, figure bundle, and run manifest.
4. Report-generation must be possible from saved predictions without retraining.

## Repo-level curation interface

The top-level `scripts/` folder exposes the refactored curation-to-reporting
pipeline:

- `01_curate_graphs.py`
- `02_attach_node_features.py`
- `03_prepare_nested_gauges.py`
- `04_build_zarr_inventory.py`
- `05_build_climate_summaries.py`
- `06_build_monthly_inventory.py`
- `07_build_lumped_inventory.py`
- `08_run_lumped_baseline.py`
- `09_run_distributed_baseline.py`
- `10_run_ablation_suite.py`
- `11_run_validation_bundle.py`
- `12_build_rebuttal_assets.py`

These define the public contract. The benchmark folders under `experiments/`
show the concrete implementations used to exercise this contract on Ohio and the
US daily subset.

## Benchmark workspaces

### Ohio daily

Use the Ohio workspace to reproduce the verified cross-tier pilot:

```bash
python experiments/ohio_daily/scripts/00_probe_environment.py
python experiments/ohio_daily/scripts/01_audit_sources.py
python experiments/ohio_daily/scripts/04_build_lumped_tensors.py
python experiments/ohio_daily/scripts/05_build_semidistributed_tensors.py
python experiments/ohio_daily/scripts/08_train_lumped.py --run-name ohio_daily_lumped_hydro_skill_core
python experiments/ohio_daily/scripts/11_train_semidistributed.py --run-name ohio_daily_semidistributed_core
python experiments/ohio_daily/scripts/13_evaluate_all.py
python experiments/ohio_daily/scripts/14_make_visuals.py --run-id ohio_daily_lumped_hydro_skill_core
python experiments/ohio_daily/scripts/14_make_visuals.py --run-id ohio_daily_semidistributed_core
```

### CAMELS-US daily subset

Use the US workspace for the broader daily follow-up:

```bash
python experiments/us_daily_subset/scripts/01_write_registry.py
python experiments/us_daily_subset/scripts/02_extract_human_use_features.py
python experiments/us_daily_subset/scripts/03_build_tensors.py --regimes lumped distributed
python experiments/us_daily_subset/scripts/03_build_tensors.py --regimes semidistributed --skip-existing
python experiments/us_daily_subset/scripts/04_fit_scalers.py --hucs 01 05 15
python experiments/us_daily_subset/scripts/08_train_lumped.py --run-name us_daily_lumped_010515_core_hydro_skill_h365_b256 --hucs 01 05 15
python experiments/us_daily_subset/scripts/11_train_semidistributed.py --run-name us_daily_semidistributed_010515_core_hydro_skill_h365_gb256 --hucs 01 05 15
python experiments/us_daily_subset/scripts/13_evaluate_all.py
python experiments/us_daily_subset/scripts/14_make_visuals.py --run-id us_daily_lumped_010515_core_hydro_skill_h365_b256
python experiments/us_daily_subset/scripts/14_make_visuals.py --run-id us_daily_semidistributed_010515_core_hydro_skill_h365_gb256
```

## Minimum metadata to save per run

- `run_id`
- `config_hash`
- `seed`
- `split_definition`
- `dynamic_group`
- `static_group`
- `loss_name`
- `history_length`
- `frequency`
- `device`
- `data_root`
- `artifact_root`

## Expected outputs

Each successful run should create:

- `runs/logs/<run_id>.log`
- `runs/manifests/<run_id>.json`
- `runs/metrics/<run_id>_summary.csv`
- `runs/metrics/<run_id>_per_basin.csv`
- `runs/predictions/<run_id>.nc`
- `runs/histories/<run_id>.csv`
- `reports/figures/<run_id>_*.png`

## Scientific reporting expectations

- Explain where GloFAS, flow direction, and contextual human-use layers enter
  the pipeline.
- Make lumped / semi-distributed / distributed data contracts explicit.
- Document the graph model at the level of layer count, hidden size, graph
  operator, and training defaults.
- Distinguish benchmark diagnostics from deployment claims.
