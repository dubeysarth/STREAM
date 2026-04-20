# Pipeline Walkthrough

The stage scripts mirror the current STREAM pipeline and the rebuttal-specific extensions:

1. `01_curate_graphs.py`
2. `02_attach_node_features.py`
3. `03_prepare_nested_gauges.py`
4. `04_build_zarr_inventory.py`
5. `05_build_climate_summaries.py`
6. `06_build_monthly_inventory.py`
7. `07_build_lumped_inventory.py`
8. `08_run_lumped_baseline.py`
9. `09_run_distributed_baseline.py`
10. `10_run_ablation_suite.py`
11. `11_run_validation_bundle.py`
12. `12_build_rebuttal_assets.py`

Every script supports `--dry-run` so the stage order can be inspected before any implementation is added.
