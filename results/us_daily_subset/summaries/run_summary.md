# CAMELS-US Run Summary

## Top Runs

|     RMSE |      NSE |    PBIAS |      KGE | split   | run_id                                                      | dynamic_group   | static_group   | loss_name   | frequency   |
|---------:|---------:|---------:|---------:|:--------|:------------------------------------------------------------|:----------------|:---------------|:------------|:------------|
| 0.953285 | 0.770877 |  9.08544 | 0.764978 | test    | us_daily_lumped_010515_core_hydro_skill_h365_b256           | era5_core       | static_base    | hydro_skill | daily       |
| 1.2139   | 0.635203 | 23.0335  | 0.567228 | test    | us_daily_semidistributed_010515_core_hydro_skill_h365_gb256 | era5_core       | static_base    | hydro_skill | daily       |

## Notes

- Models are trained on raw `mm/day` discharge targets while inputs are standardized from warmup+train data.
- Evaluation metrics clamp negative predictions to zero before scoring.
- Default dynamic inputs follow the reference lumped setup: ERA5 plus seasonal/time encodings.
- Outlet-only diagnostics use the outlet node for graph runs and the sole outlet for lumped runs.
- Detailed artifacts live under `runs/` and figures live under `reports/figures/`.
