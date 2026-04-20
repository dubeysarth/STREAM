# Ohio HUC05 Run Summary

## Top Runs

|     RMSE |      NSE |     PBIAS |      KGE | split   | run_id                                        | dynamic_group                | static_group   | loss_name     | frequency   |
|---------:|---------:|----------:|---------:|:--------|:----------------------------------------------|:-----------------------------|:---------------|:--------------|:------------|
|  1.0843  | 0.733993 | -0.593583 | 0.787385 | test    | ohio_daily_lumped_hydro_skill_core            | era5_core                    | static_base    | hydro_skill   | daily       |
|  1.14267 | 0.704583 |  5.18577  | 0.715402 | test    | lumped_hydro_skill                            | era5_full_selected           | static_base    | hydro_skill   | daily       |
|  1.16488 | 0.692986 |  5.90842  | 0.700307 | test    | lumped_rmse                                   | era5_full_selected           | static_base    | rmse          | daily       |
|  1.22169 | 0.670349 |  3.85897  | 0.693148 | test    | ohio_daily_semidistributed_core               | era5_core                    | static_base    | hydro_skill   | daily       |
|  1.28202 | 0.636991 |  4.53504  | 0.694914 | test    | semidistributed_batched_long_v2               | era5_full_selected           | static_base    | hydro_skill   | daily       |
| 31.5186  | 0.387552 | 10.0533   | 0.376367 | test    | ohio_monthly_lumped_hydro_skill_core_h12      | era5_core                    | static_base    | hydro_skill   | monthly     |
| 31.8206  | 0.375761 | 12.637    | 0.412565 | test    | ohio_monthly_lumped_rmse_full_h12             | era5_full_selected           | static_base    | rmse          | monthly     |
| 32.0451  | 0.366922 | 12.935    | 0.383281 | test    | ohio_monthly_lumped_hydro_skill_core_snow_h12 | era5_core_plus_snow          | static_base    | hydro_skill   | monthly     |
| 32.351   | 0.354778 | 16.2339   | 0.379602 | test    | ohio_monthly_lumped_hydro_skill_full_h12      | era5_full_selected           | static_base    | hydro_skill   | monthly     |
| 32.6065  | 0.344546 | 17.9868   | 0.363876 | test    | ohio_monthly_lumped_hydro_balance_full_h12    | era5_full_selected           | static_base    | hydro_balance | monthly     |
| 32.6167  | 0.344134 | 15.8906   | 0.381093 | test    | ohio_monthly_lumped_hydro_skill_core_wb_h12   | era5_core_plus_water_balance | static_base    | hydro_skill   | monthly     |
| 34.3661  | 0.307918 |  7.08846  | 0.303165 | test    | ohio_monthly_distributed_core_h12_min         | era5_core                    | static_base    | hydro_skill   | monthly     |
| 34.7737  | 0.256679 | 23.1364   | 0.31578  | test    | ohio_monthly_semidistributed_core_h12         | era5_core                    | static_base    | hydro_skill   | monthly     |

## Notes

- Models are trained on raw `mm/day` discharge targets while inputs are standardized from warmup+train data.
- Evaluation metrics clamp negative predictions to zero before scoring.
- Default dynamic inputs follow the reference lumped setup: ERA5 plus seasonal/time encodings.
- Outlet-only diagnostics use the outlet node for graph runs and the sole outlet for lumped runs.
- Detailed artifacts live under `runs/` and figures live under `reports/figures/`.
