# Ohio Fast-Track Daily/Monthly Lock Summary

## Variant Lock

- Daily lumped lock: `ohio_daily_lumped_hydro_skill_core` | loss `hydro_skill` | dynamic `era5_core` | NSE `0.734` | KGE `0.787`.
- Monthly lumped lock: `ohio_monthly_lumped_hydro_skill_core_h12` | loss `hydro_skill` | dynamic `era5_core` | NSE `0.388` | KGE `0.376`.

## Locked Graph Runs

- Daily semidistributed: `ohio_daily_semidistributed_core` | NSE `0.670` | KGE `0.693` | RMSE `1.222`.
- Monthly semidistributed: `ohio_monthly_semidistributed_core_h12` | NSE `0.257` | KGE `0.316` | RMSE `34.774`.
- Monthly distributed: `ohio_monthly_distributed_core_h12_min` | NSE `0.308` | KGE `0.303` | RMSE `34.366`.

## Notes

- Monthly tensors are derived from daily Ohio `.pt` records using variable-specific calendar-month aggregation: flux-like forcings and discharge are summed, state-like variables are averaged, and only month-of-year plus aggregated solar-insolation encodings are retained for monthly training.
- Daily lock uses the existing HUC05 daily tensor pipeline with raw mm/day targets, standardized inputs, and negative predictions clamped at evaluation.
- Human-use context tables use LISFLOOD cell sampling; reservoir counts use Global Dam Watch points assigned to the nearest Ohio graph node and then rolled up to basin or semi-distributed gauge units.
- Context-vs-skill figures are in `reports/figures/*context_vs_skill.png` and are intended for interpretation, not causal claims.
