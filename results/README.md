# Results, Metric Tables, and Figures

## Included result bundles

- `ohio_daily/`: verified Ohio pilot tables and selected figure bundles.
- `us_daily_subset/`: CAMELS-US daily subset follow-up tables and selected
  figure bundles.

## Quick metric summary

### Ohio daily

| Setting | RMSE | NSE | PBIAS | KGE |
|---|---:|---:|---:|---:|
| Lumped | 1.084 | 0.734 | -0.594 | 0.787 |
| Semi-distributed | 1.222 | 0.670 | 3.859 | 0.693 |

### US daily subset

| Setting | RMSE | NSE | PBIAS | KGE |
|---|---:|---:|---:|---:|
| Lumped | 0.953 | 0.771 | 9.085 | 0.765 |
| Semi-distributed | 1.214 | 0.635 | 23.033 | 0.567 |

## Figure bundles

### Ohio daily figures

- `figures/ohio_daily_lumped_hydro_skill_core_condensed_summary.png`
- `figures/ohio_daily_lumped_hydro_skill_core_standard_skill.png`
- `figures/ohio_daily_lumped_hydro_skill_core_context_vs_skill.png`
- `figures/ohio_daily_semidistributed_core_condensed_summary.png`
- `figures/ohio_daily_semidistributed_core_standard_skill.png`
- `figures/ohio_daily_semidistributed_core_context_vs_skill.png`

### US daily subset figures

- `figures/us_daily_lumped_010515_core_hydro_skill_h365_b256_condensed_summary.png`
- `figures/us_daily_lumped_010515_core_hydro_skill_h365_b256_standard_skill.png`
- `figures/us_daily_lumped_010515_core_hydro_skill_h365_b256_context_vs_skill.png`
- `figures/us_daily_semidistributed_010515_core_hydro_skill_h365_gb256_condensed_summary.png`
- `figures/us_daily_semidistributed_010515_core_hydro_skill_h365_gb256_standard_skill.png`
- `figures/us_daily_semidistributed_010515_core_hydro_skill_h365_gb256_context_vs_skill.png`

## Notes on interpretation

- The lumped tier is the strongest outlet benchmark.
- The semi-distributed tier is valuable because it preserves within-basin
  topology and allows context overlays and hotspot localization, not because it
  must dominate the lumped tier on every outlet metric.
- Negative or extreme basin-level NSE values are clamped to a readable plotting
  range in the standard skill figures so that the overall basin distribution
  remains interpretable.
