# Data Details

## Core dynamic forcing choice

The strongest verified daily results currently use the `era5_core` forcing set.
That bundle contains:

- total precipitation,
- potential evaporation,
- 2m temperature,
- 2m dewpoint temperature,
- surface pressure,
- 10m zonal and meridional wind,
- net solar radiation,
- net thermal radiation.

These variables were chosen because they give a clean, compact forcing package
that remains competitive across both lumped and semi-distributed settings.

## Target definition

The benchmark target is gridded or outlet-aligned `dynamic_GloFAS_discharge_mm`.
This makes cross-scale benchmarking reproducible, but it should be interpreted as
surrogate discharge evaluation rather than direct operational streamflow
forecasting.

## Static and contextual layers

The static bundle includes:

- basin or node attributes already stored in the inventories,
- transformed latitude / longitude encodings,
- LISFLOOD parameter-map summaries,
- human-use indicators such as irrigation, sealing, water fraction,
  groundwater-use fraction, and sectoral demand fields,
- reservoir / lake mask information,
- Global Dam Watch derived reservoir counts in the Ohio benchmark.

## Semi-distributed definition

Semi-distributed graphs are built from nested gauges or subcatchment assignments:

- if nested gauges exist, graph nodes correspond to gauge-subbasins,
- otherwise the basin degenerates to a valid one-node graph,
- node features are area-weighted aggregates of distributed-node features.

## Time splits

### Ohio pilot

- warm-up: `1998-01-01` to `1998-12-31`
- train: `1999-01-01` to `2009-12-31`
- validation: `2010-01-01` to `2014-12-31`
- test: `2015-01-01` to `2019-12-31`

### CAMELS-US daily subset

The same split specification is used for the daily subset experiments so the
Ohio and US results remain directly comparable at the experiment-design level.

## Frequency choices

- Daily is the primary benchmark frequency for rebuttal-facing evidence.
- Monthly tensors and runs are included in the Ohio package as implementation and
  diagnostic support, but the strongest current comparative claims rely on daily
  runs.
