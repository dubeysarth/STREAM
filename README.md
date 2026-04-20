# STREAM

STREAM is a reproducible hydrologic benchmarking and curation codebase for
building comparable lumped, semi-distributed, and distributed river-learning
experiments from the same geospatial substrate. The repository is organized so
that the curation layer, experiment layer, and reporting layer are all
inspectable rather than being hidden inside one-off notebooks or private
preprocessing scripts.

## What this repository contains

- `src/stream/`: the shared STREAM package for snapping, maskmaps, graph
  construction, inventories, baselines, and reporting.
- `scripts/`: the repo-level sequential entrypoints for curation through
  rebuttal-asset generation.
- `experiments/ohio_daily/`: the full Ohio pilot code used to build tensors,
  train lumped / semi-distributed / distributed models, generate figures, and
  summarize results.
- `experiments/us_daily_subset/`: the CAMELS-US daily subset code path that
  mirrors the Ohio workflow on a broader set of basins.
- `results/`: metric tables, figure bundles, and run summaries for the verified
  Ohio daily benchmark and the US daily subset follow-up.
- `docs/`: architecture, reproducibility, data-artifact, and experiment-walk
  through notes.
- `tests/`: repo-level smoke tests for the shared `stream` package plus
  experiment-specific tests under each benchmark folder.

## Public position

STREAM should be read as a **benchmark, curation, and diagnosis framework**.
The main contribution is not a bespoke new sequence or graph architecture. The
contribution is that the same basin can be turned into:

1. a lumped outlet example,
2. a semi-distributed nested-gauge graph,
3. a distributed river-network graph,

while reusing the same snapped outlets, maskmaps, flow-direction topology,
forcing alignment, target definition, and human-context overlays.

## Quickstart

STREAM is intended for clone-and-run usage rather than installation as a Python
package.

```bash
git clone <REPO_URL>
cd STREAM
conda env create -f environment.yml
conda activate stream
export PYTHONPATH=$PWD/src
```

For the benchmark workspaces:

```bash
export PYTHONPATH=$PWD/experiments/ohio_daily/src:$PWD/experiments/us_daily_subset/src:$PYTHONPATH
python experiments/ohio_daily/scripts/00_probe_environment.py
python experiments/us_daily_subset/scripts/01_write_registry.py
```

## Repository layout

```text
STREAM/
├── configs/                     # public template configs
├── docs/                        # architecture + reproducibility notes
├── environment.yml
├── experiments/
│   ├── ohio_daily/              # verified Ohio benchmark package
│   └── us_daily_subset/         # broader daily validation package
├── results/
│   ├── ohio_daily/
│   └── us_daily_subset/
├── scripts/                     # repo-level curation / reporting entrypoints
├── src/stream/                  # shared STREAM package
└── tests/
```

## Verified benchmark anchors

### Ohio daily cross-tier benchmark

| Setting | RMSE | NSE | PBIAS | KGE |
|---|---:|---:|---:|---:|
| Lumped | 1.084 | 0.734 | -0.594 | 0.787 |
| Semi-distributed | 1.222 | 0.670 | 3.859 | 0.693 |

### CAMELS-US daily subset follow-up (`HUC 01 + 05 + 15`)

| Setting | RMSE | NSE | PBIAS | KGE |
|---|---:|---:|---:|---:|
| Lumped | 0.953 | 0.771 | 9.085 | 0.765 |
| Semi-distributed | 1.214 | 0.635 | 23.033 | 0.567 |

These are intentionally presented as benchmark evidence rather than deployment
claims. The value of the semi-distributed tier is that it preserves
topology-aware within-basin structure and context overlays even when outlet
metrics do not exceed the lumped tier.

## Where to start

- Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the execution model.
- Read [DATA_ARTIFACTS.md](DATA_ARTIFACTS.md) for the data-product mapping.
- Read [RESULTS_STATUS.md](RESULTS_STATUS.md) for what is verified versus what extends naturally from the same repo interfaces.
- Read [docs/DATA_DETAILS.md](docs/DATA_DETAILS.md) for feature / target choices.
- Read [results/README.md](results/README.md) for metric tables and figure paths.

## Intended-use boundary

STREAM is for controlled benchmarking and diagnosis.

- GloFAS-based surrogate discharge is useful for cross-scale reproducible
  evaluation, but it is not identical to operational discharge forecasting.
- Regulation and human-use overlays are included to support interpretation of
  failure structure, not to establish a single causal mechanism.
- Regulated and strongly human-influenced basins are scientifically important
  precisely because they require extra caution in how results are read.

The codes to generate the data inventory are included in this GitHub Repo.

Sample of curated dataset, along with the assets accompanying experiments and analysis are available at Zenodo ([10.5281/zenodo.18559271](https://doi.org/10.5281/zenodo.18559270))
- *Mahanadi_at_different_scales.tar.gz*: Codes and assets supporting the creation of River Network Graph for Mahanadi River Basin at different resolutions.
- *Indian_River_Basins.tar.gz*: River Network DAG of Indian River Basins at different resolutions.
- *03arcmins_CAMELS-US_Ohio.tar.gz*: Catchment data inventory along with supporting attributes and nesting information for Ohio Eco-Region of CAMELS-US.
- *03arcmins_CAMELS-IND_Mahanadi.tar.gz*: Catchment data inventory along with supporting attributes and nesting information for Mahanadi Basin of CAMELS-IND.
- *lumped_surrogate_datasets.tar.gz*: Tensor Datasets, trained weights, scripts and metrics for lumped surrogate.
- *surrogate_distributed.tar.gz*: trained weights and script for distributed surrogate.
