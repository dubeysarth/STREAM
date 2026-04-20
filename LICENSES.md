# Dataset Licensing and Redistribution Notes

STREAM does not place all redistributed benchmark bundles under one new blanket
license. Each dataset family inherits the governing license or access terms of
the upstream products used to build it. The purpose of this note is to document
that inheritance and keep the redistribution basis explicit.

| Dataset family | Source products | Redistribution basis | Companion file / note |
|---|---|---|---|
| `03min_GloFAS_CAMELS-US_Ohio` | GloFAS-derived discharge fields, ERA5 forcings, flow-direction/topology inputs, contextual layers | Inherits source-specific license / terms from upstream products; no blanket repo license asserted | `github/DATA_ARTIFACTS.md`; `zenodo/artifact_registry.md` |
| `03min_GloFAS_CAMELS-IND_Mahanadi` | GloFAS-derived discharge fields, ERA5 forcings, flow-direction/topology inputs, contextual layers | Inherits source-specific license / terms from upstream products; no blanket repo license asserted | `github/DATA_ARTIFACTS.md`; `zenodo/artifact_registry.md` |
| `Mahanadi_at_different_scales` | Multi-resolution basin products, GIS layers, graphs, notebooks | Inherits source-specific license / terms from upstream products and derivative preparation notes | `zenodo/artifact_registry.md` |
| `lumped_surrogate_datasets` | Experiment scripts, result assets, derived benchmark tables | Distributed as derived benchmark artifacts plus source-code assets; upstream data terms remain in force where applicable | `github/results/README.md`; `zenodo/artifact_registry.md` |
| `surrogate_distributed` | Distributed run scripts, pretrained checkpoint, derived outputs | Distributed as code/checkpoint artifact with upstream data provenance preserved | `zenodo/artifact_registry.md` |
| `Indian_River_Basins` | Basin extraction notebooks and basin products | Inherits source-specific license / terms from upstream basin and hydrography inputs | `zenodo/artifact_registry.md` |

Where an upstream provider imposes access, attribution, or redistribution
conditions, those conditions continue to govern the derived benchmark bundle.
