# Zenodo Artifact Registry

This registry summarizes the bundles already present under `reference_materials/zenodo/`.

| Local bundle | What it contains | Supports which part of the paper/package | Redistribution basis | Notes |
|---|---|---|---|---|
| `03min_GloFAS_CAMELS-US_Ohio` | Ohio lumped/distributed inventories, graph files, nested gauges, scaling assets | cross-tier benchmark example, daily Ohio verification | Inherits source-specific license / terms from upstream products; provenance note required | primary current verified benchmark anchor |
| `03min_GloFAS_CAMELS-IND_Mahanadi` | Mahanadi lumped/distributed inventories, graph files, nested gauges | scale illustration, zero-shot / management-context examples | Inherits source-specific license / terms from upstream products; provenance note required | supports the India narrative |
| `Mahanadi_at_different_scales` | multi-resolution basin products, GIS, graphs, notebooks | scale-triad explanation, snapping/coarsening discussion | Inherits source-specific license / terms from upstream products and derivative preparation notes | useful for project-page and appendix visuals |
| `lumped_surrogate_datasets` | lumped experiment scripts, assets, results | historical lumped benchmark lineage and reference implementation | Derived benchmark/code artifacts; upstream data terms remain in force where applicable | useful for methodology comparison |
| `surrogate_distributed` | distributed run script and pretrained monthly checkpoint | distributed benchmark example | Derived code/checkpoint artifact with upstream data provenance preserved | useful as artifact transparency rather than a direct reviewer-facing claim |
| `Indian_River_Basins` | Indian basin extraction notebooks and basin products | basin-boundary and topology provenance | Inherits source-specific license / terms from upstream products; provenance note required | background / reproducibility support |

## Caution

The redistribution principle should remain explicit in the public-facing release:
STREAM does not collapse all source datasets into one new blanket license. Each
bundle inherits the governing source-specific terms, and companion provenance /
license notes should travel with the redistributed artifact.
