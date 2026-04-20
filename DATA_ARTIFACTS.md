# Data Artifacts

This note maps repository stages to the data and result artifacts used by the
current STREAM package.

## Artifact map

| Repo stage | Artifact type | Current local reference | Public role |
|---|---|---|---|
| Snap-to-grid + maskmap diagnostics | gauge snap tables, maskmaps, nesting metadata | `reference_materials/STREAM`, `reference_materials/zenodo/*/nested_gauges` | curation provenance |
| Lumped inventory | outlet-level zarr / tensor bundle | `reference_materials/zenodo/03min_GloFAS_CAMELS-US_Ohio/lumped_inventory` | lumped training substrate |
| Distributed inventory | node-level zarr / tensor bundle + graph files | `reference_materials/zenodo/03min_GloFAS_CAMELS-US_Ohio/inventory`, `graph_files` | semi/distributed substrate |
| Scale examples | Mahanadi multi-scale bundle | `reference_materials/zenodo/Mahanadi_at_different_scales` | scale-transition illustration |
| Lumped surrogate experiments | scripts, assets, checkpoints, result tables | `reference_materials/zenodo/lumped_surrogate_datasets` | prior surrogate baseline context |
| Distributed surrogate example | monthly checkpoint + run script | `reference_materials/zenodo/surrogate_distributed` | graph-tier example |
| Ohio daily verified runs | metric tables, figure bundles, context diagnostics | `results/ohio_daily/` | primary rebuttal-facing evidence |
| US daily subset runs | metric tables, figure bundles, context diagnostics | `results/us_daily_subset/` | broader daily validation |

## Repo-to-artifact relationship

- `src/stream/` defines the public curation and reporting interfaces.
- `experiments/ohio_daily/` shows those interfaces exercised on the verified
  Ohio pilot.
- `experiments/us_daily_subset/` shows the same benchmark logic on a broader
  daily subset.
- `results/` collects the tables and figures that are easiest to cite from the
  paper or project page.

## Licensing and redistribution

STREAM does not assert one new blanket license over redistributed benchmark
bundles. Each dataset family inherits the governing source-specific license or
terms from the underlying products, and the repository therefore documents
provenance and redistribution conditions per family rather than collapsing them
into a single statement.

See [LICENSES.md](LICENSES.md) for the per-family summary.

## What should be considered first-class artifacts

STREAM treats these as first-class products rather than side effects:

1. snapped outlets and their quality diagnostics,
2. maskmaps and nesting metadata,
3. graph files and node assignments,
4. aligned forcing / target tensors,
5. run manifests, metric tables, predictions, and figure bundles.

## Ohio verification anchor

The Ohio daily benchmark remains the first polished public example because it
demonstrates:

- reproducible tensor creation,
- lumped and semi-distributed training,
- ablation-ready experiment interfaces,
- context-vs-skill diagnostics that support the failure-analysis narrative.

## US daily subset role

The CAMELS-US subset adds breadth without changing the experiment design. It is
best used as a follow-up confirmation that the same forcing bundle, loss, and
history lock still produce interpretable daily results on a broader basin family.
