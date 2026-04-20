# Architecture Notes

STREAM is organized into five layers so the same repository can support
curation, inventory construction, benchmark training, and reporting without
relying on author-specific notebooks or private path conventions.

1. `src/stream/paths.py` and `src/stream/manifests.py`
   normalize clone-relative paths and dataset manifests so later stages stop
   depending on author-specific absolute paths.
2. `src/stream/curation.py` and `src/stream/features.py`
   expose the snapped-outlet, maskmap, DAG, nested-gauge, and feature-assembly
   stages through stable package entrypoints.
3. `src/stream/inventory.py`
   formalizes daily zarr, monthly rollup, climate-summary, and lumped
   aggregation stages.
4. `src/stream/adapters/` and `src/stream/benchmarks/`
   bridge region-specific curation/inventory sources and benchmark-specific
   training code for Ohio, CAMELS-US, and CAMELS-IND.
5. `src/stream/reporting.py`
   turns metrics and diagnostics into stable CSV outputs for validation bundles,
   ablation summaries, and reviewer-facing asset indexes.

The numbered `scripts/01` through `scripts/12` are thin wrappers over these
package layers. The `experiments/` folders keep frozen configs, verified run
recipes, and benchmark-specific test fixtures, while `src/stream` provides the
shared interfaces that the public repository exposes.
