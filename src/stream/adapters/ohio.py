"""Ohio benchmark adapter."""

from __future__ import annotations

from pathlib import Path

from .base import notebook_cmd


def build_components(repo_root: Path) -> dict[str, object]:
    from ..benchmarks.ohio import config, features, manifests, registry, runtime, splits, training
    from ..benchmarks.ohio.paths import DevpOhioPaths

    paths = DevpOhioPaths.from_root(repo_root)
    return {
        "adapter": "ohio",
        "paths": paths,
        "config_module": config,
        "feature_registry_cls": features.FeatureRegistry,
        "split_cls": splits.SplitSpec,
        "training_config_cls": training.TrainingConfig,
        "runtime_cls": runtime.RunContext,
        "runtime_parser": runtime.build_parser,
        "train_lumped_run": training.train_lumped_run,
        "train_graph_run": training.train_graph_run,
        "write_registry": registry.write_huc05_registry,
        "source_manifest_cls": manifests.SourceManifest,
    }


def legacy_commands(repo_root: Path, stage_name: str, target: str | None = None) -> list[list[str]]:
    reference_root = repo_root.parents[1] / "reference_materials" / "STREAM"
    stage_map = {
        "01_curate_graphs": notebook_cmd(reference_root / "01_Creating_Graphs" / "03min_CAMELS-US.ipynb"),
        "02_attach_node_features": notebook_cmd(reference_root / "02_Node_Features" / "03min_CAMELS-US.ipynb"),
        "03_prepare_nested_gauges": notebook_cmd(reference_root / "03_Preparation" / "CAMELS-US_NestedGauges.ipynb"),
        "04_build_zarr_inventory": [str((reference_root / "04_Zarr_Inventory" / "CAMELS-US.py").resolve())],
        "05_build_climate_summaries": notebook_cmd(reference_root / "05_Climate_Summaries" / "CAMELS-US.ipynb"),
        "06_build_monthly_inventory": notebook_cmd(reference_root / "06_Monthly_Inventory" / "CAMELS-US.ipynb"),
        "07_build_lumped_inventory": notebook_cmd(reference_root / "07_Lumped_Inventory" / "CAMELS-US.ipynb"),
    }
    if stage_name == "04_build_zarr_inventory":
        from .base import python_cmd

        return [python_cmd(Path(stage_map[stage_name][0]))]
    if stage_name not in stage_map:
        raise ValueError(f"Unsupported Ohio legacy stage: {stage_name}")
    return [stage_map[stage_name]]
