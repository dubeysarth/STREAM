"""CAMELS-IND legacy adapter."""

from __future__ import annotations

from pathlib import Path

from .base import notebook_cmd, python_cmd


def build_components(repo_root: Path) -> dict[str, object]:
    return {
        "adapter": "camels_ind",
        "paths": {"repo_root": str(repo_root)},
        "config_module": None,
        "feature_registry_cls": None,
        "split_cls": None,
        "training_config_cls": None,
        "runtime_cls": None,
        "runtime_parser": None,
        "train_lumped_run": None,
        "train_graph_run": None,
        "write_registry": None,
        "source_manifest_cls": None,
    }


def legacy_commands(repo_root: Path, stage_name: str, target: str | None = None) -> list[list[str]]:
    reference_root = repo_root.parents[1] / "reference_materials" / "STREAM"
    stage_map = {
        "01_curate_graphs": notebook_cmd(reference_root / "01_Creating_Graphs" / "03min_CAMELS-IND.ipynb"),
        "02_attach_node_features": notebook_cmd(reference_root / "02_Node_Features" / "03min_CAMELS-IND.ipynb"),
        "03_prepare_nested_gauges": notebook_cmd(reference_root / "03_Preparation" / "CAMELS-IND_NestedGauges.ipynb"),
        "04_build_zarr_inventory": python_cmd(reference_root / "04_Zarr_Inventory" / "CAMELS-IND.py"),
        "05_build_climate_summaries": notebook_cmd(reference_root / "05_Climate_Summaries" / "CAMELS-IND.ipynb"),
        "06_build_monthly_inventory": python_cmd(reference_root / "06_Monthly_Inventory" / "CAMELS-IND.py"),
        "07_build_lumped_inventory": notebook_cmd(reference_root / "07_Lumped_Inventory" / "CAMELS-IND.ipynb"),
    }
    if stage_name not in stage_map:
        raise ValueError(f"Unsupported CAMELS-IND legacy stage: {stage_name}")
    return [stage_map[stage_name]]
