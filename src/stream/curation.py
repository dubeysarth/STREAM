"""Curation-stage interfaces for STREAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters import run_legacy_stage
from .manifests import load_manifest_from_config


def run_graph_curation(config: dict[str, Any], config_path: str | None = None) -> list[list[str]]:
    manifest = load_manifest_from_config(config, config_path)
    return run_legacy_stage(manifest, "01_curate_graphs")


def run_nested_gauge_preparation(config: dict[str, Any], config_path: str | None = None) -> list[list[str]]:
    manifest = load_manifest_from_config(config, config_path)
    return run_legacy_stage(manifest, "03_prepare_nested_gauges")


@dataclass
class SnapToGridCurator:
    """Curate snapped gauge or outlet locations using the configured STREAM adapter."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "01_curate_graphs", "snap")


@dataclass
class MaskmapBuilder:
    """Build deterministic contributing-area maskmaps from the selected flow-direction product."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "01_curate_graphs", "maskmap")


@dataclass
class RiverGraphBuilder:
    """Construct a river-network DAG aligned with the selected grid and maskmaps."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "01_curate_graphs", "graph")
