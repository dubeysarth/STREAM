"""Inventory-stage interfaces for STREAM."""

from __future__ import annotations

from dataclasses import dataclass, field

from .adapters import run_legacy_stage
from .manifests import load_manifest_from_config


@dataclass
class ZarrInventoryBuilder:
    """Compile curated graph artifacts and features into a zarr-backed inventory."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "04_build_zarr_inventory")


@dataclass
class ClimateSummaryBuilder:
    """Aggregate climate summaries from daily inventory artifacts."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "05_build_climate_summaries")


@dataclass
class MonthlyAggregator:
    """Aggregate daily inventory artifacts into monthly representations."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "06_build_monthly_inventory")


@dataclass
class LumpedAggregator:
    """Aggregate graph-aware inventory artifacts into basin-level lumped datasets."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "07_build_lumped_inventory")
