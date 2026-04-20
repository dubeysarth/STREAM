"""Feature assembly interfaces for STREAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters import run_legacy_stage
from .manifests import load_manifest_from_config


def run_feature_assembly(config: dict[str, Any], config_path: str | None = None) -> list[list[str]]:
    manifest = load_manifest_from_config(config, config_path)
    return run_legacy_stage(manifest, "02_attach_node_features")


@dataclass
class DynamicFeatureAssembler:
    """Assemble dynamic forcings from the configured STREAM source bundle."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "02_attach_node_features", "dynamic")


@dataclass
class StaticFeatureAssembler:
    """Assemble static terrain, hydrography, and human-context attributes."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> list[list[str]]:
        return run_legacy_stage(load_manifest_from_config(self.config), "02_attach_node_features", "static")
