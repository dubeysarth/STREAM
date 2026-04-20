from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import write_json
from .paths import DevpOhioPaths
from .splits import SplitSpec


@dataclass(frozen=True)
class SourceManifest:
    """Writes a machine-readable manifest of all critical source paths."""

    paths: DevpOhioPaths
    split_spec: SplitSpec

    def resolve(self) -> dict[str, object]:
        payload = {
            "ohio_root": str(self.paths.source_ohio_root),
            "inventory_huc05": str(self.paths.source_ohio_root / "inventory" / "05"),
            "lumped_inventory_huc05": str(self.paths.source_ohio_root / "lumped_inventory" / "05"),
            "graph_files_huc05": str(self.paths.source_ohio_root / "graph_files" / "05"),
            "nested_gauges_root": str(self.paths.source_ohio_root / "nested_gauges"),
            "scaling_huc05": str(self.paths.source_ohio_root / "scaling" / "HUC05"),
            "lumped_reference_root": str(self.paths.source_lumped_reference_root),
            "lisflood_root": str(self.paths.lisflood_root),
            "split_spec": self.split_spec.build(),
        }
        output = self.paths.manifests_dir / "source_manifest.json"
        write_json(output, payload)
        return payload

    def verify(self) -> None:
        required = [
            self.paths.source_ohio_root / "inventory" / "05",
            self.paths.source_ohio_root / "lumped_inventory" / "05",
            self.paths.source_ohio_root / "graph_files" / "05",
            self.paths.source_ohio_root / "nested_gauges",
            self.paths.lisflood_root,
        ]
        missing = [str(path) for path in required if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required source paths: {missing}")
