from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import write_json
from .paths import DevpOhioPaths


def _list_zarr_gauges(path: Path) -> list[str]:
    return sorted(item.stem.replace(".zarr", "") for item in path.glob("*.zarr"))


def build_huc05_registry(paths: DevpOhioPaths) -> pd.DataFrame:
    inventory_gauges = set(_list_zarr_gauges(paths.source_ohio_root / "inventory" / "05"))
    lumped_gauges = set(_list_zarr_gauges(paths.source_ohio_root / "lumped_inventory" / "05"))
    graph_attrs = pd.read_csv(paths.source_ohio_root / "nested_gauges" / "graph_attributes_with_nesting.csv")
    graph_attrs["gauge_id"] = graph_attrs["gauge_id"].astype(str).str.zfill(8)
    graph_attrs = graph_attrs[graph_attrs["huc_02"] == 5].copy()
    graph_attrs["has_inventory"] = graph_attrs["gauge_id"].isin(inventory_gauges)
    graph_attrs["has_lumped_inventory"] = graph_attrs["gauge_id"].isin(lumped_gauges)
    graph_attrs["graph_dir"] = graph_attrs["gauge_id"].map(
        lambda gauge: str(paths.source_ohio_root / "graph_files" / "05" / gauge)
    )
    graph_attrs["inventory_dir"] = graph_attrs["gauge_id"].map(
        lambda gauge: str(paths.source_ohio_root / "inventory" / "05" / f"{gauge}.zarr")
    )
    graph_attrs["lumped_inventory_dir"] = graph_attrs["gauge_id"].map(
        lambda gauge: str(paths.source_ohio_root / "lumped_inventory" / "05" / f"{gauge}.zarr")
    )
    graph_attrs = graph_attrs.sort_values("gauge_id").reset_index(drop=True)
    return graph_attrs


def write_huc05_registry(paths: DevpOhioPaths) -> Path:
    registry = build_huc05_registry(paths)
    csv_path = paths.manifests_dir / "huc05_registry.csv"
    registry.to_csv(csv_path, index=False)
    write_json(
        paths.manifests_dir / "huc05_registry_summary.json",
        {
            "num_basins": int(len(registry)),
            "num_inventory": int(registry["has_inventory"].sum()),
            "num_lumped_inventory": int(registry["has_lumped_inventory"].sum()),
            "num_nested_downstream": int((registry["nesting"] == "nested_downstream").sum()),
            "num_nested_upstream": int((registry["nesting"] == "nested_upstream").sum()),
            "num_not_nested": int((registry["nesting"] == "not_nested").sum()),
        },
    )
    return csv_path
