from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import write_json
from .paths import DevpUSPaths


def _list_recursive_zarrs(path: Path) -> set[tuple[str, str]]:
    out = set()
    for item in path.glob("*/*.zarr"):
        out.add((item.parent.name, item.stem.replace(".zarr", "")))
    return out


def build_us_registry(paths: DevpUSPaths) -> pd.DataFrame:
    inventory_pairs = _list_recursive_zarrs(paths.source_us_root / "inventory")
    lumped_pairs = _list_recursive_zarrs(paths.source_us_root / "lumped_inventory")
    graph_attrs = pd.read_csv(paths.source_us_root / "nested_gauges" / "graph_attributes_with_nesting.csv")
    graph_attrs["gauge_id"] = graph_attrs["gauge_id"].astype(str).str.zfill(8)
    graph_attrs["huc_02"] = graph_attrs["huc_02"].astype(int)
    graph_attrs["huc_02_str"] = graph_attrs["huc_02"].astype(str).str.zfill(2)
    graph_attrs["has_inventory"] = graph_attrs.apply(
        lambda row: (row["huc_02_str"], row["gauge_id"]) in inventory_pairs,
        axis=1,
    )
    graph_attrs["has_lumped_inventory"] = graph_attrs.apply(
        lambda row: (row["huc_02_str"], row["gauge_id"]) in lumped_pairs,
        axis=1,
    )
    graph_attrs["graph_dir"] = graph_attrs.apply(
        lambda row: str(paths.source_us_root / "graph_files" / row["huc_02_str"] / row["gauge_id"]),
        axis=1,
    )
    graph_attrs["inventory_dir"] = graph_attrs.apply(
        lambda row: str(paths.source_us_root / "inventory" / row["huc_02_str"] / f"{row['gauge_id']}.zarr"),
        axis=1,
    )
    graph_attrs["lumped_inventory_dir"] = graph_attrs.apply(
        lambda row: str(paths.source_us_root / "lumped_inventory" / row["huc_02_str"] / f"{row['gauge_id']}.zarr"),
        axis=1,
    )
    graph_attrs = graph_attrs[graph_attrs["has_inventory"] | graph_attrs["has_lumped_inventory"]].copy()
    graph_attrs = graph_attrs.sort_values(["huc_02", "gauge_id"]).reset_index(drop=True)
    return graph_attrs


def write_us_registry(paths: DevpUSPaths) -> Path:
    registry = build_us_registry(paths)
    csv_path = paths.manifests_dir / "us_registry.csv"
    registry.to_csv(csv_path, index=False)
    write_json(
        paths.manifests_dir / "us_registry_summary.json",
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

