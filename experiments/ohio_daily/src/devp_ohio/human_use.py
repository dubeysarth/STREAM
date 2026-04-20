from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .paths import DevpOhioPaths
from .registry import build_huc05_registry


STATIC_SAMPLERS = {
    "human_fracirrigated": "Land_use/fracirrigated_Global_03min.nc",
    "human_fracsealed": "Land_use/fracsealed_Global_03min.nc",
    "human_fracwater": "Land_use/fracwater_Global_03min.nc",
    "human_reservoir_mask": "Lakes_Reservoirs_ID_tables/20220802_reservoirs_Global_03min.nc",
    "human_lake_mask": "Lakes_Reservoirs_ID_tables/20220802_lakes_Global_03min.nc",
    "human_fracgwused": "Water_demand/fracgwused_Global_03min.nc",
    "human_fracncused": "Water_demand/fracncused_Global_03min.nc",
}

TEMPORAL_SAMPLERS = {
    "human_dom": "Water_demand/dom.nc",
    "human_ene": "Water_demand/ene.nc",
    "human_ind": "Water_demand/ind.nc",
    "human_liv": "Water_demand/liv.nc",
}


def _data_var_name(ds: xr.Dataset) -> str:
    for name in ds.data_vars:
        da = ds[name]
        if name != "crs" and len(da.dims) >= 2:
            return name
    raise ValueError("No usable data variable found.")


def _select_points(da: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> xr.DataArray:
    return da.sel(
        lon=xr.DataArray(lon, dims="point"),
        lat=xr.DataArray(lat, dims="point"),
        method="nearest",
    )


def _sample_static(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    ds = xr.open_dataset(path)
    da = ds[_data_var_name(ds)]
    sampled = _select_points(da, lon, lat)
    return np.nan_to_num(np.asarray(sampled.compute()).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _sample_temporal_summary(path: Path, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    da = ds[_data_var_name(ds)].sel(time=slice("1999-01-01", "2019-12-31"))
    sampled = _select_points(da, lon, lat)
    mean = np.nan_to_num(np.asarray(sampled.mean("time").compute()).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(np.asarray(sampled.std("time").compute()).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    return mean, std


def _aggregate_weighted(df: pd.DataFrame, group_cols: list[str], weight_col: str) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col not in group_cols + [weight_col]]
    rows = []
    for group_value, group_df in df.groupby(group_cols, sort=False):
        weights = group_df[weight_col].to_numpy(dtype=float)
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        row = {}
        if isinstance(group_value, tuple):
            for key, value in zip(group_cols, group_value):
                row[key] = value
        else:
            row[group_cols[0]] = group_value
        for feature in feature_cols:
            values = group_df[feature].to_numpy(dtype=float)
            row[feature] = float(np.average(values, weights=weights))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_gdw_reservoir_counts(paths: DevpOhioPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    reservoir_path = paths.gdw_root / "GDW_v1_0_shp" / "GDW_reservoirs_v1_0.shp"
    if not reservoir_path.exists():
        empty = pd.DataFrame(columns=["gauge_id", "gdw_reservoir_count"])
        return empty, empty

    registry = build_huc05_registry(paths)
    node_rows = []
    for basin_id in registry["gauge_id"].tolist():
        nodes = pd.read_csv(paths.source_ohio_root / "graph_files" / "05" / basin_id / "nodes_coords.csv")
        nodes = nodes[["lon", "lat"]].copy()
        nodes["basin_id"] = basin_id
        assigned_path = paths.source_ohio_root / "nested_gauges" / "nesting_info" / "05" / basin_id / "nodes_coords_assigned.csv"
        if assigned_path.exists():
            assigned = pd.read_csv(assigned_path)[["lon", "lat", "assigned_gauge_id"]].copy()
            assigned["assigned_gauge_id"] = assigned["assigned_gauge_id"].astype(int).astype(str).str.zfill(8)
            nodes = nodes.merge(assigned, on=["lon", "lat"], how="left")
        nodes["assigned_gauge_id"] = nodes.get("assigned_gauge_id", basin_id)
        nodes["assigned_gauge_id"] = nodes["assigned_gauge_id"].fillna(basin_id).astype(str).str.zfill(8)
        node_rows.append(nodes)

    all_nodes = pd.concat(node_rows, ignore_index=True)
    bounds = {
        "minx": float(all_nodes["lon"].min() - 0.5),
        "maxx": float(all_nodes["lon"].max() + 0.5),
        "miny": float(all_nodes["lat"].min() - 0.5),
        "maxy": float(all_nodes["lat"].max() + 0.5),
    }
    gdw = gpd.read_file(reservoir_path, bbox=(bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]))
    if gdw.empty:
        empty = pd.DataFrame(columns=["gauge_id", "gdw_reservoir_count"])
        return empty, empty
    geom = gdw.geometry.centroid
    coords = np.column_stack([geom.x.to_numpy(dtype=float), geom.y.to_numpy(dtype=float)])
    tree = cKDTree(all_nodes[["lon", "lat"]].to_numpy(dtype=float))
    _, nearest_idx = tree.query(coords, k=1)
    assigned = all_nodes.iloc[np.asarray(nearest_idx, dtype=int)].copy()
    basin_counts = (
        assigned.groupby("basin_id", sort=True)
        .size()
        .rename("gdw_reservoir_count")
        .reset_index()
        .rename(columns={"basin_id": "gauge_id"})
    )
    semi_counts = (
        assigned.groupby(["basin_id", "assigned_gauge_id"], sort=True)
        .size()
        .rename("gdw_reservoir_count")
        .reset_index()
        .rename(columns={"assigned_gauge_id": "gauge_id"})
    )
    return basin_counts, semi_counts


def extract_huc05_human_use(paths: DevpOhioPaths) -> dict[str, Path]:
    registry = build_huc05_registry(paths)
    distributed_dir = paths.human_use_dir / "distributed"
    semidistributed_dir = paths.human_use_dir / "semidistributed"
    distributed_dir.mkdir(parents=True, exist_ok=True)
    semidistributed_dir.mkdir(parents=True, exist_ok=True)
    basin_reservoir_counts, semi_reservoir_counts = _load_gdw_reservoir_counts(paths)

    basin_rows = []

    for basin_id in registry["gauge_id"].tolist():
        nodes = pd.read_csv(paths.source_ohio_root / "graph_files" / "05" / basin_id / "nodes_coords.csv")
        lon = nodes["lon"].to_numpy(dtype=float)
        lat = nodes["lat"].to_numpy(dtype=float)
        node_table = nodes[["lon", "lat"]].copy()
        node_table.insert(0, "node_idx", np.arange(len(node_table), dtype=int))

        for feature_name, relative_path in STATIC_SAMPLERS.items():
            node_table[feature_name] = _sample_static(paths.lisflood_root / relative_path, lon, lat)
        for prefix, relative_path in TEMPORAL_SAMPLERS.items():
            mean, std = _sample_temporal_summary(paths.lisflood_root / relative_path, lon, lat)
            node_table[f"{prefix}_mean"] = mean
            node_table[f"{prefix}_std"] = std

        ds = xr.open_zarr(paths.source_ohio_root / "inventory" / "05" / f"{basin_id}.zarr")
        node_table["cellarea_km2"] = np.asarray(ds["static_GloFAS_cellarea_km2"].values, dtype=float)
        node_path = distributed_dir / f"05_{basin_id}.csv"
        node_table.to_csv(node_path, index=False)

        assigned_path = paths.source_ohio_root / "nested_gauges" / "nesting_info" / "05" / basin_id / "nodes_coords_assigned.csv"
        if assigned_path.exists():
            assigned = pd.read_csv(assigned_path)
            merged = node_table.merge(
                assigned[["lon", "lat", "assigned_gauge_id"]],
                on=["lon", "lat"],
                how="left",
            )
            merged["assigned_gauge_id"] = merged["assigned_gauge_id"].fillna(int(basin_id)).astype(int).astype(str).str.zfill(8)
        else:
            merged = node_table.copy()
            merged["assigned_gauge_id"] = basin_id
        semi = _aggregate_weighted(merged, ["assigned_gauge_id"], "cellarea_km2").rename(
            columns={"assigned_gauge_id": "gauge_id"}
        )
        if not semi_reservoir_counts.empty:
            semi = semi.merge(
                semi_reservoir_counts[semi_reservoir_counts["basin_id"] == basin_id][["gauge_id", "gdw_reservoir_count"]],
                on="gauge_id",
                how="left",
            )
        semi["gdw_reservoir_count"] = semi.get("gdw_reservoir_count", 0.0)
        semi["gdw_reservoir_count"] = semi["gdw_reservoir_count"].fillna(0.0).astype(float)
        semi.to_csv(semidistributed_dir / f"05_{basin_id}.csv", index=False)

        basin_summary = _aggregate_weighted(merged, ["assigned_gauge_id"], "cellarea_km2")
        basin_row = basin_summary.drop(columns=["assigned_gauge_id"]).mean(axis=0).to_dict()
        basin_row["gauge_id"] = basin_id
        if not basin_reservoir_counts.empty:
            match = basin_reservoir_counts[basin_reservoir_counts["gauge_id"] == basin_id]
            basin_row["gdw_reservoir_count"] = float(match["gdw_reservoir_count"].iloc[0]) if not match.empty else 0.0
        else:
            basin_row["gdw_reservoir_count"] = 0.0
        basin_rows.append(basin_row)

    lumped = pd.DataFrame(basin_rows).sort_values("gauge_id").reset_index(drop=True)
    lumped_path = paths.human_use_dir / "lumped_huc05.csv"
    lumped.to_csv(lumped_path, index=False)

    return {
        "distributed_dir": distributed_dir,
        "semidistributed_dir": semidistributed_dir,
        "lumped_table": lumped_path,
    }
