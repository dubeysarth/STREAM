from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from .features import FeatureRegistry, HUMAN_USE_FEATURES, WATER_BALANCE_FEATURES
from .paths import DevpOhioPaths
from .splits import SplitSpec


def _to_numpy_array(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.compute().values, dtype=np.float32)


def _dataset_feature_names(ds: xr.Dataset, prefix: str) -> list[str]:
    return sorted(name for name in ds.data_vars if name.startswith(prefix))


def _lumped_static_features(ds: xr.Dataset) -> tuple[list[str], np.ndarray]:
    names = [name for name in _dataset_feature_names(ds, "static_")]
    coords = ["encoding_transformed_longitude", "encoding_transformed_latitude"]
    values = [_to_numpy_array(ds[name]).reshape(1) for name in names]
    for coord in coords:
        values.append(_to_numpy_array(ds[coord]).reshape(1))
        names.append(coord)
    return names, np.concatenate(values).astype(np.float32)


def _distributed_static_features(ds: xr.Dataset) -> tuple[list[str], np.ndarray]:
    names = []
    arrays = []
    for name in _dataset_feature_names(ds, "static_"):
        da = ds[name]
        if "idx" in da.dims:
            arrays.append(_to_numpy_array(da).reshape(-1, 1))
            names.append(name)
    for coord_name in ["encoding_transformed_longitude", "encoding_transformed_latitude"]:
        arrays.append(_to_numpy_array(ds[coord_name]).reshape(-1, 1))
        names.append(coord_name)
    return names, np.concatenate(arrays, axis=1).astype(np.float32)


def _lumped_dynamic_features(ds: xr.Dataset, feature_registry: FeatureRegistry) -> tuple[list[str], np.ndarray]:
    feature_names = feature_registry.all_dynamic_union_with_time()
    arrays = [_to_numpy_array(ds[name]).reshape(-1, 1) for name in feature_names]
    return feature_names, np.concatenate(arrays, axis=1).astype(np.float32)


def _distributed_dynamic_features(ds: xr.Dataset, feature_registry: FeatureRegistry) -> tuple[list[str], np.ndarray]:
    feature_names = feature_registry.all_dynamic_union_with_time()
    arrays = []
    node_count = int(ds.sizes["idx"])
    for name in feature_names:
        arr = _to_numpy_array(ds[name])
        if arr.ndim == 1:
            arr = np.repeat(arr[:, None], node_count, axis=1)
        arrays.append(arr[..., None])
    return feature_names, np.concatenate(arrays, axis=-1).astype(np.float32)


def _load_human_use_node_table(paths: DevpOhioPaths, basin_id: str) -> pd.DataFrame | None:
    path = paths.human_use_dir / "distributed" / f"05_{basin_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_human_use_semidistributed_table(paths: DevpOhioPaths, basin_id: str) -> pd.DataFrame | None:
    path = paths.human_use_dir / "semidistributed" / f"05_{basin_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_human_use_lumped_table(paths: DevpOhioPaths, basin_id: str) -> pd.Series | None:
    path = paths.human_use_dir / "lumped_huc05.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["gauge_id"].astype(str).str.zfill(8) == basin_id]
    if row.empty:
        return None
    return row.iloc[0]


def _edge_index_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    edges = pd.read_csv(path)
    edge_index = edges[["from_idx", "to_idx"]].to_numpy(dtype=np.int64).T
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weight


def _record_path(root: Path, regime: str, basin_id: str) -> Path:
    return root / regime / f"05_{basin_id}.pt"


def _analysis_slice(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(time=slice("1998-01-01", "2019-12-31"))


def build_lumped_tensors(paths: DevpOhioPaths, feature_registry: FeatureRegistry) -> list[Path]:
    output_dir = paths.tensor_dir / "lumped"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for zarr_path in sorted((paths.source_ohio_root / "lumped_inventory" / "05").glob("*.zarr")):
        basin_id = zarr_path.stem.replace(".zarr", "")
        ds = _analysis_slice(xr.open_zarr(zarr_path))
        static_names, static_values = _lumped_static_features(ds)
        human = _load_human_use_lumped_table(paths, basin_id)
        if human is not None:
            human_values = np.asarray([float(human.get(feature, 0.0)) for feature in HUMAN_USE_FEATURES], dtype=np.float32)
            human_values = np.nan_to_num(human_values, nan=0.0, posinf=0.0, neginf=0.0)
            static_values = np.concatenate([static_values, human_values], axis=0)
            static_names = static_names + HUMAN_USE_FEATURES
        dynamic_names, dynamic = _lumped_dynamic_features(ds, feature_registry)
        payload = {
            "basin_id": basin_id,
            "regime": "lumped",
            "frequency": "daily",
            "time": pd.to_datetime(ds.time.values).strftime("%Y-%m-%d").tolist(),
            "dynamic_feature_names": dynamic_names,
            "static_feature_names": static_names,
            "dynamic": torch.from_numpy(dynamic),
            "static": torch.from_numpy(static_values),
            "target": torch.from_numpy(_to_numpy_array(ds["dynamic_GloFAS_discharge_mm"])),
            "swi": torch.from_numpy(_to_numpy_array(ds["dynamic_GloFAS_soil_wetness_index"])),
            "usgs": torch.from_numpy(_to_numpy_array(ds["outlet_USGS_Q_mm"])),
        }
        output_path = _record_path(paths.tensor_dir, "lumped", basin_id)
        torch.save(payload, output_path)
        written.append(output_path)
    return written


def build_distributed_tensors(paths: DevpOhioPaths, feature_registry: FeatureRegistry) -> list[Path]:
    output_dir = paths.tensor_dir / "distributed"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for zarr_path in sorted((paths.source_ohio_root / "inventory" / "05").glob("*.zarr")):
        basin_id = zarr_path.stem.replace(".zarr", "")
        ds = _analysis_slice(xr.open_zarr(zarr_path))
        static_names, static_values = _distributed_static_features(ds)
        human = _load_human_use_node_table(paths, basin_id)
        if human is not None:
            human_values = np.nan_to_num(human[HUMAN_USE_FEATURES].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            static_values = np.concatenate([static_values, human_values], axis=1)
            static_names = static_names + HUMAN_USE_FEATURES
        dynamic_names, dynamic = _distributed_dynamic_features(ds, feature_registry)
        target = _to_numpy_array(ds["dynamic_GloFAS_discharge_mm"])
        outlet_target = _to_numpy_array(ds["outlet_USGS_Q_mm"])
        edge_index, edge_weight = _edge_index_from_csv(paths.source_ohio_root / "graph_files" / "05" / basin_id / "edges.csv")
        payload = {
            "basin_id": basin_id,
            "regime": "distributed",
            "frequency": "daily",
            "time": pd.to_datetime(ds.time.values).strftime("%Y-%m-%d").tolist(),
            "dynamic_feature_names": dynamic_names,
            "static_feature_names": static_names,
            "dynamic": torch.from_numpy(dynamic),
            "static": torch.from_numpy(static_values),
            "target": torch.from_numpy(target),
            "swi": torch.from_numpy(_to_numpy_array(ds["dynamic_GloFAS_soil_wetness_index"])),
            "usgs": torch.from_numpy(outlet_target),
            "edge_index": torch.from_numpy(edge_index),
            "edge_weight": torch.from_numpy(edge_weight),
            "outlet_idx": int(np.asarray(ds.attrs["gauge_idx"], dtype=int)[0]),
            "node_ids": list(range(static_values.shape[0])),
        }
        output_path = _record_path(paths.tensor_dir, "distributed", basin_id)
        torch.save(payload, output_path)
        written.append(output_path)
    return written


def build_semidistributed_tensors(paths: DevpOhioPaths, feature_registry: FeatureRegistry) -> list[Path]:
    output_dir = paths.tensor_dir / "semidistributed"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record_path in sorted((paths.tensor_dir / "distributed").glob("*.pt")):
        if record_path.name == "scalers.pt":
            continue
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        basin_id = record["basin_id"]
        assigned_path = paths.source_ohio_root / "nested_gauges" / "nesting_info" / "05" / basin_id / "nodes_coords_assigned.csv"
        edge_gauge_path = paths.source_ohio_root / "nested_gauges" / "nesting_info" / "05" / basin_id / "edges_gauges.csv"
        if assigned_path.exists():
            assigned = pd.read_csv(assigned_path)
            groups = assigned["assigned_gauge_id"].astype(int).astype(str).str.zfill(8).tolist()
        else:
            node_count = record["static"].shape[0]
            groups = [basin_id] * node_count
        node_df = pd.DataFrame(
            {
                "node_idx": list(range(len(groups))),
                "gauge_id": groups,
                "cellarea_km2": record["static"][:, record["static_feature_names"].index("static_GloFAS_cellarea_km2")].numpy(),
            }
        )
        gauge_ids = list(dict.fromkeys(groups))
        static_rows = []
        target_rows = []
        swi_rows = []
        dynamic_rows = []
        outlet_index_map: dict[str, int] = {basin_id: int(record["outlet_idx"])}
        nested_json = paths.source_ohio_root / "nested_gauges" / "nesting_info" / "05" / basin_id / "nested_gauges.json"
        if nested_json.exists():
            import json

            nested = json.loads(nested_json.read_text())
            outlet_index_map[nested["downstream_gauge"]] = int(record["outlet_idx"])
            for upstream_gauge, meta in nested.get("upstream_gauges", {}).items():
                outlet_index_map[str(upstream_gauge).zfill(8)] = int(meta["node_idx"])
        for gauge_id in gauge_ids:
            member_indices = node_df.loc[node_df["gauge_id"] == gauge_id, "node_idx"].to_numpy(dtype=int)
            weights = node_df.loc[node_df["gauge_id"] == gauge_id, "cellarea_km2"].to_numpy(dtype=float)
            weights = weights / weights.sum()
            static_rows.append(np.average(record["static"].numpy()[member_indices], axis=0, weights=weights))
            dynamic_rows.append(np.average(record["dynamic"].numpy()[:, member_indices, :], axis=1, weights=weights))
            target_rows.append(record["target"].numpy()[:, outlet_index_map.get(gauge_id, int(record["outlet_idx"]))])
            swi_rows.append(np.average(record["swi"].numpy()[:, member_indices], axis=1, weights=weights))
        static_arr = np.stack(static_rows, axis=0).astype(np.float32)
        dynamic_arr = np.stack(dynamic_rows, axis=1).astype(np.float32)
        target_arr = np.stack(target_rows, axis=1).astype(np.float32)
        swi_arr = np.stack(swi_rows, axis=1).astype(np.float32)
        if nested_json.exists():
            edges = []
            downstream_gauge = str(nested["downstream_gauge"]).zfill(8)
            downstream_pos = gauge_ids.index(downstream_gauge) if downstream_gauge in gauge_ids else 0
            for upstream_gauge in nested.get("upstream_gauges", {}).keys():
                upstream_gauge = str(upstream_gauge).zfill(8)
                if upstream_gauge in gauge_ids:
                    edges.append([gauge_ids.index(upstream_gauge), downstream_pos])
            if edges:
                edge_index = np.asarray(edges, dtype=np.int64).T
                edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_weight = np.zeros((0,), dtype=np.float32)
        elif edge_gauge_path.exists():
            edge_index, edge_weight = _edge_index_from_csv(edge_gauge_path)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_weight = np.zeros((0,), dtype=np.float32)
        payload = {
            "basin_id": basin_id,
            "regime": "semidistributed",
            "frequency": "daily",
            "time": record["time"],
            "dynamic_feature_names": record["dynamic_feature_names"],
            "static_feature_names": record["static_feature_names"],
            "dynamic": torch.from_numpy(dynamic_arr),
            "static": torch.from_numpy(static_arr),
            "target": torch.from_numpy(target_arr),
            "swi": torch.from_numpy(swi_arr),
            "usgs": record["usgs"],
            "edge_index": torch.from_numpy(edge_index),
            "edge_weight": torch.from_numpy(edge_weight),
            "outlet_idx": int(gauge_ids.index(basin_id)) if basin_id in gauge_ids else 0,
            "node_ids": gauge_ids,
        }
        output_path = _record_path(paths.tensor_dir, "semidistributed", basin_id)
        torch.save(payload, output_path)
        written.append(output_path)
    return written


def _monthly_group_indices(time_values: list[str]) -> tuple[list[str], list[np.ndarray]]:
    series = pd.Series(pd.to_datetime(time_values))
    month_codes = series.dt.to_period("M")
    grouped = series.groupby(month_codes, sort=True)
    out_dates: list[str] = []
    groups: list[np.ndarray] = []
    for period, group in grouped:
        out_dates.append(period.to_timestamp(how="end").strftime("%Y-%m-%d"))
        groups.append(group.index.to_numpy(dtype=int))
    return out_dates, groups


def _monthly_mean_tensor(tensor: torch.Tensor, groups: list[np.ndarray]) -> torch.Tensor:
    source = tensor.float()
    reduced = [source[group].mean(dim=0) for group in groups]
    return torch.stack(reduced, dim=0).to(torch.float32)


def _monthly_sum_tensor(tensor: torch.Tensor, groups: list[np.ndarray]) -> torch.Tensor:
    source = tensor.float()
    reduced = [source[group].sum(dim=0) for group in groups]
    return torch.stack(reduced, dim=0).to(torch.float32)


def _monthly_dynamic_agg(name: str) -> str:
    sum_features = {
        "dynamic_ERA5_total_precipitation",
        "dynamic_ERA5_surface_net_solar_radiation",
        "dynamic_ERA5_surface_net_thermal_radiation",
        "dynamic_ERA5_evaporation",
        "dynamic_ERA5_potential_evaporation",
        "dynamic_ERA5_runoff",
        "dynamic_ERA5_surface_runoff",
        "dynamic_ERA5_sub_surface_runoff",
        "dynamic_GloFAS_discharge_mm",
        "dynamic_GloFAS_runoff_water_equivalent",
        "encoding_solar_insolation",
    }
    if name in sum_features:
        return "sum"
    return "mean"


def _aggregate_monthly_dynamic(
    tensor: torch.Tensor,
    feature_names: list[str],
    groups: list[np.ndarray],
) -> torch.Tensor:
    source = tensor.float()
    out = []
    for feature_idx, feature_name in enumerate(feature_names):
        feature_tensor = source[..., feature_idx]
        agg = _monthly_dynamic_agg(feature_name)
        if agg == "sum":
            reduced = [feature_tensor[group].sum(dim=0) for group in groups]
        else:
            reduced = [feature_tensor[group].mean(dim=0) for group in groups]
        out.append(torch.stack(reduced, dim=0))
    return torch.stack(out, dim=-1).to(torch.float32)


def build_monthly_tensors(paths: DevpOhioPaths, regime: str) -> list[Path]:
    source_dir = paths.tensor_regime_dir(regime, "daily")
    output_dir = paths.tensor_regime_dir(regime, "monthly")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record_path in sorted(source_dir.glob("*.pt")):
        if record_path.name == "scalers.pt":
            continue
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        monthly_time, groups = _monthly_group_indices(record["time"])
        payload = {key: value for key, value in record.items()}
        payload["time"] = monthly_time
        payload["frequency"] = "monthly"
        payload["dynamic"] = _aggregate_monthly_dynamic(record["dynamic"], record["dynamic_feature_names"], groups)
        payload["target"] = _monthly_sum_tensor(record["target"], groups)
        payload["usgs"] = _monthly_sum_tensor(record["usgs"], groups)
        if "swi" in record:
            payload["swi"] = _monthly_mean_tensor(record["swi"], groups)
        output_path = output_dir / record_path.name
        torch.save(payload, output_path)
        written.append(output_path)
    return written


def _fit_standard_scalers(records: list[dict[str, Any]], split_spec: SplitSpec, dynamic_names: list[str], static_names: list[str]) -> dict[str, torch.Tensor]:
    dyn_values = []
    static_values = []
    for record in records:
        dates = pd.to_datetime(record["time"])
        warmup_mask = pd.Series(dates).between(split_spec.warmup_start, split_spec.train_end).to_numpy()
        dyn_values.append(record["dynamic"][warmup_mask].reshape(-1, len(dynamic_names)).numpy())
        static_values.append(record["static"].reshape(-1, len(static_names)).numpy())
    dyn = np.concatenate(dyn_values, axis=0)
    stat = np.concatenate(static_values, axis=0)
    return {
        "dynamic_mean": torch.from_numpy(dyn.mean(axis=0).astype(np.float32)),
        "dynamic_std": torch.from_numpy(np.clip(dyn.std(axis=0), 1e-6, None).astype(np.float32)),
        "static_mean": torch.from_numpy(stat.mean(axis=0).astype(np.float32)),
        "static_std": torch.from_numpy(np.clip(stat.std(axis=0), 1e-6, None).astype(np.float32)),
    }


def fit_and_save_scalers(
    paths: DevpOhioPaths,
    split_spec: SplitSpec,
    *,
    frequency: str = "daily",
    regimes: tuple[str, ...] = ("lumped", "semidistributed", "distributed"),
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for regime in regimes:
        regime_dir = paths.tensor_regime_dir(regime, frequency)
        records = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(regime_dir.glob("*.pt"))
            if path.name != "scalers.pt"
        ]
        if not records:
            continue
        scalers = _fit_standard_scalers(
            records,
            split_spec=split_spec,
            dynamic_names=records[0]["dynamic_feature_names"],
            static_names=records[0]["static_feature_names"],
        )
        output_path = regime_dir / "scalers.pt"
        torch.save(scalers, output_path)
        written[regime] = output_path
    return written


@dataclass
class WindowSample:
    basin_id: str
    split: str
    date: str
    dynamic: torch.Tensor
    static: torch.Tensor
    target: torch.Tensor
    usgs: torch.Tensor | None
    node_ids: list[Any]
    edge_index: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    outlet_idx: int = 0


class LumpedWindowDataset(Dataset):
    def __init__(
        self,
        tensor_root: Path,
        feature_registry: FeatureRegistry,
        split_spec: SplitSpec,
        split: str,
        dynamic_group: str,
        static_group: str,
        history_length: int = 365,
        frequency: str = "daily",
    ) -> None:
        self.split = split
        self.history_length = history_length
        self.frequency = frequency
        self.records = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(tensor_root.glob("*.pt"))
            if path.name != "scalers.pt"
        ]
        self.scalers = torch.load(tensor_root / "scalers.pt", map_location="cpu", weights_only=False)
        self.dynamic_names = feature_registry.dynamic_with_time(dynamic_group, frequency=frequency)
        self.static_group = static_group
        self.feature_registry = feature_registry
        self.dynamic_name_to_idx = {name: idx for idx, name in enumerate(self.records[0]["dynamic_feature_names"])}
        self.index: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            indices = split_spec.target_indices(record["time"], split=split, history_length=history_length)
            self.index.extend((record_idx, idx) for idx in indices)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record_idx, target_idx = self.index[item]
        record = self.records[record_idx]
        dynamic_idx = [record["dynamic_feature_names"].index(name) for name in self.dynamic_names]
        selected_static = self.feature_registry.resolve_static_names(self.static_group, record["static_feature_names"])
        static_idx = [record["static_feature_names"].index(name) for name in selected_static]
        dynamic = record["dynamic"][target_idx - self.history_length : target_idx, dynamic_idx]
        static = record["static"][static_idx]
        target = record["target"][target_idx]
        dynamic = (dynamic - self.scalers["dynamic_mean"][dynamic_idx]) / self.scalers["dynamic_std"][dynamic_idx]
        static = (static - self.scalers["static_mean"][static_idx]) / self.scalers["static_std"][static_idx]
        precip = torch.clamp(record["dynamic"][target_idx, self.dynamic_name_to_idx[WATER_BALANCE_FEATURES["precip"]]], min=0.0) * 1000.0
        pet = torch.clamp(-record["dynamic"][target_idx, self.dynamic_name_to_idx[WATER_BALANCE_FEATURES["pet"]]], min=0.0) * 1000.0
        swi_prev = record["swi"][target_idx - 1] if "swi" in record else torch.zeros((), dtype=torch.float32)
        return {
            "basin_id": record["basin_id"],
            "date": record["time"][target_idx],
            "dynamic": dynamic.float(),
            "static": static.float(),
            "target": target.float(),
            "target_raw": target.float(),
            "usgs": record["usgs"][target_idx].float(),
            "wb_precip": precip.float(),
            "wb_pet": pet.float(),
            "wb_swi_prev": swi_prev.float(),
        }


class _GraphWindowBase(Dataset):
    def __init__(
        self,
        tensor_root: Path,
        feature_registry: FeatureRegistry,
        split_spec: SplitSpec,
        split: str,
        dynamic_group: str,
        static_group: str,
        history_length: int = 365,
        frequency: str = "daily",
    ) -> None:
        self.split = split
        self.history_length = history_length
        self.static_group = static_group
        self.frequency = frequency
        self.records = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(tensor_root.glob("*.pt"))
            if path.name != "scalers.pt"
        ]
        self.scalers = torch.load(tensor_root / "scalers.pt", map_location="cpu", weights_only=False)
        self.dynamic_names = feature_registry.dynamic_with_time(dynamic_group, frequency=frequency)
        self.feature_registry = feature_registry
        self.dynamic_name_to_idx = {name: idx for idx, name in enumerate(self.records[0]["dynamic_feature_names"])}
        self.index: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            indices = split_spec.target_indices(record["time"], split=split, history_length=history_length)
            self.index.extend((record_idx, idx) for idx in indices)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record_idx, target_idx = self.index[item]
        record = self.records[record_idx]
        dynamic_idx = [record["dynamic_feature_names"].index(name) for name in self.dynamic_names]
        selected_static = self.feature_registry.resolve_static_names(self.static_group, record["static_feature_names"])
        static_idx = [record["static_feature_names"].index(name) for name in selected_static]
        dynamic = record["dynamic"][target_idx - self.history_length : target_idx, :, :][:, :, dynamic_idx]
        static = record["static"][..., static_idx]
        target = record["target"][target_idx]
        dynamic = (dynamic - self.scalers["dynamic_mean"][dynamic_idx]) / self.scalers["dynamic_std"][dynamic_idx]
        static = (static - self.scalers["static_mean"][static_idx]) / self.scalers["static_std"][static_idx]
        precip = torch.clamp(record["dynamic"][target_idx, :, self.dynamic_name_to_idx[WATER_BALANCE_FEATURES["precip"]]], min=0.0) * 1000.0
        pet = torch.clamp(-record["dynamic"][target_idx, :, self.dynamic_name_to_idx[WATER_BALANCE_FEATURES["pet"]]], min=0.0) * 1000.0
        swi_prev = record["swi"][target_idx - 1] if "swi" in record else torch.zeros_like(target)
        node_count = int(static.shape[0])
        outlet_mask = torch.zeros(node_count, dtype=torch.bool)
        outlet_mask[int(record["outlet_idx"])] = True
        usgs_per_node = torch.full((node_count,), torch.nan, dtype=torch.float32)
        usgs_per_node[int(record["outlet_idx"])] = record["usgs"][target_idx].float()
        return {
            "basin_id": record["basin_id"],
            "date": record["time"][target_idx],
            "dynamic": dynamic.float(),
            "static": static.float(),
            "target": target.float(),
            "target_raw": target.float(),
            "usgs": record["usgs"][target_idx].float(),
            "edge_index": record["edge_index"].long(),
            "edge_weight": record["edge_weight"].float(),
            "outlet_idx": int(record["outlet_idx"]),
            "node_ids": record["node_ids"],
            "wb_precip": precip.float(),
            "wb_pet": pet.float(),
            "wb_swi_prev": swi_prev.float(),
            "basin_id_per_node": [record["basin_id"]] * node_count,
            "date_per_node": [record["time"][target_idx]] * node_count,
            "is_outlet_mask": outlet_mask,
            "usgs_per_node": usgs_per_node,
        }


class SemiDistributedGraphDataset(_GraphWindowBase):
    pass


class DistributedGraphDataset(_GraphWindowBase):
    pass


def graph_batch_cache_dir(
    tensor_root: Path,
    dynamic_group: str,
    static_group: str,
    history_length: int,
    batch_size: int,
    split: str,
) -> Path:
    cache_key = f"{dynamic_group}__{static_group}__h{history_length}__b{batch_size}"
    return tensor_root / "batch" / cache_key / split


def _collate_graph_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    dynamic_chunks = []
    static_chunks = []
    target_chunks = []
    target_raw_chunks = []
    wb_precip_chunks = []
    wb_pet_chunks = []
    wb_swi_prev_chunks = []
    edge_indices = []
    edge_weights = []
    node_ids: list[Any] = []
    basin_id_per_node: list[str] = []
    date_per_node: list[str] = []
    outlet_mask_chunks = []
    usgs_per_node_chunks = []
    node_offset = 0

    for sample in samples:
        node_count = int(sample["static"].shape[0])
        dynamic_chunks.append(sample["dynamic"].half())
        static_chunks.append(sample["static"].half())
        target_chunks.append(sample["target"].float())
        target_raw_chunks.append(sample["target_raw"].float())
        wb_precip_chunks.append(sample["wb_precip"].float())
        wb_pet_chunks.append(sample["wb_pet"].float())
        wb_swi_prev_chunks.append(sample["wb_swi_prev"].float())
        node_ids.extend(sample["node_ids"])
        basin_id_per_node.extend(sample["basin_id_per_node"])
        date_per_node.extend(sample["date_per_node"])
        outlet_mask_chunks.append(sample["is_outlet_mask"])
        usgs_per_node_chunks.append(sample["usgs_per_node"])
        if sample["edge_index"].numel() > 0:
            edge_indices.append(sample["edge_index"] + node_offset)
            edge_weights.append(sample["edge_weight"])
        node_offset += node_count

    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    edge_weight = torch.cat(edge_weights, dim=0) if edge_weights else torch.zeros((0,), dtype=torch.float32)
    return {
        "dynamic": torch.cat(dynamic_chunks, dim=1),
        "static": torch.cat(static_chunks, dim=0),
        "target": torch.cat(target_chunks, dim=0),
        "target_raw": torch.cat(target_raw_chunks, dim=0),
        "wb_precip": torch.cat(wb_precip_chunks, dim=0),
        "wb_pet": torch.cat(wb_pet_chunks, dim=0),
        "wb_swi_prev": torch.cat(wb_swi_prev_chunks, dim=0),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "node_ids": node_ids,
        "basin_id_per_node": basin_id_per_node,
        "date_per_node": date_per_node,
        "is_outlet_mask": torch.cat(outlet_mask_chunks, dim=0),
        "usgs_per_node": torch.cat(usgs_per_node_chunks, dim=0),
        "num_graphs": len(samples),
    }


def build_graph_batch_cache(
    tensor_root: Path,
    feature_registry: FeatureRegistry,
    split_spec: SplitSpec,
    regime: str,
    dynamic_group: str,
    static_group: str,
    history_length: int,
    batch_size: int,
    frequency: str = "daily",
) -> dict[str, list[Path]]:
    dataset_cls = SemiDistributedGraphDataset if regime == "semidistributed" else DistributedGraphDataset
    written: dict[str, list[Path]] = {}
    for split in ["train", "val", "test"]:
        dataset = dataset_cls(
            tensor_root,
            feature_registry,
            split_spec,
            split,
            dynamic_group,
            static_group,
            history_length,
            frequency=frequency,
        )
        out_dir = graph_batch_cache_dir(tensor_root, dynamic_group, static_group, history_length, batch_size, split)
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.pt"):
            old.unlink()
        split_written: list[Path] = []
        for batch_idx, start in enumerate(range(0, len(dataset), batch_size)):
            samples = [dataset[idx] for idx in range(start, min(start + batch_size, len(dataset)))]
            payload = _collate_graph_samples(samples)
            payload["split"] = split
            payload["dynamic_group"] = dynamic_group
            payload["static_group"] = static_group
            payload["history_length"] = history_length
            output_path = out_dir / f"batch_{batch_idx:05d}.pt"
            torch.save(payload, output_path)
            split_written.append(output_path)
        written[split] = split_written
    return written


class GraphBatchDataset(Dataset):
    def __init__(
        self,
        tensor_root: Path,
        dynamic_group: str,
        static_group: str,
        history_length: int,
        batch_size: int,
        split: str,
    ) -> None:
        self.batch_dir = graph_batch_cache_dir(tensor_root, dynamic_group, static_group, history_length, batch_size, split)
        self.paths = sorted(self.batch_dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No cached graph batches found under {self.batch_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int) -> dict[str, Any]:
        payload = torch.load(self.paths[item], map_location="cpu", weights_only=False)
        payload["dynamic"] = payload["dynamic"].float()
        payload["static"] = payload["static"].float()
        return payload
