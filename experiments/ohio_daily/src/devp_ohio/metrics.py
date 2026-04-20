from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(array)
    return array[mask]


def _paired_arrays(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(true_arr), len(pred_arr))
    true_arr = true_arr[:n]
    pred_arr = pred_arr[:n]
    mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    true_arr = true_arr[mask]
    pred_arr = pred_arr[mask]
    true_arr = np.clip(true_arr, a_min=0.0, a_max=None)
    pred_arr = np.clip(pred_arr, a_min=0.0, a_max=None)
    return true_arr, pred_arr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_pred - y_true) ** 2) / denom)


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    denom = np.sum(y_true)
    if abs(denom) <= 1e-12:
        return float("nan")
    return float(100.0 * np.sum(y_true - y_pred) / denom)


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan")
    std_true = np.std(y_true)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    if std_true <= 0.0 or mean_true == 0.0:
        return float("nan")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(corr):
        return float("nan")
    alpha = std_pred / std_true
    beta = mean_pred / mean_true
    return float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


@dataclass(frozen=True)
class MetricBundle:
    """Reusable multi-metric evaluation helper."""

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "RMSE": rmse(y_true, y_pred),
            "NSE": nse(y_true, y_pred),
            "PBIAS": pbias(y_true, y_pred),
            "KGE": kge(y_true, y_pred),
        }

    @staticmethod
    def summarize_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_rows = []
        per_basin_rows = []
        for split in sorted(df["split"].unique()):
            split_df = df[df["split"] == split].copy()
            split_metrics = MetricBundle.compute(split_df["observed"].to_numpy(), split_df["predicted"].to_numpy())
            split_metrics["split"] = split
            summary_rows.append(split_metrics)

            for basin_id, basin_df in split_df.groupby("basin_id", sort=True):
                basin_metrics = MetricBundle.compute(basin_df["observed"].to_numpy(), basin_df["predicted"].to_numpy())
                basin_metrics["split"] = split
                basin_metrics["basin_id"] = basin_id
                basin_metrics["n_samples"] = int(len(basin_df))
                basin_metrics["is_outlet_only"] = bool(basin_df["is_outlet"].all())
                per_basin_rows.append(basin_metrics)

        return pd.DataFrame(summary_rows), pd.DataFrame(per_basin_rows)
