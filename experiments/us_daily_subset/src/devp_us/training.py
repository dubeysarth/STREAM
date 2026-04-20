from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import (
    DistributedGraphDataset,
    GraphBatchDataset,
    LumpedWindowDataset,
    SemiDistributedGraphDataset,
    build_graph_batch_cache,
)
from .features import FeatureRegistry
from .metrics import MetricBundle
from .models import HybridSeqGConvSeq2One, LumpedSeq2One
from .paths import DevpUSPaths
from .runtime import RunContext
from .splits import SplitSpec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingConfig:
    seed: int = 42
    frequency: str = "daily"
    history_length: int = 365
    batch_size: int = 64
    max_epochs: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    embedding_dim: int = 32
    dropout: float = 0.25
    patience: int = 10
    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None
    device: str = "cpu"
    static_group: str = "static_base"
    loss_name: str = "hydro_skill"
    gradient_clip_norm: float | None = 1.0
    graph_batch_size: int = 64
    hucs: list[str] | None = None


def _collate_lumped(batch: list[dict[str, Any]]) -> dict[str, Any]:
    stacked_keys = [
        "dynamic",
        "static",
        "target",
        "target_raw",
        "usgs",
        "wb_precip",
        "wb_pet",
        "wb_swi_prev",
    ]
    payload: dict[str, Any] = {
        "basin_id": [item["basin_id"] for item in batch],
        "date": [item["date"] for item in batch],
    }
    for key in stacked_keys:
        payload[key] = torch.stack([item[key] for item in batch], dim=0)
    return payload


def _rmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + 1e-8)


def _nse_penalty(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    denom = torch.sum((true_flat - torch.mean(true_flat)) ** 2)
    if float(denom.detach().cpu()) <= 1e-12:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)
    return torch.sum((pred_flat - true_flat) ** 2) / (denom + 1e-12)


def _water_balance_loss(
    q_pred: torch.Tensor,
    swi_pred: torch.Tensor,
    precip: torch.Tensor,
    pet: torch.Tensor,
    swi_prev: torch.Tensor,
) -> torch.Tensor:
    pred_nonnegative = torch.clamp(q_pred, min=0.0)
    residual = precip - pet - pred_nonnegative - (swi_pred - swi_prev)
    return torch.sqrt(torch.mean(residual**2) + 1e-8)


def _objective_terms(prediction: dict[str, torch.Tensor], batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    target = batch["target"].to(device)
    discharge = prediction["discharge"]
    swi = prediction["swi"]
    terms = {
        "rmse": _rmse_loss(target, discharge),
        "nse_penalty": _nse_penalty(target, discharge),
        "wb": _water_balance_loss(
            discharge,
            swi,
            batch["wb_precip"].to(device),
            batch["wb_pet"].to(device),
            batch["wb_swi_prev"].to(device),
        ),
    }
    return terms


def _select_objective(terms: dict[str, torch.Tensor], loss_name: str) -> torch.Tensor:
    if loss_name == "rmse":
        return terms["rmse"]
    if loss_name == "hydro_skill":
        return 0.85 * terms["rmse"] + 0.15 * terms["nse_penalty"]
    if loss_name == "hydro_balance":
        return 0.84 * terms["rmse"] + 0.15 * terms["nse_penalty"] + 0.01 * terms["wb"]
    raise ValueError(f"Unknown loss_name: {loss_name}")


def _train_epoch_lumped(
    model: LumpedSeq2One,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    model.train()
    losses = []
    for batch_idx, batch in enumerate(loader):
        if config.limit_train_batches is not None and batch_idx >= config.limit_train_batches:
            break
        optimizer.zero_grad()
        pred = model(batch["dynamic"].to(device), batch["static"].to(device))
        terms = _objective_terms(pred, batch, device)
        loss = _select_objective(terms, config.loss_name)
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _evaluate_lumped(
    model: LumpedSeq2One,
    loader: DataLoader,
    device: torch.device,
    split: str,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows = []
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if config.limit_eval_batches is not None and batch_idx >= config.limit_eval_batches:
                break
            pred_raw = model(batch["dynamic"].to(device), batch["static"].to(device))
            pred_q_raw = pred_raw["discharge"].detach().cpu()
            pred_swi = pred_raw["swi"].detach().cpu()
            terms = _objective_terms(
                {"discharge": pred_q_raw, "swi": pred_swi},
                batch,
                device=torch.device("cpu"),
            )
            loss = _select_objective(terms, config.loss_name)
            losses.append(float(loss.item()))
            pred = torch.clamp(pred_q_raw, min=0.0).numpy()
            pred_unclamped = pred_q_raw.numpy()
            obs = batch["target_raw"].numpy()
            for basin_id, date, predicted, predicted_raw, predicted_swi, observed, usgs in zip(
                batch["basin_id"],
                batch["date"],
                pred,
                pred_unclamped,
                pred_swi.numpy(),
                obs,
                batch["usgs"].numpy(),
            ):
                rows.append(
                    {
                        "basin_id": basin_id,
                        "date": date,
                        "node_id": 0,
                        "split": split,
                        "is_outlet": True,
                        "predicted": float(predicted),
                        "predicted_raw": float(predicted_raw),
                        "predicted_swi": float(predicted_swi),
                        "observed": float(observed),
                        "usgs_observed": float(usgs),
                    }
                )
    return pd.DataFrame(rows), float(np.mean(losses)) if losses else float("nan")


def _run_graph_epoch(
    model: HybridSeqGConvSeq2One,
    dataset: SemiDistributedGraphDataset | DistributedGraphDataset | GraphBatchDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    split: str,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    rows = []
    losses = []
    iterator = range(len(dataset))
    with torch.set_grad_enabled(is_train):
        for batch_idx in iterator:
            if config.limit_train_batches is not None and is_train and batch_idx >= config.limit_train_batches:
                break
            if config.limit_eval_batches is not None and not is_train and batch_idx >= config.limit_eval_batches:
                break
            sample = dataset[batch_idx]
            if is_train:
                optimizer.zero_grad()
            pred_raw = model(
                sample["dynamic"].to(device),
                sample["static"].to(device),
                sample["edge_index"].to(device),
                sample["edge_weight"].to(device) if sample["edge_weight"].numel() > 0 else None,
            )
            terms = _objective_terms(pred_raw, sample, device)
            loss = _select_objective(terms, config.loss_name)
            if is_train:
                loss.backward()
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            pred_q_raw = pred_raw["discharge"].detach().cpu()
            pred_swi = pred_raw["swi"].detach().cpu()
            pred = torch.clamp(pred_q_raw, min=0.0).numpy()
            pred_unclamped = pred_q_raw.numpy()
            obs = sample["target_raw"].detach().cpu().numpy()
            for node_pos, node_id in enumerate(sample["node_ids"]):
                rows.append(
                    {
                        "basin_id": sample["basin_id_per_node"][node_pos],
                        "date": sample["date_per_node"][node_pos],
                        "node_id": node_id,
                        "split": split,
                        "is_outlet": bool(sample["is_outlet_mask"][node_pos].item()),
                        "predicted": float(pred[node_pos]),
                        "predicted_raw": float(pred_unclamped[node_pos]),
                        "predicted_swi": float(pred_swi[node_pos]),
                        "observed": float(obs[node_pos]),
                        "usgs_observed": float(sample["usgs_per_node"][node_pos].item())
                        if torch.isfinite(sample["usgs_per_node"][node_pos])
                        else math.nan,
                    }
                )
    return pd.DataFrame(rows), float(np.mean(losses)) if losses else float("nan")


def _save_training_artifacts(
    context: RunContext,
    run_id: str,
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict[str, Path]:
    history_path = context.paths.histories_dir / f"{run_id}.csv"
    predictions_path = context.paths.predictions_dir / f"{run_id}.csv"
    summary_path = context.paths.metrics_dir / f"{run_id}_summary.csv"
    per_basin_path = context.paths.metrics_dir / f"{run_id}_per_basin.csv"
    history_df.to_csv(history_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_df, per_basin_df = MetricBundle.summarize_predictions(predictions_df)
    summary_df.to_csv(summary_path, index=False)
    per_basin_df.to_csv(per_basin_path, index=False)
    return {
        "history": history_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "per_basin": per_basin_path,
    }


def _validation_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    summary_df, _ = MetricBundle.summarize_predictions(predictions)
    if summary_df.empty:
        return {"RMSE": float("nan"), "NSE": float("nan"), "PBIAS": float("nan"), "KGE": float("nan")}
    row = summary_df.iloc[0].to_dict()
    return {key: float(row[key]) for key in ["RMSE", "NSE", "PBIAS", "KGE"]}


def train_lumped_run(
    context: RunContext,
    paths: DevpUSPaths,
    feature_registry: FeatureRegistry,
    split_spec: SplitSpec,
    dynamic_group: str,
    config: TrainingConfig,
) -> dict[str, Path]:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_ds = LumpedWindowDataset(
        paths.tensor_regime_dir("lumped", config.frequency),
        feature_registry,
        split_spec,
        "train",
        dynamic_group,
        config.static_group,
        config.history_length,
        frequency=config.frequency,
        allowed_hucs=config.hucs,
    )
    val_ds = LumpedWindowDataset(
        paths.tensor_regime_dir("lumped", config.frequency),
        feature_registry,
        split_spec,
        "val",
        dynamic_group,
        config.static_group,
        config.history_length,
        frequency=config.frequency,
        allowed_hucs=config.hucs,
    )
    test_ds = LumpedWindowDataset(
        paths.tensor_regime_dir("lumped", config.frequency),
        feature_registry,
        split_spec,
        "test",
        dynamic_group,
        config.static_group,
        config.history_length,
        frequency=config.frequency,
        allowed_hucs=config.hucs,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_lumped)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_lumped)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_lumped)
    model = LumpedSeq2One(
        dynamic_dim=len(feature_registry.dynamic_with_time(dynamic_group, frequency=config.frequency)),
        static_dim=int(train_ds[0]["static"].shape[0]),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    history_rows = []
    best_score = -float("inf")
    best_state = None
    patience_left = config.patience

    for epoch in range(config.max_epochs):
        train_loss = _train_epoch_lumped(model, train_loader, optimizer, device, config)
        val_predictions, val_loss = _evaluate_lumped(model, val_loader, device, "val", config)
        val_metrics = _validation_metrics(val_predictions)
        scheduler.step(val_metrics["NSE"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["RMSE"],
                "val_nse": val_metrics["NSE"],
                "val_pbias": val_metrics["PBIAS"],
                "val_kge": val_metrics["KGE"],
            }
        )
        context.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_rmse=%.4f val_nse=%.4f val_kge=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["RMSE"],
            val_metrics["NSE"],
            val_metrics["KGE"],
        )
        if val_metrics["NSE"] > best_score:
            best_score = val_metrics["NSE"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = context.paths.checkpoints_dir / f"{context.run_id}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    all_predictions = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        pred_df, _ = _evaluate_lumped(model, loader, device, split_name, config)
        all_predictions.append(pred_df)
    outputs = _save_training_artifacts(context, context.run_id, pd.DataFrame(history_rows), pd.concat(all_predictions, ignore_index=True))
    outputs["checkpoint"] = checkpoint_path
    return outputs


def train_graph_run(
    context: RunContext,
    paths: DevpUSPaths,
    feature_registry: FeatureRegistry,
    split_spec: SplitSpec,
    dynamic_group: str,
    config: TrainingConfig,
    regime: str,
) -> dict[str, Path]:
    set_seed(config.seed)
    device = torch.device(config.device)
    dataset_cls = SemiDistributedGraphDataset if regime == "semidistributed" else DistributedGraphDataset
    batch_cache_root = paths.tensor_regime_dir(regime, config.frequency)
    try:
        train_ds = GraphBatchDataset(batch_cache_root, dynamic_group, config.static_group, config.history_length, config.graph_batch_size, "train", allowed_hucs=config.hucs)
        val_ds = GraphBatchDataset(batch_cache_root, dynamic_group, config.static_group, config.history_length, config.graph_batch_size, "val", allowed_hucs=config.hucs)
        test_ds = GraphBatchDataset(batch_cache_root, dynamic_group, config.static_group, config.history_length, config.graph_batch_size, "test", allowed_hucs=config.hucs)
        context.info(
            "using cached batched graphs for %s with batch_size=%s",
            regime,
            config.graph_batch_size,
        )
    except FileNotFoundError:
        context.info(
            "graph batch cache missing for %s; building cached batches with batch_size=%s",
            regime,
            config.graph_batch_size,
        )
        build_graph_batch_cache(
            batch_cache_root,
            feature_registry,
            split_spec,
            regime,
            dynamic_group,
            config.static_group,
            config.history_length,
            config.graph_batch_size,
            frequency=config.frequency,
            allowed_hucs=config.hucs,
        )
        train_ds = GraphBatchDataset(batch_cache_root, dynamic_group, config.static_group, config.history_length, config.graph_batch_size, "train", allowed_hucs=config.hucs)
        val_ds = GraphBatchDataset(batch_cache_root, dynamic_group, config.static_group, config.history_length, config.graph_batch_size, "val", allowed_hucs=config.hucs)
        test_ds = GraphBatchDataset(batch_cache_root, dynamic_group, config.static_group, config.history_length, config.graph_batch_size, "test", allowed_hucs=config.hucs)
    model = HybridSeqGConvSeq2One(
        dynamic_dim=len(feature_registry.dynamic_with_time(dynamic_group, frequency=config.frequency)),
        static_dim=int(train_ds[0]["static"].shape[1]),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    history_rows = []
    best_score = -float("inf")
    best_state = None
    patience_left = config.patience

    for epoch in range(config.max_epochs):
        _, train_loss = _run_graph_epoch(model, train_ds, optimizer, device, "train", config)
        val_predictions, val_loss = _run_graph_epoch(model, val_ds, None, device, "val", config)
        outlet_predictions = val_predictions[val_predictions["is_outlet"]].copy()
        if outlet_predictions.empty:
            outlet_predictions = val_predictions
        val_metrics = _validation_metrics(outlet_predictions)
        scheduler.step(val_metrics["NSE"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["RMSE"],
                "val_nse": val_metrics["NSE"],
                "val_pbias": val_metrics["PBIAS"],
                "val_kge": val_metrics["KGE"],
            }
        )
        context.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_rmse=%.4f val_nse=%.4f val_kge=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["RMSE"],
            val_metrics["NSE"],
            val_metrics["KGE"],
        )
        if val_metrics["NSE"] > best_score:
            best_score = val_metrics["NSE"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = context.paths.checkpoints_dir / f"{context.run_id}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    all_predictions = []
    for split_name, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pred_df, _ = _run_graph_epoch(model, dataset, None, device, split_name, config)
        all_predictions.append(pred_df)
    outputs = _save_training_artifacts(context, context.run_id, pd.DataFrame(history_rows), pd.concat(all_predictions, ignore_index=True))
    outputs["checkpoint"] = checkpoint_path
    return outputs
