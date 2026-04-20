"""Model training and ablation interfaces for STREAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import build_benchmark_components
from .manifests import load_manifest_from_config
from .paths import StreamPaths


def _feature_registry(components: dict[str, Any], paths: Any):
    return components["feature_registry_cls"].load(
        components["config_module"].read_json(paths.configs_dir / "feature_registry.json")
    )


def _split_spec(components: dict[str, Any], paths: Any):
    return components["split_cls"](
        **components["config_module"].read_json(paths.configs_dir / "split_spec.json")
    )


def _dynamic_group(manifest: dict[str, Any]) -> str:
    return manifest.get("stage_config", {}).get("model", {}).get("dynamic_group", "era5_full_selected")


def _training_config(components: dict[str, Any], paths: Any, kind: str, manifest: dict[str, Any]):
    config_name = "training_lumped.json" if kind == "lumped" else "training_graph.json"
    config = components["training_config_cls"](
        **components["config_module"].read_json(paths.configs_dir / config_name)
    )
    stage_config = manifest.get("stage_config", {})
    runtime = stage_config.get("runtime", {})
    model = stage_config.get("model", {})
    task = stage_config.get("task", {})
    data = stage_config.get("data", {})
    if runtime.get("device"):
        config.device = runtime["device"]
    if runtime.get("seed") is not None:
        config.seed = runtime["seed"]
    if model.get("history_length") is not None:
        config.history_length = int(model["history_length"])
    if model.get("batch_size") is not None and hasattr(config, "batch_size"):
        config.batch_size = int(model["batch_size"])
    if model.get("graph_batch_size") is not None and hasattr(config, "graph_batch_size"):
        config.graph_batch_size = int(model["graph_batch_size"])
    if model.get("max_epochs") is not None:
        config.max_epochs = int(model["max_epochs"])
    if model.get("loss_name"):
        config.loss_name = model["loss_name"]
    if model.get("static_group"):
        config.static_group = model["static_group"]
    if data.get("frequency"):
        config.frequency = data["frequency"]
    if data.get("hucs") and hasattr(config, "hucs"):
        config.hucs = data["hucs"]
    if task.get("representation") == "distributed" and hasattr(config, "graph_batch_size"):
        config.graph_batch_size = int(model.get("graph_batch_size", config.graph_batch_size))
    return config


@dataclass
class LumpedBaselineRunner:
    """Train and evaluate a controlled lumped baseline on curated STREAM artifacts."""

    config: dict[str, object] = field(default_factory=dict)
    _last_outputs: dict[str, Path] | None = field(default=None, init=False, repr=False)

    def train(self) -> dict[str, Path]:
        manifest = load_manifest_from_config(self.config)
        components = build_benchmark_components(manifest)
        paths = components["paths"]
        context = components["runtime_cls"](
            "train_lumped",
            paths,
            run_name=self.config.get("run_name"),
            dry_run=bool(self.config.get("dry_run")),
        )
        if context.dry_run:
            return {}
        config = _training_config(components, paths, "lumped", manifest)
        outputs = components["train_lumped_run"](
            context,
            paths,
            _feature_registry(components, paths),
            _split_spec(components, paths),
            _dynamic_group(manifest),
            config,
        )
        context.write_manifest({key: str(value) for key, value in outputs.items()})
        context.write_resolved_config({"dynamic_group": _dynamic_group(manifest), **config.__dict__})
        self._last_outputs = outputs
        return outputs

    def evaluate(self) -> pd.DataFrame:
        if self._last_outputs is None:
            self.train()
        return pd.read_csv(self._last_outputs["summary"])


@dataclass
class DistributedBaselineRunner:
    """Train and evaluate the controlled semidistributed or distributed baseline."""

    config: dict[str, object] = field(default_factory=dict)
    _last_outputs: dict[str, Path] | None = field(default=None, init=False, repr=False)

    def train(self) -> dict[str, Path]:
        manifest = load_manifest_from_config(self.config)
        components = build_benchmark_components(manifest)
        paths = components["paths"]
        representation = str(self.config.get("task", {}).get("representation", "distributed"))
        regime = "distributed" if representation == "distributed" else "semidistributed"
        context = components["runtime_cls"](
            f"train_{regime}",
            paths,
            run_name=self.config.get("run_name"),
            dry_run=bool(self.config.get("dry_run")),
        )
        if context.dry_run:
            return {}
        config = _training_config(components, paths, "graph", manifest)
        outputs = components["train_graph_run"](
            context,
            paths,
            _feature_registry(components, paths),
            _split_spec(components, paths),
            _dynamic_group(manifest),
            config,
            regime,
        )
        context.write_manifest({key: str(value) for key, value in outputs.items()})
        context.write_resolved_config({"dynamic_group": _dynamic_group(manifest), "representation": regime, **config.__dict__})
        self._last_outputs = outputs
        return outputs

    def evaluate(self) -> pd.DataFrame:
        if self._last_outputs is None:
            self.train()
        return pd.read_csv(self._last_outputs["summary"])


@dataclass
class AblationSuite:
    """Run or summarize the minimum sufficient ablation bundle."""

    config: dict[str, object] = field(default_factory=dict)

    def run(self) -> Path:
        manifest = load_manifest_from_config(self.config)
        stage_config = manifest.get("stage_config", {})
        repo_root = Path(
            StreamPaths(Path(manifest.get("config_path", Path.cwd())).resolve().parent).resolve()["project_root"]
        )
        outputs_root = (repo_root / stage_config.get("outputs", {}).get("root", "outputs/10_ablation_suite")).resolve()
        outputs_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for csv_path in (repo_root / "results").glob("**/*lock_candidates.csv"):
            df = pd.read_csv(csv_path)
            df.insert(0, "source", csv_path.name)
            rows.append(df)
        summary_path = outputs_root / "ablation_suite_summary.csv"
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(summary_path, index=False)
        else:
            pd.DataFrame(columns=["source"]).to_csv(summary_path, index=False)
        return summary_path
