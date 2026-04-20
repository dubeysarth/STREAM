"""Validation and reporting interfaces for STREAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .manifests import load_manifest_from_config
from .paths import StreamPaths


@dataclass
class ValidationBundle:
    """Render validation summaries that bridge metrics, diagnostics, and reviewer-facing evidence."""

    config: dict[str, object] = field(default_factory=dict)

    def render(self) -> Path:
        manifest = load_manifest_from_config(self.config)
        repo_root = Path(
            StreamPaths(Path(manifest.get("config_path", Path.cwd())).resolve().parent).resolve()["project_root"]
        )
        outputs_root = (repo_root / manifest["stage_config"].get("outputs", {}).get("root", "outputs/11_validation_bundle")).resolve()
        outputs_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for csv_path in (repo_root / "results").glob("**/evaluation_matrix.csv"):
            df = pd.read_csv(csv_path)
            df.insert(0, "source", csv_path.parent.name)
            rows.append(df)
        summary_path = outputs_root / "validation_bundle.csv"
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(summary_path, index=False)
        else:
            pd.DataFrame(columns=["source"]).to_csv(summary_path, index=False)
        return summary_path


@dataclass
class RebuttalAssetBuilder:
    """Render reviewer-facing assets and shared result indexes."""

    config: dict[str, object] = field(default_factory=dict)

    def render(self) -> Path:
        manifest = load_manifest_from_config(self.config)
        repo_root = Path(
            StreamPaths(Path(manifest.get("config_path", Path.cwd())).resolve().parent).resolve()["project_root"]
        )
        outputs_root = (repo_root / manifest["stage_config"].get("outputs", {}).get("root", "outputs/12_rebuttal_assets")).resolve()
        outputs_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for path in sorted((repo_root / "results").glob("**/*.csv")):
            rows.append({"kind": "csv", "path": str(path.relative_to(repo_root))})
        for path in sorted((repo_root / "results").glob("**/*.png")):
            rows.append({"kind": "figure", "path": str(path.relative_to(repo_root))})
        for path in sorted((repo_root / "docs").glob("*.md")):
            rows.append({"kind": "doc", "path": str(path.relative_to(repo_root))})
        summary_path = outputs_root / "rebuttal_asset_index.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        return summary_path
