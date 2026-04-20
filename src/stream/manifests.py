"""Dataset manifest contracts for the STREAM package."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import StreamPaths


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_relative_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        if isinstance(value, str) and value and not value.startswith("<"):
            path_like = Path(value)
            if not path_like.is_absolute() and ("/" in value or value.endswith(".json") or value.endswith(".py") or value.endswith(".ipynb")):
                return str((base_dir / path_like).resolve())
        return value

    return _resolve(payload)


def _resolve_manifest_path(manifest_path: str | None, config_path: str | None) -> str | None:
    if not manifest_path:
        return None

    candidate = Path(manifest_path)
    if candidate.is_absolute() or not config_path:
        return str(candidate.resolve())

    config_dir = Path(config_path).resolve().parent
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return str(config_relative)

    repo_root = Path(StreamPaths(str(config_dir)).resolve()["project_root"])
    repo_relative = (repo_root / candidate).resolve()
    if repo_relative.exists():
        return str(repo_relative)

    return str(config_relative)


@dataclass
class DatasetManifest:
    """Describe one curated STREAM dataset slice and its dependencies."""

    name: str
    resolution: str
    region: str
    sources: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    manifest_path: str | None = None

    def load(self) -> dict[str, object]:
        payload: dict[str, Any] = {
            "dataset_id": self.name,
            "name": self.name,
            "resolution": self.resolution,
            "region": self.region,
            "sources": self.sources,
            "notes": dict(self.notes),
        }
        if self.manifest_path:
            manifest_file = Path(self.manifest_path).resolve()
            disk_payload = _read_json(manifest_file)
            payload.update(_resolve_relative_paths(disk_payload, manifest_file.parent))
            payload["manifest_path"] = str(manifest_file)
        payload.setdefault("adapter", payload.get("region"))
        payload.setdefault("benchmark", {})
        payload.setdefault("legacy", {})
        payload.setdefault("paths", {})
        payload.setdefault("sources", self.sources)
        payload.setdefault("notes", dict(self.notes))
        return payload


def load_manifest_from_config(config: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    data = config.get("data", {})
    manifest_path = _resolve_manifest_path(data.get("manifest_path"), config_path)
    manifest = DatasetManifest(
        name=str(data.get("dataset_name", data.get("name", "stream_dataset"))),
        resolution=str(data.get("resolution", "unknown")),
        region=str(data.get("region", "unknown")),
        sources=list(data.get("sources", [])),
        notes=dict(data.get("notes", {})),
        manifest_path=manifest_path,
    ).load()
    manifest["stage_config"] = config
    if config_path:
        manifest["config_path"] = str(Path(config_path).resolve())
    return manifest
