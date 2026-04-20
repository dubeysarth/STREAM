"""Adapter resolution and legacy stage execution helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from ..paths import StreamPaths


def resolve_adapter_name(manifest: dict[str, Any]) -> str:
    adapter = str(
        manifest.get("adapter")
        or manifest.get("benchmark", {}).get("adapter")
        or manifest.get("region")
        or ""
    ).lower()
    if adapter in {"ohio", "ohio_daily"}:
        return "ohio"
    if adapter in {"camels_us", "continental_us", "us", "us_daily_subset"}:
        return "camels_us"
    if adapter in {"camels_ind", "india", "ind"}:
        return "camels_ind"
    return adapter


def _repo_root(manifest: dict[str, Any]) -> Path:
    config = manifest.get("stage_config", {})
    config_root = (
        Path(manifest["config_path"]).resolve().parent
        if manifest.get("config_path")
        else Path.cwd()
    )
    project_root = config.get("project_root") or config_root
    return Path(StreamPaths(str(project_root)).resolve()["project_root"])


def build_benchmark_components(manifest: dict[str, Any]) -> dict[str, Any]:
    adapter = resolve_adapter_name(manifest)
    repo_root = _repo_root(manifest)
    if adapter == "ohio":
        from .ohio import build_components

        return build_components(repo_root)
    if adapter == "camels_us":
        from .camels_us import build_components

        return build_components(repo_root)
    if adapter == "camels_ind":
        from .camels_ind import build_components

        return build_components(repo_root)
    raise ValueError(f"Unsupported STREAM adapter: {adapter}")


def legacy_stage_commands(manifest: dict[str, Any], stage_name: str, target: str | None = None) -> list[list[str]]:
    adapter = resolve_adapter_name(manifest)
    repo_root = _repo_root(manifest)
    if adapter == "ohio":
        from .ohio import legacy_commands

        return legacy_commands(repo_root, stage_name, target)
    if adapter == "camels_us":
        from .camels_us import legacy_commands

        return legacy_commands(repo_root, stage_name, target)
    if adapter == "camels_ind":
        from .camels_ind import legacy_commands

        return legacy_commands(repo_root, stage_name, target)
    raise ValueError(f"Unsupported STREAM adapter: {adapter}")


def run_legacy_stage(manifest: dict[str, Any], stage_name: str, target: str | None = None) -> list[list[str]]:
    commands = legacy_stage_commands(manifest, stage_name, target)
    for command in commands:
        subprocess.run(command, check=True)
    return commands


def python_cmd(path: Path) -> list[str]:
    return [sys.executable, str(path)]


def notebook_cmd(path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(path),
    ]
