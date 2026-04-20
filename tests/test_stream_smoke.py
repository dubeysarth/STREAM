"""Smoke tests for the shared STREAM package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from stream import DatasetManifest, StreamPaths


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{current}" if current else str(REPO_ROOT / "src")
    return env


def test_stream_paths_resolve() -> None:
    resolved = StreamPaths(str(REPO_ROOT)).resolve()
    assert Path(resolved["project_root"]) == REPO_ROOT
    assert Path(resolved["src_root"]).exists()
    assert Path(resolved["configs_root"]).exists()
    assert Path(resolved["reference_stream_root"]).exists()


def test_manifest_loads_template_manifest() -> None:
    manifest = DatasetManifest(
        name="ohio_daily",
        resolution="03min",
        region="ohio",
        manifest_path=str(REPO_ROOT / "configs" / "manifests" / "ohio_daily.json"),
    ).load()
    assert manifest["adapter"] == "ohio"
    assert manifest["benchmark"]["experiment_root"].endswith("experiments/ohio_daily")


def test_main_module_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "stream"],
        cwd=REPO_ROOT,
        env=_script_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "STREAM package is importable." in proc.stdout


def test_stage_scripts_support_help() -> None:
    for script in sorted((REPO_ROOT / "scripts").glob("*.py")):
        if script.name.startswith("_"):
            continue
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            env=_script_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        assert "STREAM" in proc.stdout or "usage:" in proc.stdout


def test_stage_scripts_support_dry_run() -> None:
    configs = {
        "01_curate_graphs.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "02_attach_node_features.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "03_prepare_nested_gauges.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "04_build_zarr_inventory.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "05_build_climate_summaries.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "06_build_monthly_inventory.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "07_build_lumped_inventory.py": REPO_ROOT / "configs" / "templates" / "shared_template.json",
        "08_run_lumped_baseline.py": REPO_ROOT / "configs" / "templates" / "08_us_monthly_lumped.json",
        "09_run_distributed_baseline.py": REPO_ROOT / "configs" / "templates" / "09_us_monthly_distributed.json",
        "10_run_ablation_suite.py": REPO_ROOT / "configs" / "templates" / "10_topology_off_control.json",
        "11_run_validation_bundle.py": REPO_ROOT / "configs" / "templates" / "11_validation_bundle.json",
        "12_build_rebuttal_assets.py": REPO_ROOT / "configs" / "templates" / "12_rebuttal_assets.json",
    }
    for script_name, config_path in configs.items():
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--config", str(config_path), "--dry-run"],
            cwd=REPO_ROOT,
            env=_script_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        assert "[STREAM]" in proc.stdout
