"""Helpers shared by the numbered STREAM stage scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream.manifests import load_manifest_from_config


def build_parser(stage_name: str) -> argparse.ArgumentParser:
    """Create a consistent CLI parser for the numbered STREAM stage scripts."""

    parser = argparse.ArgumentParser(description=f"STREAM stage: {stage_name}")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved stage plan without executing it.",
    )
    return parser


def load_config(config_path: str) -> tuple[dict[str, object], str]:
    """Load a JSON config for dry-run reporting or execution."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), str(path.resolve())


def summarize(stage_name: str, config: dict[str, object], config_path: str, actions: list[str]) -> None:
    """Print a stable dry-run summary for human inspection."""

    manifest = load_manifest_from_config(config, config_path)
    print(f"[STREAM] stage={stage_name}")
    print(f"  data.dataset={config['data']['dataset_name']}")
    print(f"  task.name={config['task']['name']}")
    print(f"  model.name={config['model']['name']}")
    print(f"  runtime.device={config['runtime']['device']}")
    print(f"  outputs.root={config['outputs']['root']}")
    print(f"  adapter={manifest.get('adapter')}")
    print(f"  manifest={manifest.get('manifest_path', '<inline>')}")
    print("  planned_actions=")
    for action in actions:
        print(f"    - {action}")
