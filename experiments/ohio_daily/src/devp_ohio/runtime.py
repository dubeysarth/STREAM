from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import write_json
from .paths import DevpOhioPaths


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", default="/data/projects/KDD2026", help="Project root.")
    parser.add_argument("--run-name", default=None, help="Optional stable run name.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    return parser


@dataclass
class RunContext:
    """Shared logger and artifact helper for scripts."""

    name: str
    paths: DevpOhioPaths
    run_name: str | None = None
    dry_run: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        self.paths.ensure()
        self.run_id = self.run_name or f"{self.name}_{self.timestamp}"
        self.log_path = self.paths.logs_dir / f"{self.run_id}.log"
        self.logger = logging.getLogger(self.run_id)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.info("run_id=%s dry_run=%s cwd=%s", self.run_id, self.dry_run, os.getcwd())

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)
        text = message % args if args else message
        print(text)

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        path = self.paths.manifests_runtime_dir / f"{self.run_id}.json"
        write_json(path, payload)
        return path

    def write_resolved_config(self, payload: dict[str, Any]) -> Path:
        path = self.paths.manifests_runtime_dir / f"{self.run_id}_resolved_config.json"
        write_json(path, payload)
        return path

    def append_summary_markdown(self, path: str | Path, title: str, body: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            handle.write(f"# {title}\n\n{body.strip()}\n")
