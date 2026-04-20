from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser


def main() -> None:
    parser = build_parser("Collect run summaries across all prediction files.")
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("evaluate_all", paths, run_name=args.run_name, dry_run=args.dry_run)
    rows = []
    for summary_path in sorted(paths.metrics_dir.glob("*_summary.csv")):
        df = pd.read_csv(summary_path)
        run_id = summary_path.name.replace("_summary.csv", "")
        resolved_path = paths.manifests_runtime_dir / f"{run_id}_resolved_config.json"
        resolved = {}
        if resolved_path.exists():
            resolved = json.loads(resolved_path.read_text())
        df["run_id"] = run_id
        df["dynamic_group"] = resolved.get("dynamic_group", "")
        df["static_group"] = resolved.get("static_group", "")
        df["loss_name"] = resolved.get("loss_name", "")
        df["history_length"] = resolved.get("history_length", "")
        df["frequency"] = resolved.get("frequency", "daily")
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No summary CSVs found in runs/metrics.")
    combined = pd.concat(rows, ignore_index=True)
    output_path = paths.reports_dir / "evaluation_matrix.csv"
    combined.to_csv(output_path, index=False)
    context.write_manifest({"evaluation_matrix": str(output_path), "num_rows": int(len(combined))})
    context.info("wrote combined evaluation matrix with %s rows", len(combined))


if __name__ == "__main__":
    main()
