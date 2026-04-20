from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser

OBSOLETE_RUN_IDS: set[str] = set()


def main() -> None:
    parser = build_parser("Write markdown run summaries from saved metrics.")
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("write_run_summary", paths, run_name=args.run_name, dry_run=args.dry_run)
    summary_rows = []
    for summary_path in sorted(paths.metrics_dir.glob("*_summary.csv")):
        df = pd.read_csv(summary_path)
        run_id = summary_path.name.replace("_summary.csv", "")
        if run_id.startswith("bench_") or "smoke" in run_id:
            continue
        if run_id in OBSOLETE_RUN_IDS:
            continue
        resolved_path = paths.manifests_runtime_dir / f"{run_id}_resolved_config.json"
        resolved = {}
        if resolved_path.exists():
            resolved = json.loads(resolved_path.read_text())
        test = df[df["split"] == "test"]
        if test.empty:
            continue
        row = test.iloc[0].to_dict()
        row["run_id"] = run_id
        row["dynamic_group"] = resolved.get("dynamic_group", "")
        row["static_group"] = resolved.get("static_group", "")
        row["loss_name"] = resolved.get("loss_name", "")
        row["frequency"] = resolved.get("frequency", "daily")
        summary_rows.append(row)
    if not summary_rows:
        raise FileNotFoundError("No test summaries available.")
    table = pd.DataFrame(summary_rows).sort_values("NSE", ascending=False)
    table_path = paths.reports_dir / "run_summary_table.csv"
    table.to_csv(table_path, index=False)
    body = [
        "## Top Runs",
        "",
        table.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Models are trained on raw `mm/day` discharge targets while inputs are standardized from warmup+train data.",
        "- Evaluation metrics clamp negative predictions to zero before scoring.",
        "- Default dynamic inputs follow the reference lumped setup: ERA5 plus seasonal/time encodings.",
        "- Outlet-only diagnostics use the outlet node for graph runs and the sole outlet for lumped runs.",
        "- Detailed artifacts live under `runs/` and figures live under `reports/figures/`.",
    ]
    markdown_path = paths.summaries_dir / "run_summary.md"
    context.append_summary_markdown(markdown_path, "CAMELS-US Run Summary", "\n".join(body))
    context.write_manifest({"summary_markdown": str(markdown_path), "summary_table": str(table_path)})
    context.info("wrote markdown summary for %s runs", len(table))


if __name__ == "__main__":
    main()
