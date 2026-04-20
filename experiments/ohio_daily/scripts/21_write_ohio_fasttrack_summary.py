from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser


def _resolved(paths: DevpOhioPaths, run_id: str) -> dict:
    path = paths.manifests_runtime_dir / f"{run_id}_resolved_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _test_metrics(paths: DevpOhioPaths, run_id: str) -> dict:
    summary = pd.read_csv(paths.metrics_dir / f"{run_id}_summary.csv")
    test = summary[summary["split"] == "test"].iloc[0].to_dict()
    test["run_id"] = run_id
    test.update(_resolved(paths, run_id))
    return test


def main() -> None:
    parser = build_parser("Write the Ohio fast-track daily/monthly lock summary.")
    parser.add_argument("--daily-lock-csv", required=True)
    parser.add_argument("--monthly-lock-csv", required=True)
    parser.add_argument("--daily-semi-run-id", required=True)
    parser.add_argument("--monthly-semi-run-id", required=True)
    parser.add_argument("--monthly-dist-run-id", required=True)
    args = parser.parse_args()

    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("write_ohio_fasttrack_summary", paths, run_name=args.run_name, dry_run=args.dry_run)
    daily_lock = pd.read_csv(args.daily_lock_csv)
    monthly_lock = pd.read_csv(args.monthly_lock_csv)
    daily_choice = daily_lock.sort_values(["NSE", "KGE"], ascending=False).iloc[0].to_dict()
    monthly_choice = monthly_lock.sort_values(["NSE", "KGE"], ascending=False).iloc[0].to_dict()
    daily_semi = _test_metrics(paths, args.daily_semi_run_id)
    monthly_semi = _test_metrics(paths, args.monthly_semi_run_id)
    monthly_dist = _test_metrics(paths, args.monthly_dist_run_id)

    body = [
        "## Variant Lock",
        "",
        f"- Daily lumped lock: `{daily_choice['run_id']}` | loss `{daily_choice['loss_name']}` | dynamic `{daily_choice['dynamic_group']}` | NSE `{daily_choice['NSE']:.3f}` | KGE `{daily_choice['KGE']:.3f}`.",
        f"- Monthly lumped lock: `{monthly_choice['run_id']}` | loss `{monthly_choice['loss_name']}` | dynamic `{monthly_choice['dynamic_group']}` | NSE `{monthly_choice['NSE']:.3f}` | KGE `{monthly_choice['KGE']:.3f}`.",
        "",
        "## Locked Graph Runs",
        "",
        f"- Daily semidistributed: `{args.daily_semi_run_id}` | NSE `{daily_semi['NSE']:.3f}` | KGE `{daily_semi['KGE']:.3f}` | RMSE `{daily_semi['RMSE']:.3f}`.",
        f"- Monthly semidistributed: `{args.monthly_semi_run_id}` | NSE `{monthly_semi['NSE']:.3f}` | KGE `{monthly_semi['KGE']:.3f}` | RMSE `{monthly_semi['RMSE']:.3f}`.",
        f"- Monthly distributed: `{args.monthly_dist_run_id}` | NSE `{monthly_dist['NSE']:.3f}` | KGE `{monthly_dist['KGE']:.3f}` | RMSE `{monthly_dist['RMSE']:.3f}`.",
        "",
        "## Notes",
        "",
        "- Monthly tensors are derived from daily Ohio `.pt` records using variable-specific calendar-month aggregation: flux-like forcings and discharge are summed, state-like variables are averaged, and only month-of-year plus aggregated solar-insolation encodings are retained for monthly training.",
        "- Daily lock uses the existing HUC05 daily tensor pipeline with raw mm/day targets, standardized inputs, and negative predictions clamped at evaluation.",
        "- Human-use context tables use LISFLOOD cell sampling; reservoir counts use Global Dam Watch points assigned to the nearest Ohio graph node and then rolled up to basin or semi-distributed gauge units.",
        "- Context-vs-skill figures are in `reports/figures/*context_vs_skill.png` and are intended for interpretation, not causal claims.",
    ]
    output = paths.summaries_dir / "ohio_fasttrack_summary.md"
    context.append_summary_markdown(output, "Ohio Fast-Track Daily/Monthly Lock Summary", "\n".join(body))
    context.write_manifest({"summary_markdown": str(output)})
    context.info("wrote Ohio fast-track summary to %s", output)


if __name__ == "__main__":
    main()
