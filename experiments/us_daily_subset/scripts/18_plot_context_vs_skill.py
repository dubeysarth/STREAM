from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.metrics import MetricBundle
from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser


PLOT_FEATURES = [
    ("gdw_reservoir_count", "Reservoir Count"),
    ("human_fracirrigated", "Irrigated Fraction"),
    ("human_fracsealed", "Sealed Fraction"),
    ("human_dom_mean", "Domestic Demand Mean"),
]


def _style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.9,
        rc={
            "axes.facecolor": "#fbfbfc",
            "figure.facecolor": "white",
            "axes.edgecolor": "#c8c8cf",
            "grid.color": "#e3e4ea",
        },
    )


def _lumped_context(paths: DevpUSPaths, run_id: str) -> pd.DataFrame:
    predictions = pd.read_csv(paths.predictions_dir / f"{run_id}.csv")
    test = predictions[predictions["split"] == "test"].copy()
    test["basin_id"] = test["basin_id"].astype(str).str.zfill(8)
    rows = []
    for basin_id, basin_df in test.groupby("basin_id", sort=True):
        metrics = MetricBundle.compute(basin_df["observed"].to_numpy(), basin_df["predicted"].to_numpy())
        metrics["gauge_id"] = basin_id
        rows.append(metrics)
    perf = pd.DataFrame(rows)
    human = pd.read_csv(paths.human_use_dir / "lumped_camels_us.csv")
    human["gauge_id"] = human["gauge_id"].astype(str).str.zfill(8)
    return perf.merge(human, on="gauge_id", how="left")


def _semi_context(paths: DevpUSPaths, run_id: str) -> pd.DataFrame:
    predictions = pd.read_csv(paths.predictions_dir / f"{run_id}.csv")
    test = predictions[predictions["split"] == "test"].copy()
    test["basin_id"] = test["basin_id"].astype(str).str.zfill(8)
    test["node_id"] = test["node_id"].astype(str).str.zfill(8)
    rows = []
    for (basin_id, node_id), node_df in test.groupby(["basin_id", "node_id"], sort=True):
        metrics = MetricBundle.compute(node_df["observed"].to_numpy(), node_df["predicted"].to_numpy())
        metrics["basin_id"] = basin_id
        metrics["gauge_id"] = node_id
        rows.append(metrics)
    perf = pd.DataFrame(rows)
    human_rows = []
    for path in sorted((paths.human_use_dir / "semidistributed").glob("*.csv")):
        df = pd.read_csv(path)
        basin_id = path.stem.split("_", 1)[1]
        df["basin_id"] = basin_id
        df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
        human_rows.append(df)
    human = pd.concat(human_rows, ignore_index=True)
    return perf.merge(human, on=["basin_id", "gauge_id"], how="left")


def _render_panel(df: pd.DataFrame, output_path: Path, title: str) -> Path:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (feature, label) in zip(axes.flat, PLOT_FEATURES):
        plot_df = df[[feature, "NSE", "KGE"]].copy()
        plot_df = plot_df[pd.notna(plot_df[feature])].copy()
        if plot_df.empty:
            ax.set_axis_off()
            continue
        sns.regplot(
            data=plot_df,
            x=feature,
            y="NSE",
            scatter_kws={"s": 36, "alpha": 0.75, "color": "#457b9d"},
            line_kws={"color": "#d62828", "linewidth": 2.0},
            ax=ax,
        )
        rho = plot_df[[feature, "NSE"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{label} vs NSE | Spearman={rho:.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("NSE")
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = build_parser("Generate context-vs-skill figures for lumped and semidistributed runs.")
    parser.add_argument("--lumped-run-id", required=True)
    parser.add_argument("--semi-run-id", required=True)
    args = parser.parse_args()

    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("plot_context_vs_skill", paths, run_name=args.run_name, dry_run=args.dry_run)
    lumped_df = _lumped_context(paths, args.lumped_run_id)
    semi_df = _semi_context(paths, args.semi_run_id)
    lumped_csv = paths.reports_dir / f"{args.lumped_run_id}_context_metrics.csv"
    semi_csv = paths.reports_dir / f"{args.semi_run_id}_context_metrics.csv"
    lumped_df.to_csv(lumped_csv, index=False)
    semi_df.to_csv(semi_csv, index=False)
    outputs = [
        _render_panel(
            lumped_df,
            paths.figures_dir / f"{args.lumped_run_id}_context_vs_skill.png",
            f"Lumped Outlet Skill vs Human/Reservoir Context | {args.lumped_run_id}",
        ),
        _render_panel(
            semi_df,
            paths.figures_dir / f"{args.semi_run_id}_context_vs_skill.png",
            f"Semi-Distributed Subcatchment Skill vs Human/Reservoir Context | {args.semi_run_id}",
        ),
    ]
    context.write_manifest(
        {
            "lumped_csv": str(lumped_csv),
            "semi_csv": str(semi_csv),
            "figures": [str(path) for path in outputs],
        }
    )
    context.info("generated %s context-vs-skill figures", len(outputs))


if __name__ == "__main__":
    main()
