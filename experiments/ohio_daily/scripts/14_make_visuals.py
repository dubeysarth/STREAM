from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.paths import DevpOhioPaths
from devp_ohio.plots import PlotBundle
from devp_ohio.runtime import RunContext, build_parser


def _render_reference_graph(paths: DevpOhioPaths) -> Path:
    basin_id = "03164000"
    nodes = pd.read_csv(paths.source_ohio_root / "graph_files" / "05" / basin_id / "nodes_coords.csv")
    edges = pd.read_csv(paths.source_ohio_root / "graph_files" / "05" / basin_id / "edges.csv")
    plt.figure(figsize=(9, 7))
    plt.style.use("seaborn-v0_8-whitegrid")
    for _, edge in edges.iterrows():
        src = nodes.iloc[int(edge["from_idx"])]
        dst = nodes.iloc[int(edge["to_idx"])]
        plt.plot([src["lon"], dst["lon"]], [src["lat"], dst["lat"]], color="#4d908e", linewidth=1.2, alpha=0.55)
    plt.scatter(nodes["lon"], nodes["lat"], s=24, color="#1d3557", edgecolors="white", linewidths=0.3)
    plt.title("Representative HUC05 Nested Basin Graph | 03164000", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    out = paths.figures_dir / "reference_graph_03164000.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def main() -> None:
    parser = build_parser("Generate figures from saved histories and predictions.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("make_visuals", paths, run_name=args.run_name, dry_run=args.dry_run)
    outputs = []
    if args.run_id is not None:
        run_ids = [args.run_id]
    else:
        run_ids = sorted({path.stem for path in paths.predictions_dir.glob("*.csv")} & {path.stem for path in paths.histories_dir.glob("*.csv")})
    for run_id in run_ids:
        outputs.extend(
            PlotBundle.render(
                paths,
                run_id,
                paths.predictions_dir / f"{run_id}.csv",
                paths.histories_dir / f"{run_id}.csv",
            )
        )
    outputs.append(_render_reference_graph(paths))
    context.write_manifest({"figures": [str(path) for path in outputs]})
    context.info("generated %s figures", len(outputs))


if __name__ == "__main__":
    main()
