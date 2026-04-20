from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.config import read_json
from devp_us.datasets import fit_and_save_scalers
from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser
from devp_us.splits import SplitSpec


def main() -> None:
    parser = build_parser("Fit and save tensor scalers for the selected CAMELS-US regimes.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=["lumped", "semidistributed", "distributed"],
        default=["lumped", "semidistributed", "distributed"],
    )
    parser.add_argument("--frequency", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--hucs", nargs="+", default=None, help="Optional HUC-02 subset, e.g. 01 05 10.")
    args = parser.parse_args()

    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("fit_scalers", paths, run_name=args.run_name, dry_run=args.dry_run)
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    written = fit_and_save_scalers(
        paths,
        split_spec,
        frequency=args.frequency,
        regimes=tuple(args.regimes),
        allowed_hucs=args.hucs,
    )
    context.write_manifest({"regimes": args.regimes, "frequency": args.frequency, "hucs": args.hucs, "written": written})
    context.info("wrote %s scaler files", len(written))


if __name__ == "__main__":
    main()
