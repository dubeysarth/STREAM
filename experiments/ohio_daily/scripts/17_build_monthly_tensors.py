from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.config import read_json
from devp_ohio.datasets import build_monthly_tensors, fit_and_save_scalers
from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser
from devp_ohio.splits import SplitSpec


def main() -> None:
    parser = build_parser("Build monthly tensor datasets from cached daily .pt records.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["lumped", "semidistributed", "distributed"],
        choices=["lumped", "semidistributed", "distributed"],
    )
    args = parser.parse_args()

    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("build_monthly_tensors", paths, run_name=args.run_name, dry_run=args.dry_run)
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    written = {}
    for regime in args.regimes:
        paths.tensor_regime_dir(regime, "monthly").mkdir(parents=True, exist_ok=True)
        written[regime] = [str(path) for path in build_monthly_tensors(paths, regime)]
        context.info("built %s monthly tensor files for %s", len(written[regime]), regime)
    scaler_paths = fit_and_save_scalers(paths, split_spec, frequency="monthly", regimes=tuple(args.regimes))
    payload = {
        "regimes": args.regimes,
        "num_files": {regime: len(paths_written) for regime, paths_written in written.items()},
        "scalers": {regime: str(path) for regime, path in scaler_paths.items()},
    }
    context.write_manifest(payload)
    context.write_resolved_config(payload)


if __name__ == "__main__":
    main()
