from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.config import read_json
from devp_us.datasets import (
    build_distributed_tensors,
    build_lumped_tensors,
    build_semidistributed_tensors,
    fit_and_save_scalers,
)
from devp_us.features import FeatureRegistry
from devp_us.paths import DevpUSPaths
from devp_us.registry import build_us_registry
from devp_us.runtime import RunContext, build_parser
from devp_us.splits import SplitSpec


def _filter_registry_for_ready_human_use(paths: DevpUSPaths, registry, regimes: list[str]):
    filtered = registry.copy()
    if "distributed" in regimes or "semidistributed" in regimes:
        ready_ids = {
            path.stem
            for path in (paths.human_use_dir / "distributed").glob("*.csv")
        }
        filtered = filtered[
            filtered.apply(
                lambda row: f"{row['huc_02_str']}_{row['gauge_id']}" in ready_ids,
                axis=1,
            )
        ].copy()
    if "lumped" in regimes:
        lumped_path = paths.human_use_dir / "lumped_camels_us.csv"
        if not lumped_path.exists():
            filtered = filtered.iloc[0:0].copy()
    return filtered


def _normalize_huc_codes(values: list[str]) -> set[str]:
    return {str(value).zfill(2) for value in values}


def _filter_registry_for_hucs(registry, hucs: list[str]):
    filtered = registry.copy()
    filtered["huc_02_str"] = filtered["huc_02_str"].astype(str).str.zfill(2)
    wanted = _normalize_huc_codes(hucs)
    return filtered[filtered["huc_02_str"].isin(wanted)].copy()


def _filter_registry_for_missing_outputs(paths: DevpUSPaths, registry, regimes: list[str]):
    filtered = registry.copy()
    filtered["huc_02_str"] = filtered["huc_02_str"].astype(str).str.zfill(2)
    filtered["gauge_id"] = filtered["gauge_id"].astype(str).str.zfill(8)

    def needs_output(row) -> bool:
        for regime in regimes:
            output_path = paths.tensor_dir / regime / f"{row['huc_02_str']}_{row['gauge_id']}.pt"
            if not output_path.exists():
                return True
        return False

    return filtered[filtered.apply(needs_output, axis=1)].copy()


def main() -> None:
    parser = build_parser("Build CAMELS-US lumped, semidistributed, and distributed tensor caches.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=["lumped", "distributed", "semidistributed"],
        default=["lumped", "distributed", "semidistributed"],
    )
    parser.add_argument("--hucs", nargs="+", default=None, help="Optional HUC-02 codes to process, e.g. 01 02 03.")
    parser.add_argument("--only-ready-human-use", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-scalers", action="store_true")
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("build_us_tensors", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    registry = build_us_registry(paths)
    if args.hucs:
        registry = _filter_registry_for_hucs(registry, args.hucs)
    if args.only_ready_human_use:
        registry = _filter_registry_for_ready_human_use(paths, registry, args.regimes)
    if args.skip_existing:
        registry = _filter_registry_for_missing_outputs(paths, registry, args.regimes)

    lumped = []
    distributed = []
    semidistributed = []
    if "lumped" in args.regimes:
        lumped = build_lumped_tensors(paths, feature_registry, registry)
    if "distributed" in args.regimes:
        distributed = build_distributed_tensors(paths, feature_registry, registry)
    if "semidistributed" in args.regimes:
        semidistributed = build_semidistributed_tensors(paths, registry)
    scalers = {} if args.skip_scalers else fit_and_save_scalers(paths, split_spec)
    payload = {
        "regimes": args.regimes,
        "hucs": args.hucs,
        "only_ready_human_use": bool(args.only_ready_human_use),
        "skip_existing": bool(args.skip_existing),
        "num_lumped": len(lumped),
        "num_distributed": len(distributed),
        "num_semidistributed": len(semidistributed),
        "scalers": scalers,
    }
    context.write_manifest(payload)
    context.info(
        "built tensors lumped=%s distributed=%s semidistributed=%s",
        len(lumped),
        len(distributed),
        len(semidistributed),
    )


if __name__ == "__main__":
    main()
