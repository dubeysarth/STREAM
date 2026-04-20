from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.config import read_json
from devp_us.datasets import build_graph_batch_cache
from devp_us.features import FeatureRegistry
from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser
from devp_us.splits import SplitSpec


def main() -> None:
    parser = build_parser("Build cached batched-graph .pt files for semidistributed or distributed training.")
    parser.add_argument("--regime", required=True, choices=["semidistributed", "distributed"])
    parser.add_argument("--frequency", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--dynamic-group", default="era5_full_selected")
    parser.add_argument("--static-group", default="static_base")
    parser.add_argument("--history-length", type=int, default=365)
    parser.add_argument("--graph-batch-size", type=int, default=64)
    parser.add_argument("--hucs", nargs="+", default=None, help="Optional HUC-02 subset, e.g. 01 05 10.")
    args = parser.parse_args()

    paths = DevpUSPaths.from_root(args.root)
    context = RunContext(f"build_{args.regime}_graph_batches", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    written = build_graph_batch_cache(
        paths.tensor_regime_dir(args.regime, args.frequency),
        feature_registry,
        split_spec,
        args.regime,
        args.dynamic_group,
        args.static_group,
        args.history_length,
        args.graph_batch_size,
        allowed_hucs=args.hucs,
    )
    payload = {
        "regime": args.regime,
        "dynamic_group": args.dynamic_group,
        "static_group": args.static_group,
        "frequency": args.frequency,
        "history_length": args.history_length,
        "graph_batch_size": args.graph_batch_size,
        "hucs": args.hucs,
        "num_batches": {split: len(paths_written) for split, paths_written in written.items()},
        "first_batch": {
            split: str(paths_written[0]) if paths_written else None for split, paths_written in written.items()
        },
    }
    context.write_manifest(payload)
    context.write_resolved_config(payload)
    context.info(
        "built cached graph batches for %s with counts train=%s val=%s test=%s",
        args.regime,
        len(written.get("train", [])),
        len(written.get("val", [])),
        len(written.get("test", [])),
    )


if __name__ == "__main__":
    main()
