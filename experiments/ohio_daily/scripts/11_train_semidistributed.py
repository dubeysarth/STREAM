from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.config import read_json
from devp_ohio.features import FeatureRegistry
from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser
from devp_ohio.splits import SplitSpec
from devp_ohio.training import TrainingConfig, train_graph_run


def main() -> None:
    parser = build_parser("Train the semidistributed graph model.")
    parser.add_argument("--dynamic-group", default="era5_full_selected")
    parser.add_argument("--frequency", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--static-group", default=None)
    parser.add_argument("--loss-name", default=None, choices=["rmse", "hydro_skill", "hydro_balance"])
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--graph-batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("train_semidistributed", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    config = TrainingConfig(**read_json(paths.configs_dir / "training_graph.json"))
    config.frequency = args.frequency
    if args.history_length is not None:
        config.history_length = args.history_length
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.limit_train_batches is not None:
        config.limit_train_batches = args.limit_train_batches
    if args.limit_eval_batches is not None:
        config.limit_eval_batches = args.limit_eval_batches
    if args.graph_batch_size is not None:
        config.graph_batch_size = args.graph_batch_size
    if args.device is not None:
        config.device = args.device
    if args.static_group is not None:
        config.static_group = args.static_group
    if args.loss_name is not None:
        config.loss_name = args.loss_name
    outputs = train_graph_run(context, paths, feature_registry, split_spec, args.dynamic_group, config, "semidistributed")
    context.write_manifest({key: str(value) for key, value in outputs.items()})
    context.write_resolved_config({"dynamic_group": args.dynamic_group, "frequency": args.frequency, **config.__dict__})


if __name__ == "__main__":
    main()
