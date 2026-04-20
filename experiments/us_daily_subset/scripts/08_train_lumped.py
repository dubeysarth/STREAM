from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.config import read_json
from devp_us.features import FeatureRegistry
from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser
from devp_us.splits import SplitSpec
from devp_us.training import TrainingConfig, train_lumped_run


def main() -> None:
    parser = build_parser("Train the lumped seq-to-1 model.")
    parser.add_argument("--dynamic-group", default="era5_full_selected")
    parser.add_argument("--frequency", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--hucs", nargs="+", default=None, help="Optional HUC-02 subset, e.g. 01 05 10.")
    parser.add_argument("--static-group", default=None)
    parser.add_argument("--loss-name", default=None, choices=["rmse", "hydro_skill", "hydro_balance"])
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("train_lumped", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    config = TrainingConfig(**read_json(paths.configs_dir / "training_lumped.json"))
    config.frequency = args.frequency
    if args.history_length is not None:
        config.history_length = args.history_length
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.limit_train_batches is not None:
        config.limit_train_batches = args.limit_train_batches
    if args.limit_eval_batches is not None:
        config.limit_eval_batches = args.limit_eval_batches
    if args.device is not None:
        config.device = args.device
    if args.static_group is not None:
        config.static_group = args.static_group
    if args.loss_name is not None:
        config.loss_name = args.loss_name
    config.hucs = args.hucs
    outputs = train_lumped_run(context, paths, feature_registry, split_spec, args.dynamic_group, config)
    context.write_manifest({key: str(value) for key, value in outputs.items()})
    context.write_resolved_config({"dynamic_group": args.dynamic_group, "frequency": args.frequency, **config.__dict__})


if __name__ == "__main__":
    main()
