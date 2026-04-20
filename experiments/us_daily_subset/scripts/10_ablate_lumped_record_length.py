from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.config import read_json
from devp_us.features import FeatureRegistry
from devp_us.paths import DevpUSPaths
from devp_us.runtime import RunContext, build_parser
from devp_us.splits import SplitSpec
from devp_us.training import TrainingConfig, train_lumped_run


def main() -> None:
    parser = build_parser("Run lumped history-length ablations.")
    parser.add_argument("--loss-name", default=None, choices=["rmse", "hydro_skill", "hydro_balance"])
    parser.add_argument("--static-group", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("ablate_lumped_record_length", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    base_config = TrainingConfig(**read_json(paths.configs_dir / "training_lumped.json"))
    if args.max_epochs is not None:
        base_config.max_epochs = args.max_epochs
    if args.limit_train_batches is not None:
        base_config.limit_train_batches = args.limit_train_batches
    if args.limit_eval_batches is not None:
        base_config.limit_eval_batches = args.limit_eval_batches
    if args.loss_name is not None:
        base_config.loss_name = args.loss_name
    if args.static_group is not None:
        base_config.static_group = args.static_group
    if args.device is not None:
        base_config.device = args.device
    lengths = [30, 90, 180, 365]
    rows = []
    for history_length in lengths:
        config = TrainingConfig(**base_config.__dict__)
        config.history_length = history_length
        run_context = RunContext(f"ablate_lumped_h{history_length}", paths)
        outputs = train_lumped_run(run_context, paths, feature_registry, split_spec, "era5_full_selected", config)
        summary = pd.read_csv(outputs["summary"])
        test_row = summary[summary["split"] == "test"].iloc[0].to_dict()
        test_row["history_length"] = history_length
        rows.append(test_row)
    out = paths.metrics_dir / f"{context.run_id}_record_length_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    context.write_manifest({"summary_csv": str(out), "history_lengths": lengths})


if __name__ == "__main__":
    main()
