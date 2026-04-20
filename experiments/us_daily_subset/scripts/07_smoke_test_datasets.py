from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_us.config import read_json
from devp_us.datasets import DistributedGraphDataset, LumpedWindowDataset, SemiDistributedGraphDataset
from devp_us.features import FeatureRegistry
from devp_us.models import HybridSeqGConvSeq2One, LumpedSeq2One
from devp_us.paths import DevpUSPaths
from devp_us.registry import build_huc05_registry
from devp_us.runtime import RunContext, build_parser
from devp_us.splits import SplitSpec


def main() -> None:
    parser = build_parser("Smoke-test datasets, split boundaries, and model forward passes.")
    parser.add_argument("--history-length", type=int, default=365)
    args = parser.parse_args()
    paths = DevpUSPaths.from_root(args.root)
    context = RunContext("smoke_test_datasets", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    registry = build_huc05_registry(paths)
    assert len(registry) == 26, "Expected 26 CAMELS-US basins."
    lumped = LumpedWindowDataset(paths.tensor_dir / "lumped", feature_registry, split_spec, "train", "era5_full_selected", "static_base", args.history_length)
    semi = SemiDistributedGraphDataset(paths.tensor_dir / "semidistributed", feature_registry, split_spec, "train", "era5_full_selected", "static_base", args.history_length)
    dist = DistributedGraphDataset(paths.tensor_dir / "distributed", feature_registry, split_spec, "train", "era5_full_selected", "static_base", args.history_length)
    lumped_sample = lumped[0]
    semi_sample = semi[0]
    dist_sample = dist[0]

    assert lumped_sample["dynamic"].shape[0] == args.history_length
    assert semi_sample["dynamic"].shape[0] == args.history_length
    assert dist_sample["dynamic"].shape[0] == args.history_length
    assert lumped_sample["date"] >= split_spec.train_start

    lumped_model = LumpedSeq2One(
        dynamic_dim=lumped_sample["dynamic"].shape[-1],
        static_dim=lumped_sample["static"].shape[-1],
    )
    with torch.no_grad():
        pred = lumped_model(lumped_sample["dynamic"].unsqueeze(0), lumped_sample["static"].unsqueeze(0))
    assert pred.shape == (1,)

    graph_model = HybridSeqGConvSeq2One(
        dynamic_dim=semi_sample["dynamic"].shape[-1],
        static_dim=semi_sample["static"].shape[-1],
    )
    with torch.no_grad():
        pred = graph_model(semi_sample["dynamic"], semi_sample["static"], semi_sample["edge_index"], semi_sample["edge_weight"])
    assert pred.shape[0] == semi_sample["static"].shape[0]
    context.write_manifest(
        {
            "registry_count": int(len(registry)),
            "lumped_train_windows": int(len(lumped)),
            "semi_train_windows": int(len(semi)),
            "dist_train_windows": int(len(dist)),
        }
    )
    context.info("smoke tests passed for lumped, semidistributed, and distributed datasets")


if __name__ == "__main__":
    main()
