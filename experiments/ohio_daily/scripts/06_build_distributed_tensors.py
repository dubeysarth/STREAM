from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.config import read_json
from devp_ohio.datasets import build_distributed_tensors, fit_and_save_scalers
from devp_ohio.features import FeatureRegistry
from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser
from devp_ohio.splits import SplitSpec


def main() -> None:
    parser = build_parser("Build distributed HUC05 tensor records.")
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("build_distributed_tensors", paths, run_name=args.run_name, dry_run=args.dry_run)
    feature_registry = FeatureRegistry.load(read_json(paths.configs_dir / "feature_registry.json"))
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    written = build_distributed_tensors(paths, feature_registry)
    scalers = fit_and_save_scalers(paths, split_spec)
    payload = {"num_records": len(written), "first_record": str(written[0]) if written else None, "scalers": {k: str(v) for k, v in scalers.items()}}
    context.write_manifest(payload)
    context.write_resolved_config(payload)
    context.info("built %s distributed tensor records", len(written))


if __name__ == "__main__":
    main()
