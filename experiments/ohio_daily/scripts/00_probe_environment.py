from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.paths import DevpOhioPaths
from devp_ohio.runtime import RunContext, build_parser


def main() -> None:
    parser = build_parser("Probe the Ohio pilot runtime environment.")
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    context = RunContext("probe_environment", paths, run_name=args.run_name, dry_run=args.dry_run)
    modules = ["torch", "xarray", "pandas", "numpy", "matplotlib", "seaborn", "torch_geometric"]
    results = {}
    for name in modules:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "n/a")
        results[name] = version
        context.info("module=%s version=%s", name, version)
    try:
        module = importlib.import_module("torch_geometric_temporal")
        results["torch_geometric_temporal"] = getattr(module, "__version__", "n/a")
        results["graph_runtime"] = "external_torch_geometric_temporal"
    except ModuleNotFoundError:
        results["torch_geometric_temporal"] = "missing"
        results["graph_runtime"] = "local_devp_ohio_fallback"
        context.info("torch_geometric_temporal is missing; local devp_Ohio GConvLSTM fallback will be used.")
    manifest = {
        "environment_name": "main_latest",
        "imports": results,
        "paths_checked": {
            "ohio_root": str(paths.source_ohio_root),
            "lisflood_root": str(paths.lisflood_root),
        },
    }
    context.write_manifest(manifest)
    context.write_resolved_config(manifest)


if __name__ == "__main__":
    main()
