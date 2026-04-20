from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from devp_ohio.config import read_json, write_json
from devp_ohio.manifests import SourceManifest
from devp_ohio.paths import DevpOhioPaths
from devp_ohio.registry import build_huc05_registry
from devp_ohio.runtime import RunContext, build_parser
from devp_ohio.splits import SplitSpec


def main() -> None:
    parser = build_parser("Audit Ohio HUC05 sources and split availability.")
    args = parser.parse_args()
    paths = DevpOhioPaths.from_root(args.root)
    split_spec = SplitSpec(**read_json(paths.configs_dir / "split_spec.json"))
    context = RunContext("audit_sources", paths, run_name=args.run_name, dry_run=args.dry_run)
    source_manifest = SourceManifest(paths, split_spec)
    source_manifest.verify()
    manifest = source_manifest.resolve()
    registry = build_huc05_registry(paths)
    rows = []
    for basin_id in registry["gauge_id"].tolist():
        ds = xr.open_zarr(paths.source_ohio_root / "inventory" / "05" / f"{basin_id}.zarr").sel(
            time=slice(split_spec.analysis_start, split_spec.analysis_end)
        )
        rows.append(
            {
                "gauge_id": basin_id,
                "n_days": int(ds.sizes["time"]),
                "n_nodes": int(ds.sizes["idx"]),
                "target_nan_frac": float(ds["dynamic_GloFAS_discharge_mm"].isnull().mean().compute().item()),
                "era5_nan_frac": float(ds["dynamic_ERA5_total_precipitation"].isnull().mean().compute().item()),
                "usgs_nan_frac": float(ds["outlet_USGS_Q_mm"].isnull().mean().compute().item()),
            }
        )
    audit_df = pd.DataFrame(rows)
    audit_path = paths.manifests_dir / "source_audit_huc05.csv"
    audit_df.to_csv(audit_path, index=False)
    summary = {
        "num_basins": int(len(audit_df)),
        "analysis_days": int(audit_df["n_days"].iloc[0]),
        "target_nan_max": float(audit_df["target_nan_frac"].max()),
        "era5_nan_max": float(audit_df["era5_nan_frac"].max()),
        "usgs_nan_max": float(audit_df["usgs_nan_frac"].max()),
        "source_manifest": manifest,
        "audit_csv": str(audit_path),
    }
    write_json(paths.manifests_dir / "source_audit_summary.json", summary)
    context.write_manifest(summary)
    context.write_resolved_config(summary)
    context.info("audited %s basins with %s days each", len(audit_df), audit_df["n_days"].iloc[0])


if __name__ == "__main__":
    main()
