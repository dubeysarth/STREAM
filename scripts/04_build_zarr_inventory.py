"""Stage 04: build zarr inventory."""

from stream import ZarrInventoryBuilder

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("04_build_zarr_inventory")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "04_build_zarr_inventory",
        config,
        config_path,
        ["ZarrInventoryBuilder.run()"],
    )
    if args.dry_run:
        return 0
    ZarrInventoryBuilder(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
