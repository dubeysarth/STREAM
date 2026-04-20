"""Stage 12: build rebuttal assets."""

from stream import RebuttalAssetBuilder

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("12_build_rebuttal_assets")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "12_build_rebuttal_assets",
        config,
        config_path,
        ["RebuttalAssetBuilder.render()"],
    )
    if args.dry_run:
        return 0
    RebuttalAssetBuilder(config).render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
