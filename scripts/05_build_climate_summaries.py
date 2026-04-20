"""Stage 05: build climate summaries."""

from stream import ClimateSummaryBuilder
from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("05_build_climate_summaries")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "05_build_climate_summaries",
        config,
        config_path,
        [
            "ClimateSummaryBuilder.run()",
        ],
    )
    if args.dry_run:
        return 0
    ClimateSummaryBuilder(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
