"""Stage 11: render validation bundle."""

from stream import ValidationBundle

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("11_run_validation_bundle")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "11_run_validation_bundle",
        config,
        config_path,
        ["ValidationBundle.render()"],
    )
    if args.dry_run:
        return 0
    ValidationBundle(config).render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
