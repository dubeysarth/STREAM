"""Stage 02: attach dynamic and static node features."""

from stream.features import run_feature_assembly

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("02_attach_node_features")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "02_attach_node_features",
        config,
        config_path,
        [
            "Run the configured feature-assembly bundle for dynamic and static inputs.",
            "Persist aligned forcing and context attributes for downstream inventories.",
        ],
    )
    if args.dry_run:
        return 0
    run_feature_assembly(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
