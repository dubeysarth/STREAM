"""Stage 01: curate snapped outlets, maskmaps, and river graphs."""

from stream.curation import run_graph_curation

from _script_common import build_parser, load_config, summarize


def main() -> int:
    parser = build_parser("01_curate_graphs")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    summarize(
        "01_curate_graphs",
        config,
        config_path,
        [
            "Run the configured legacy or adapter-backed graph curation bundle.",
            "Persist snapped outlets, maskmaps, and graph artifacts under one manifest.",
        ],
    )
    if args.dry_run:
        return 0
    run_graph_curation(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
