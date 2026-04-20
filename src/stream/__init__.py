"""STREAM package for reproducible hydrologic curation and benchmarking."""

from .baselines import AblationSuite, DistributedBaselineRunner, LumpedBaselineRunner
from .curation import MaskmapBuilder, RiverGraphBuilder, SnapToGridCurator
from .features import DynamicFeatureAssembler, StaticFeatureAssembler
from .inventory import ClimateSummaryBuilder, LumpedAggregator, MonthlyAggregator, ZarrInventoryBuilder
from .manifests import DatasetManifest
from .paths import StreamPaths
from .reporting import RebuttalAssetBuilder, ValidationBundle

__all__ = [
    "AblationSuite",
    "ClimateSummaryBuilder",
    "DatasetManifest",
    "DistributedBaselineRunner",
    "DynamicFeatureAssembler",
    "LumpedAggregator",
    "LumpedBaselineRunner",
    "MaskmapBuilder",
    "MonthlyAggregator",
    "RebuttalAssetBuilder",
    "RiverGraphBuilder",
    "SnapToGridCurator",
    "StaticFeatureAssembler",
    "StreamPaths",
    "ValidationBundle",
    "ZarrInventoryBuilder",
]
