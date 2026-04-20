from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _find_workspace(base: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = base / candidate
        if (path / "src").exists() and (path / "configs").exists():
            return path
    return base / candidates[0]


@dataclass(frozen=True)
class DevpUSPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DevpUSPaths":
        base = Path(root).resolve()
        return cls(
            _find_workspace(
                base,
                "experiments/us_daily_subset",
                "devp_US",
            )
        )

    @property
    def package_root(self) -> Path:
        return self.root

    @property
    def src_dir(self) -> Path:
        return self.root / "src"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def logs_dir(self) -> Path:
        return self.runs_dir / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.runs_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.runs_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.runs_dir / "predictions"

    @property
    def histories_dir(self) -> Path:
        return self.runs_dir / "histories"

    @property
    def manifests_runtime_dir(self) -> Path:
        return self.runs_dir / "manifests"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def summaries_dir(self) -> Path:
        return self.reports_dir / "summaries"

    @property
    def tensor_dir(self) -> Path:
        return self.data_dir / "tensors"

    @property
    def human_use_dir(self) -> Path:
        return self.data_dir / "human_use"

    @property
    def source_us_root(self) -> Path:
        return Path("/data/sarth/rootdir/datadir/data/devp_datasets/03min_GloFAS_CAMELS-US")

    @property
    def lisflood_root(self) -> Path:
        return Path("/data/sarth/rootdir/datadir/data/raw/GloFAS/LISFLOOD_Parameter_Maps")

    @property
    def gdw_root(self) -> Path:
        return Path("/data/sarth/rootdir/datadir/data/raw/Global_Dam_Watch")

    def tensor_regime_dir(self, regime: str, frequency: str = "daily") -> Path:
        if frequency == "daily":
            return self.tensor_dir / regime
        return self.tensor_dir / f"{regime}_{frequency}"

    def ensure(self) -> None:
        for path in [
            self.manifests_dir,
            self.data_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.histories_dir,
            self.manifests_runtime_dir,
            self.figures_dir,
            self.summaries_dir,
            self.tensor_dir,
            self.human_use_dir,
            self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
