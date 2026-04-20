"""Path resolution helpers for the STREAM package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "stream").exists() and (candidate / "scripts").exists():
            return candidate
    return start


def _find_upward(start: Path, parts: tuple[str, ...]) -> Path | None:
    for candidate in (start, *start.parents):
        path = candidate.joinpath(*parts)
        if path.exists():
            return path
    return None


@dataclass
class StreamPaths:
    """Describe clone-relative paths for a STREAM checkout."""

    project_root: str
    data_root: str | None = None
    outputs_root: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    def resolve(self) -> dict[str, str]:
        repo_root = _find_repo_root(Path(self.project_root).resolve())
        workspace_root = _find_upward(repo_root, ("reference_materials",))
        workspace_root = workspace_root.parent if workspace_root is not None else repo_root.parent

        data_root = Path(self.data_root).resolve() if self.data_root else (repo_root / "data")
        outputs_root = Path(self.outputs_root).resolve() if self.outputs_root else (repo_root / "outputs")

        resolved = {
            "project_root": str(repo_root),
            "src_root": str(repo_root / "src"),
            "configs_root": str(repo_root / "configs"),
            "scripts_root": str(repo_root / "scripts"),
            "experiments_root": str(repo_root / "experiments"),
            "results_root": str(repo_root / "results"),
            "docs_root": str(repo_root / "docs"),
            "tests_root": str(repo_root / "tests"),
            "data_root": str(data_root),
            "outputs_root": str(outputs_root),
            "reference_stream_root": str(workspace_root / "reference_materials" / "STREAM"),
            "reference_zenodo_root": str(workspace_root / "reference_materials" / "zenodo"),
            "workspace_root": str(workspace_root),
        }
        for key, value in self.extras.items():
            resolved[key] = str((repo_root / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
        return resolved
