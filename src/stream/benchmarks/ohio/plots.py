from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import MetricBundle
from .paths import DevpOhioPaths


class PlotBundle:
    """Conference-quality figure outputs for Ohio training runs."""

    @staticmethod
    def _style() -> None:
        sns.set_theme(
            style="whitegrid",
            context="talk",
            font_scale=0.9,
            rc={
                "axes.facecolor": "#fbfbfc",
                "figure.facecolor": "white",
                "axes.edgecolor": "#c8c8cf",
                "grid.color": "#e3e4ea",
                "axes.labelcolor": "#1f2430",
                "xtick.color": "#1f2430",
                "ytick.color": "#1f2430",
            },
        )

    @staticmethod
    def _outlet_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
        outlet = predictions[predictions["is_outlet"]].copy()
        if outlet.empty:
            outlet = predictions.copy()
        outlet["date"] = pd.to_datetime(outlet["date"])
        return outlet.sort_values(["split", "basin_id", "date"])

    @staticmethod
    def _select_basin_triplet(outlet: pd.DataFrame) -> list[str]:
        test_df = outlet[outlet["split"] == "test"].copy()
        if test_df.empty:
            test_df = outlet
        basin_scores = test_df.groupby("basin_id", sort=True).apply(
            lambda df: MetricBundle.compute(df["observed"].to_numpy(), df["predicted"].to_numpy())["NSE"]
        )
        basin_scores = basin_scores.sort_values()
        chosen: list[str] = []
        if len(basin_scores) >= 1:
            chosen.append(basin_scores.index[-1])
        if len(basin_scores) >= 3:
            chosen.append(basin_scores.index[len(basin_scores) // 2])
        if len(basin_scores) >= 2:
            chosen.append(basin_scores.index[0])
        return list(dict.fromkeys(chosen))

    @staticmethod
    def _flow_duration_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranked = np.sort(np.clip(np.asarray(values, dtype=float), a_min=0.0, a_max=None))[::-1]
        exceedance = np.linspace(0.0, 100.0, len(ranked), endpoint=False)
        return exceedance, ranked

    @staticmethod
    def render(paths: DevpOhioPaths, run_id: str, predictions_path: Path, history_path: Path) -> list[Path]:
        PlotBundle._style()
        paths.figures_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.read_csv(predictions_path)
        history = pd.read_csv(history_path)
        outputs: list[Path] = []
        outlet = PlotBundle._outlet_predictions(predictions)
        test_outlet = outlet[outlet["split"] == "test"].copy()
        if test_outlet.empty:
            test_outlet = outlet.copy()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        curve_specs = [
            ("train_loss", "Train Loss", "#355070"),
            ("val_loss", "Validation Loss", "#6d597a"),
            ("val_nse", "Validation NSE", "#b56576"),
        ]
        for ax, (column, title, color) in zip(axes, curve_specs):
            if column in history:
                ax.plot(history["epoch"], history[column], color=color, linewidth=2.3)
                ax.scatter(history["epoch"], history[column], color=color, s=18, zorder=3)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
        axes[0].set_ylabel("Objective")
        out = paths.figures_dir / f"{run_id}_standard_training.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)

        basin_triplet = PlotBundle._select_basin_triplet(outlet)
        if basin_triplet:
            fig, axes = plt.subplots(len(basin_triplet), 1, figsize=(15, 3.6 * len(basin_triplet)), constrained_layout=True)
            if len(basin_triplet) == 1:
                axes = [axes]
            labels = ["Best basin", "Median basin", "Worst basin"]
            for ax, basin_id, label in zip(axes, basin_triplet, labels):
                subset = test_outlet[test_outlet["basin_id"] == basin_id].copy().tail(365)
                skill = MetricBundle.compute(subset["observed"].to_numpy(), subset["predicted"].to_numpy())
                ax.plot(subset["date"], subset["observed"], color="#1d3557", linewidth=2.2, label="Observed")
                ax.plot(subset["date"], subset["predicted"], color="#e76f51", linewidth=2.0, label="Predicted")
                ax.set_title(f"{label} | Basin {basin_id} | NSE={skill['NSE']:.3f}, KGE={skill['KGE']:.3f}")
                ax.set_ylabel("Discharge")
                ax.legend(frameon=False, ncol=2, loc="upper right")
            axes[-1].set_xlabel("Date")
            out = paths.figures_dir / f"{run_id}_standard_hydrographs.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out)

        if not test_outlet.empty:
            per_basin = []
            for basin_id, basin_df in test_outlet.groupby("basin_id", sort=True):
                metrics = MetricBundle.compute(basin_df["observed"].to_numpy(), basin_df["predicted"].to_numpy())
                metrics["basin_id"] = basin_id
                per_basin.append(metrics)
            per_basin_df = pd.DataFrame(per_basin).sort_values("NSE", ascending=True)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

            scatter_df = test_outlet.sample(n=min(len(test_outlet), 3000), random_state=42)
            sns.scatterplot(
                data=scatter_df,
                x="observed",
                y="predicted",
                s=16,
                alpha=0.5,
                color="#457b9d",
                edgecolor=None,
                ax=axes[0, 0],
            )
            max_val = float(max(scatter_df["observed"].max(), scatter_df["predicted"].max(), 1.0))
            axes[0, 0].plot([0, max_val], [0, max_val], linestyle="--", color="#d62828", linewidth=1.6)
            axes[0, 0].set_title("Observed vs Predicted")
            axes[0, 0].set_xlabel("Observed")
            axes[0, 0].set_ylabel("Predicted")

            fdc_x_obs, fdc_obs = PlotBundle._flow_duration_curve(test_outlet["observed"].to_numpy())
            fdc_x_pred, fdc_pred = PlotBundle._flow_duration_curve(test_outlet["predicted"].to_numpy())
            axes[0, 1].plot(fdc_x_obs, fdc_obs, color="#1d3557", linewidth=2.1, label="Observed")
            axes[0, 1].plot(fdc_x_pred, fdc_pred, color="#e76f51", linewidth=2.1, label="Predicted")
            axes[0, 1].set_yscale("log")
            axes[0, 1].set_title("Flow-Duration Curve")
            axes[0, 1].set_xlabel("Exceedance probability (%)")
            axes[0, 1].set_ylabel("Discharge")
            axes[0, 1].legend(frameon=False)

            sns.barplot(
                data=per_basin_df,
                x="NSE",
                y="basin_id",
                color="#4d908e",
                orient="h",
                ax=axes[1, 0],
            )
            axes[1, 0].set_title("Per-Basin NSE")
            axes[1, 0].set_xlabel("NSE")
            axes[1, 0].set_ylabel("Basin")

            sns.barplot(
                data=per_basin_df.sort_values("KGE", ascending=True),
                x="KGE",
                y="basin_id",
                color="#577590",
                orient="h",
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("Per-Basin KGE")
            axes[1, 1].set_xlabel("KGE")
            axes[1, 1].set_ylabel("Basin")

            out = paths.figures_dir / f"{run_id}_standard_skill.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out)

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)
            axes[0, 0].plot(history["epoch"], history["train_loss"], color="#355070", linewidth=2.0, label="Train")
            if "val_loss" in history:
                axes[0, 0].plot(history["epoch"], history["val_loss"], color="#b56576", linewidth=2.0, label="Val")
            axes[0, 0].set_title("Optimization")
            axes[0, 0].legend(frameon=False)

            axes[0, 1].plot(fdc_x_obs, fdc_obs, color="#1d3557", linewidth=2.0)
            axes[0, 1].plot(fdc_x_pred, fdc_pred, color="#e76f51", linewidth=2.0)
            axes[0, 1].set_yscale("log")
            axes[0, 1].set_title("FDC")

            sns.scatterplot(
                data=scatter_df,
                x="observed",
                y="predicted",
                s=12,
                alpha=0.45,
                color="#457b9d",
                edgecolor=None,
                ax=axes[1, 0],
            )
            axes[1, 0].plot([0, max_val], [0, max_val], linestyle="--", color="#d62828", linewidth=1.4)
            axes[1, 0].set_title("Fit")

            summary_metrics = MetricBundle.compute(test_outlet["observed"].to_numpy(), test_outlet["predicted"].to_numpy())
            metric_frame = pd.DataFrame(
                {
                    "metric": ["RMSE", "NSE", "PBIAS", "KGE"],
                    "value": [
                        summary_metrics["RMSE"],
                        summary_metrics["NSE"],
                        summary_metrics["PBIAS"],
                        summary_metrics["KGE"],
                    ],
                }
            )
            sns.barplot(
                data=metric_frame,
                x="metric",
                y="value",
                hue="metric",
                palette=["#577590", "#4d908e", "#f8961e", "#277da1"],
                dodge=False,
                legend=False,
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("Test Summary")
            axes[1, 1].set_xlabel("")

            out = paths.figures_dir / f"{run_id}_condensed_summary.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out)

        return outputs
