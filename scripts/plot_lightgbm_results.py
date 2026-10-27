#!/usr/bin/env python3
"""Visualize outputs produced by train_lightgbm_demo.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_predictions(predictions: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    for fold, group in predictions.groupby("fold", sort=True):
        axis.scatter(group["target"], group["prediction"], s=36, alpha=0.75, label=f"Fold {fold}")
    minimum = float(min(predictions["target"].min(), predictions["prediction"].min()))
    maximum = float(max(predictions["target"].max(), predictions["prediction"].max()))
    margin = (maximum - minimum) * 0.05 or 1.0
    axis.plot([minimum - margin, maximum + margin], [minimum - margin, maximum + margin], "k--", linewidth=1)
    axis.set(xlabel="Actual target (106)", ylabel="Predicted target (106)", title="LightGBM: actual vs predicted")
    axis.legend(title="Test fold")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_interactive_predictions(predictions: pd.DataFrame, output: Path) -> None:
    """Write a self-contained Plotly chart with CIF filename hover labels."""
    figure = go.Figure()
    for fold, group in predictions.groupby("fold", sort=True):
        filenames = group["source_cif"].fillna("").map(lambda path: Path(str(path)).name)
        paths = group["source_cif"].fillna("")
        residuals = group["prediction"] - group["target"]
        customdata = list(zip(filenames, paths, residuals))
        figure.add_trace(
            go.Scatter(
                x=group["target"],
                y=group["prediction"],
                mode="markers",
                name=f"Fold {fold}",
                customdata=customdata,
                marker={"size": 9, "opacity": 0.78},
                hovertemplate=(
                    "File: %{customdata[0]}<br>"
                    "Path: %{customdata[1]}<br>"
                    "Actual: %{x:.5g}<br>Predicted: %{y:.5g}<br>"
                    "Residual: %{customdata[2]:.5g}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    minimum = float(min(predictions["target"].min(), predictions["prediction"].min()))
    maximum = float(max(predictions["target"].max(), predictions["prediction"].max()))
    margin = (maximum - minimum) * 0.05 or 1.0
    figure.add_trace(
        go.Scatter(
            x=[minimum - margin, maximum + margin],
            y=[minimum - margin, maximum + margin],
            mode="lines",
            name="Perfect prediction",
            line={"color": "black", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title="LightGBM: actual vs predicted (hover points for CIF filename)",
        xaxis_title="Actual target (106)",
        yaxis_title="Predicted target (106)",
        template="plotly_white",
        hovermode="closest",
    )
    figure.write_html(output, include_plotlyjs=True)


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(len(metrics))
    width = 0.36
    axes[0].bar(positions - width / 2, metrics["train_r2"], width, label="Train R²")
    axes[0].bar(positions + width / 2, metrics["test_r2"], width, label="Test R²")
    axes[0].set_xticks(positions, [f"Fold {fold}" for fold in metrics["fold"]])
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² by fold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(positions, metrics["test_rmse"], color="#d95f02")
    axes[1].set_xticks(positions, [f"Fold {fold}" for fold in metrics["fold"]])
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Test RMSE by fold")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_importance(importances: pd.DataFrame, output: Path) -> None:
    data = importances.sort_values("importance_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color="#377eb8", alpha=0.85)
    axis.set(xlabel="Mean LightGBM split importance", title="Feature importance across folds")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LightGBM demo results.")
    parser.add_argument("input_dir", type=Path, help="Directory containing model result CSV files")
    parser.add_argument("-o", "--output-dir", type=Path, help="Output directory; defaults to input_dir")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.input_dir / "test_predictions.csv")
    metrics = pd.read_csv(args.input_dir / "fold_metrics.csv")
    importances = pd.read_csv(args.input_dir / "feature_importance.csv")
    required_prediction_columns = {"fold", "target", "prediction", "source_cif"}
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(f"test_predictions.csv must contain {sorted(required_prediction_columns)}")
    plot_predictions(predictions, output_dir / "actual_vs_predicted.png")
    plot_interactive_predictions(predictions, output_dir / "actual_vs_predicted.html")
    plot_metrics(metrics, output_dir / "fold_metrics.png")
    plot_importance(importances, output_dir / "feature_importance.png")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
