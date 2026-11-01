#!/usr/bin/env python3
"""Run a leakage-safe Gaussian Process 5-fold regression demo."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sib_features.extractor import SUPPORTED_SPACE_GROUP_NUMBERS

from train_lightgbm_demo import (
    DEFAULT_FEATURES,
    DEFAULT_SPLIT_ROOT,
    attach_folds,
    parse_feature_selection,
    target_column,
)

DEFAULT_OUTPUT = Path("ml_output/gaussian_process_demo")


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gaussian_process_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def uncertainty_error_correlation(errors: np.ndarray, std: np.ndarray) -> float:
    if len(errors) < 2 or np.std(errors) == 0 or np.std(std) == 0:
        return float("nan")
    return float(pearsonr(errors, std).statistic)


def uncertainty_error_rank_correlation(errors: np.ndarray, std: np.ndarray) -> float:
    if len(errors) < 2 or np.std(errors) == 0 or np.std(std) == 0:
        return float("nan")
    return float(spearmanr(errors, std).statistic)


def save_result_plots(
    output_dir: Path,
    predictions: pd.DataFrame,
    feature_columns: list[str],
    feature_table: pd.DataFrame,
    fitted_folds: list[tuple[int, GaussianProcessRegressor, StandardScaler, pd.DataFrame]],
) -> list[Path]:
    plot_paths: list[Path] = []
    actual = predictions["target"].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=float)
    predicted_std = predictions["prediction_std"].to_numpy(dtype=float)
    low = min(float(np.min(actual)), float(np.min(predicted)))
    high = max(float(np.max(actual)), float(np.max(predicted)))
    padding = (high - low) * 0.05 or 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    for fold, group in predictions.groupby("fold", sort=True):
        ax.errorbar(
            group["target"], group["prediction"], yerr=1.96 * group["prediction_std"],
            fmt="o", capsize=2, alpha=0.75, label=f"Fold {fold}",
        )
    ax.plot([low - padding, high + padding], [low - padding, high + padding], "k--", label="Parity")
    ax.set(xlabel="Actual value", ylabel="Predicted value", title="GPR parity plot (95% predictive interval)")
    ax.legend()
    fig.tight_layout()
    parity_path = output_dir / "parity_plot.png"
    fig.savefig(parity_path, dpi=160)
    plt.close(fig)
    plot_paths.append(parity_path)

    order = np.argsort(predicted)
    x_rank = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        x_rank, predicted[order] - 1.96 * predicted_std[order],
        predicted[order] + 1.96 * predicted_std[order], alpha=0.25,
        label="95% predictive interval",
    )
    ax.plot(x_rank, predicted[order], color="C0", label="Prediction")
    ax.scatter(x_rank, actual[order], s=18, alpha=0.7, color="C1", label="Actual value")
    ax.set(xlabel="Test samples (sorted by prediction)", ylabel="Target value", title="GPR uncertainty ribbon")
    ax.legend()
    fig.tight_layout()
    ribbon_path = output_dir / "uncertainty_ribbon.png"
    fig.savefig(ribbon_path, dpi=160)
    plt.close(fig)
    plot_paths.append(ribbon_path)

    ncols = 3
    nrows = math.ceil(len(feature_columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.7 * nrows), squeeze=False)
    for index, feature in enumerate(feature_columns):
        ax = axes.flat[index]
        values = pd.to_numeric(feature_table[feature], errors="coerce").dropna().to_numpy(dtype=float)
        x_min, x_max = np.quantile(values, [0.05, 0.95])
        if x_min == x_max:
            x_min, x_max = float(np.min(values)), float(np.max(values))
        if x_min == x_max:
            ax.text(0.5, 0.5, "Feature is constant", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(feature)
            continue
        grid = np.linspace(x_min, x_max, 25)
        fold_means: list[np.ndarray] = []
        fold_variances: list[np.ndarray] = []
        for _, model, scaler, test_frame in fitted_folds:
            background = test_frame[feature_columns]
            if len(background) > 100:
                background = background.sample(n=100, random_state=42)
            curve_mean = []
            curve_variance = []
            for value in grid:
                changed = background.copy()
                changed.loc[:, feature] = value
                transformed = scaler.transform(changed)
                curve_prediction, curve_std = model.predict(transformed, return_std=True)
                curve_mean.append(float(np.mean(curve_prediction)))
                curve_variance.append(float(np.mean(curve_std**2) + np.var(curve_prediction)))
            fold_means.append(np.asarray(curve_mean))
            fold_variances.append(np.asarray(curve_variance))
        fold_means_array = np.vstack(fold_means)
        mean_curve = np.mean(fold_means_array, axis=0)
        total_variance = np.mean(
            np.vstack(fold_variances) + (fold_means_array - mean_curve) ** 2,
            axis=0,
        )
        interval = 1.96 * np.sqrt(np.maximum(total_variance, 0))
        ax.plot(grid, mean_curve, color="C0")
        ax.fill_between(grid, mean_curve - interval, mean_curve + interval, color="C0", alpha=0.2)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial dependence")
    for index in range(len(feature_columns), nrows * ncols):
        axes.flat[index].set_visible(False)
    fig.suptitle("GPR 1D partial dependence (95% predictive interval)")
    fig.tight_layout()
    pdp_path = output_dir / "partial_dependence.png"
    fig.savefig(pdp_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(pdp_path)
    return plot_paths


def display_plots_in_colab(plot_paths: list[Path]) -> None:
    try:
        import google.colab  # noqa: F401
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return
    shell = get_ipython()
    if shell is None or not hasattr(shell, "kernel"):
        return
    if not os.environ.get("COLAB_RELEASE_TAG") and not os.path.exists("/content"):
        return
    print("Displaying GPR plots in the Colab output:")
    for path in plot_paths:
        print(path.name)
        display(Image(filename=str(path), width=500))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a leakage-safe Gaussian Process demo model.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="features_all.csv from feature extraction")
    parser.add_argument("--add-features", nargs="+", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--exclude-features", nargs="+", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--target", help="Target column (defaults to the only column whose name starts with '106_')")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--data-root", type=Path, help="CIF root used to create the feature table")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir / "train.log")
    started = time.perf_counter()

    summary_path = args.split_root / "summary.json"
    membership_path = args.split_root / "fold_membership.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    membership = pd.read_csv(membership_path)
    table = pd.read_csv(args.features)
    target = args.target or target_column(table)
    if target not in table.columns:
        raise ValueError(f"Target column {target!r} is missing from the feature CSV")
    added = [feature for group in args.add_features for feature in group]
    excluded = [feature for group in args.exclude_features for feature in group]
    selected_features = parse_feature_selection(table.columns.tolist(), added, excluded)
    if target in selected_features:
        raise ValueError(f"Target column {target!r} cannot also be selected as an input feature")

    required_sg = {"source_cif_space_group_number", "source_json_space_group_number"}
    missing_sg = required_sg - set(table.columns)
    if missing_sg:
        raise ValueError(f"Feature CSV lacks source CIF/JSON space-group columns: {sorted(missing_sg)}")
    cif_sg = pd.to_numeric(table["source_cif_space_group_number"], errors="coerce")
    json_sg = pd.to_numeric(table["source_json_space_group_number"], errors="coerce")
    supported = cif_sg.isin(SUPPORTED_SPACE_GROUP_NUMBERS) & (
        json_sg.isna() | json_sg.isin(SUPPORTED_SPACE_GROUP_NUMBERS)
    )
    logger.info("Excluded %d rows with unsupported or unknown space group", int((~supported).sum()))
    table = attach_folds(table.loc[supported].copy(), membership, args.data_root)
    before = len(table)
    table = table.dropna(subset=selected_features + [target]).copy()
    logger.info("Dataset: %d rows usable (%d dropped due to missing selected values)", len(table), before - len(table))
    folds = sorted(int(value) for value in membership["fold"].unique())
    fold_counts = table["fold"].value_counts().reindex(folds, fill_value=0)
    logger.info("Usable rows per test fold: %s", fold_counts.to_dict())
    empty = fold_counts[fold_counts == 0].index.tolist()
    if empty:
        raise ValueError(f"No usable test rows remain in fold(s) {empty}")
    logger.info("Features: %s", ", ".join(selected_features))
    logger.info("Target: %s", target)
    logger.info("Split summary: %d samples, %d folds", summary["n_samples"], len(summary["folds"]))
    model_params = {
        "kernel": "ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))",
        "n_restarts_optimizer": 10,
        "normalize_y": True,
        "random_state": 42,
    }
    logger.info("Model parameters: %s", model_params)

    fold_results: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []
    fitted_folds: list[tuple[int, GaussianProcessRegressor, StandardScaler, pd.DataFrame]] = []
    for fold in folds:
        train = table[table["fold"] != fold]
        test = table[table["fold"] == fold]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[selected_features])
        x_test = scaler.transform(test[selected_features])
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.ones(len(selected_features)), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42
        )
        model.fit(x_train, train[target].to_numpy())
        fitted_folds.append((fold, model, scaler, test[selected_features].copy()))
        train_pred = model.predict(x_train)
        test_pred, test_std = model.predict(x_test, return_std=True)
        errors = np.abs(test[target].to_numpy() - test_pred)
        result = {
            "fold": fold,
            "train_r2": float(r2_score(train[target], train_pred)),
            "test_r2": float(r2_score(test[target], test_pred)),
            "test_rmse": float(math.sqrt(mean_squared_error(test[target], test_pred))),
            "abs_error_std_pearson_r": uncertainty_error_correlation(errors, test_std),
            "abs_error_std_spearman_rho": uncertainty_error_rank_correlation(errors, test_std),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "optimized_kernel": str(model.kernel_),
        }
        fold_results.append(result)
        predictions.append(pd.DataFrame({
            "source_cif": test["source_cif"].values,
            "fold": fold,
            "target": test[target].values,
            "prediction": test_pred,
            "prediction_std": test_std,
            "absolute_error": errors,
        }))
        logger.info(
            "Fold %d | train R2=%.4f | test R2=%.4f | test RMSE=%.4f | |error|-std Pearson r=%s | Spearman rho=%s | train=%d test=%d",
            fold, result["train_r2"], result["test_r2"], result["test_rmse"],
            f"{result['abs_error_std_pearson_r']:.4f}" if math.isfinite(result["abs_error_std_pearson_r"]) else "undefined",
            f"{result['abs_error_std_spearman_rho']:.4f}" if math.isfinite(result["abs_error_std_spearman_rho"]) else "undefined",
            result["n_train"], result["n_test"],
        )
        logger.info("Fold %d optimized kernel: %s", fold, model.kernel_)

    metric_names = ("train_r2", "test_r2", "test_rmse")
    aggregate = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in fold_results])
        aggregate[f"mean_{name}"] = float(np.mean(values))
        aggregate[f"std_{name}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        logger.info("%s: %.4f ± %.4f", name, aggregate[f"mean_{name}"], aggregate[f"std_{name}"])
    elapsed = time.perf_counter() - started
    logger.info("Total elapsed time: %.3f s", elapsed)

    pd.DataFrame(fold_results).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    plot_paths = save_result_plots(args.output_dir, all_predictions, selected_features, table, fitted_folds)
    for path in plot_paths:
        logger.info("Saved plot: %s", path)
    display_plots_in_colab(plot_paths)
    report = {
        "features": selected_features,
        "target": target,
        "parameters": model_params,
        **aggregate,
        "total_elapsed_seconds": elapsed,
        "fold_metrics": fold_results,
        "n_usable_rows": int(len(table)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
