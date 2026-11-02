#!/usr/bin/env python3
"""Run a leakage-safe Random Forest 5-fold regression demo."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sib_features.extractor import SUPPORTED_SPACE_GROUP_NUMBERS

from train_lightgbm_demo import (
    DEFAULT_FEATURES,
    DEFAULT_SPLIT_ROOT,
    attach_folds,
    parse_feature_selection,
    target_column,
)

DEFAULT_OUTPUT = Path("ml_output/random_forest_demo")
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": -1,
}


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("random_forest_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_tune_value(name: str, value: str):
    if name == "bootstrap":
        normalized = value.lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("expected true/false")
        return normalized in {"true", "1", "yes"}
    if name in {"criterion"}:
        if value not in {"squared_error", "absolute_error", "friedman_mse", "poisson"}:
            raise ValueError("unsupported criterion")
        return value
    if name == "max_depth":
        depth = None if value.lower() in {"none", "null"} else int(value)
        if depth is not None and depth < 1:
            raise ValueError("max_depth must be >= 1 or none")
        return depth
    if name == "max_features":
        if value in {"sqrt", "log2"}:
            return value
        try:
            number = int(value)
        except ValueError:
            number = float(value)
        if isinstance(number, int):
            if number < 1:
                raise ValueError("integer max_features must be >= 1")
        elif not 0 < number <= 1:
            raise ValueError("fractional max_features must be in (0, 1]")
        return number
    if name in {"min_samples_split", "min_samples_leaf"}:
        try:
            number = int(value)
        except ValueError:
            number = float(value)
        if not math.isfinite(number):
            raise ValueError("must be finite")
        minimum = 2 if name == "min_samples_split" else 1
        if isinstance(number, int):
            if number < minimum:
                raise ValueError(f"integer value must be >= {minimum}")
        elif not 0 < number <= 1:
            raise ValueError("fractional value must be in (0, 1]")
        return number
    if name in {"n_estimators", "n_jobs", "random_state"}:
        number = int(value)
        if name == "n_estimators" and number < 1:
            raise ValueError("n_estimators must be >= 1")
        if name == "n_jobs" and number == 0:
            raise ValueError("n_jobs cannot be 0")
        if name == "random_state" and number < 0:
            raise ValueError("random_state must be >= 0")
        return number
    raise ValueError(f"Unsupported Random Forest parameter {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a leakage-safe Random Forest 5-fold demo model.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="features_all.csv from feature extraction")
    parser.add_argument("--add-features", nargs="+", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--exclude-features", nargs="+", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--target", help="Target column (defaults to the only column whose name starts with '106_')")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--data-root", type=Path, help="CIF root used to create the feature table")
    parser.add_argument(
        "--tune", nargs="+", metavar="PARAM=VALUE",
        help="Override Random Forest parameters, e.g. --tune n_estimators=500 max_depth=12",
    )
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
    empty_folds = fold_counts[fold_counts == 0].index.tolist()
    if empty_folds:
        raise ValueError(f"No usable test rows remain in fold(s) {empty_folds}")
    logger.info("Features: %s", ", ".join(selected_features))
    logger.info("Target: %s", target)
    logger.info("Split summary: %d samples, %d folds", summary["n_samples"], len(summary["folds"]))

    params = DEFAULT_PARAMS.copy()
    if args.tune:
        for item in args.tune:
            if "=" not in item:
                raise ValueError(f"Invalid --tune value {item!r}; use PARAM=VALUE")
            name, value = item.split("=", 1)
            if name not in params:
                raise ValueError(f"Unknown Random Forest parameter {name!r}; choose from {sorted(params)}")
            try:
                params[name] = parse_tune_value(name, value)
            except ValueError as exc:
                raise ValueError(f"Invalid value {value!r} for Random Forest parameter {name!r}: {exc}") from exc
    logger.info("Random Forest parameters: %s", params)

    fold_results: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []
    feature_importances: list[np.ndarray] = []
    for fold in folds:
        train = table[table["fold"] != fold]
        test = table[table["fold"] == fold]
        model = RandomForestRegressor(**params)
        model.fit(train[selected_features], train[target])
        feature_importances.append(np.asarray(model.feature_importances_, dtype=float))
        train_pred = model.predict(train[selected_features])
        test_pred = model.predict(test[selected_features])
        result = {
            "fold": fold,
            "train_r2": float(r2_score(train[target], train_pred)),
            "test_r2": float(r2_score(test[target], test_pred)),
            "test_rmse": float(math.sqrt(mean_squared_error(test[target], test_pred))),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "parameters": json.dumps(params, sort_keys=True),
        }
        fold_results.append(result)
        predictions.append(pd.DataFrame({
            "source_cif": test["source_cif"].values,
            "fold": fold,
            "target": test[target].values,
            "prediction": test_pred,
        }))
        logger.info(
            "Fold %d | train R2=%.4f | test R2=%.4f | test RMSE=%.4f | train=%d test=%d",
            fold, result["train_r2"], result["test_r2"], result["test_rmse"], result["n_train"], result["n_test"],
        )

    aggregate: dict[str, float] = {}
    for name in ("train_r2", "test_r2", "test_rmse"):
        values = np.asarray([float(row[name]) for row in fold_results])
        aggregate[f"mean_{name}"] = float(np.mean(values))
        aggregate[f"std_{name}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        logger.info("%s: %.4f ± %.4f", name, aggregate[f"mean_{name}"], aggregate[f"std_{name}"])
    elapsed = time.perf_counter() - started
    logger.info("Total elapsed time: %.3f s", elapsed)

    pd.DataFrame(fold_results).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame({
        "feature": selected_features,
        "importance_mean": np.mean(feature_importances, axis=0),
        "importance_std": np.std(feature_importances, axis=0),
    }).sort_values("importance_mean", ascending=False).to_csv(args.output_dir / "feature_importance.csv", index=False)
    report = {
        "features": selected_features,
        "target": target,
        "parameters": params,
        **aggregate,
        "total_elapsed_seconds": elapsed,
        "fold_metrics": fold_results,
        "n_usable_rows": int(len(table)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
