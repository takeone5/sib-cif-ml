#!/usr/bin/env python3
"""Run a leakage-safe SVR 5-fold regression demo."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sib_features.extractor import SUPPORTED_SPACE_GROUP_NUMBERS

from train_lightgbm_demo import (
    DEFAULT_FEATURES,
    DEFAULT_SPLIT_ROOT,
    attach_folds,
    parse_feature_selection,
    target_column,
)

DEFAULT_OUTPUT = Path("ml_output/svr_demo")
BASELINE_PARAMS = {"kernel": "rbf", "C": 10.0, "epsilon": 0.03, "gamma": "scale"}


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("svr_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a leakage-safe SVR 5-fold demo model.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="features_all.csv from feature extraction")
    parser.add_argument("--add-features", nargs="+", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--exclude-features", nargs="+", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--target", help="Target column (defaults to the only column whose name starts with '106_')")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--data-root", type=Path, help="CIF root used to create the feature table")
    parser.add_argument(
        "--tune", nargs="+", metavar="PARAM=VALUE",
        help="Override SVR parameters, e.g. --tune C=100 epsilon=0.1 gamma=0.01",
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
    empty = fold_counts[fold_counts == 0].index.tolist()
    if empty:
        raise ValueError(f"No usable test rows remain in fold(s) {empty}")
    logger.info("Features: %s", ", ".join(selected_features))
    logger.info("Target: %s", target)
    logger.info("Split summary: %d samples, %d folds", summary["n_samples"], len(summary["folds"]))
    params = BASELINE_PARAMS.copy()
    if args.tune:
        valid_params = {"C", "epsilon", "gamma", "kernel"}
        for item in args.tune:
            if "=" not in item:
                raise ValueError(f"Invalid --tune value {item!r}; use PARAM=VALUE")
            name, value = item.split("=", 1)
            if name not in valid_params:
                raise ValueError(f"Unknown SVR parameter {name!r}; choose from {sorted(valid_params)}")
            if name in {"C", "epsilon"}:
                try:
                    params[name] = float(value)
                except ValueError as exc:
                    raise ValueError(f"SVR parameter {name} must be numeric") from exc
            elif name == "gamma" and value not in {"scale", "auto"}:
                try:
                    params[name] = float(value)
                except ValueError as exc:
                    raise ValueError("SVR parameter gamma must be 'scale', 'auto', or numeric") from exc
            else:
                params[name] = value
    logger.info("SVR parameters: %s", params)

    fold_results: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []
    for fold in folds:
        train = table[table["fold"] != fold]
        test = table[table["fold"] == fold]
        x_train = train[selected_features]
        x_test = test[selected_features]
        y_train = train[target].to_numpy()

        # Fit this fold's final scaler only on its outer training partition.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = SVR(**params)
        model.fit(x_train_scaled, y_train)
        train_pred = model.predict(x_train_scaled)
        test_pred = model.predict(x_test_scaled)
        result = {
            "fold": fold,
            "train_r2": float(r2_score(y_train, train_pred)),
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
    report = {
        "features": selected_features,
        "target": target,
        "parameters": params,
        **aggregate,
        "total_elapsed_seconds": elapsed,
        "fold_metrics": fold_results,
        "n_usable_rows": int(len(table)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
