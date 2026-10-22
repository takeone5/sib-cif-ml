#!/usr/bin/env python3
"""Run a fixed-parameter LightGBM demo using the existing 5-fold split."""

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
from sib_features.extractor import SUPPORTED_SPACE_GROUP_NUMBERS


FEATURE_COLUMNS = [
    "002_TM_layer_entropy",
    "003_Na_content_per_O2",
    "008_alkali_fraction_in_TM_layer",
    "010_alkaline_earth_fraction_in_TM_layer",
    "016_Ni_Co_Mn_Fe_required_charge",
    "107_AtomicRadius_mean",
    "108_AtomicRadius_avg_dev",
    "110_Electronegativity_avg_dev",
    "111_NdValence",
    "112_CohesiveEnergy",
    "113_is_Prismatic",
    "114_is_2Layer",
]
TARGET_PREFIX = "106_"
DEFAULT_FEATURES = Path("correlation_output/fold_1_train_spearman/features_all.csv")
DEFAULT_SPLIT_ROOT = Path("split_data_v1")
DEFAULT_OUTPUT = Path("ml_output/lightgbm_demo")


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lightgbm_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def target_column(table: pd.DataFrame) -> str:
    targets = [column for column in table.columns if str(column).startswith(TARGET_PREFIX)]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target column starting with {TARGET_PREFIX!r}; found {targets}")
    return targets[0]


def source_to_relative(source: str, data_root: Path | None) -> str:
    source_path = Path(str(source))
    if data_root is not None:
        try:
            return source_path.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            pass
    return source_path.as_posix().lstrip("./")


def attach_folds(
    table: pd.DataFrame,
    membership: pd.DataFrame,
    data_root: Path | None,
) -> pd.DataFrame:
    if "source_cif" not in table.columns:
        raise ValueError("Feature table must contain a 'source_cif' column")
    required_membership_columns = {"fold", "split", "relative_path"}
    missing_membership_columns = required_membership_columns - set(membership.columns)
    if missing_membership_columns:
        raise ValueError(f"Fold membership is missing columns: {sorted(missing_membership_columns)}")
    membership = membership.copy()
    membership["relative_path"] = membership["relative_path"].astype(str).str.replace("\\", "/")
    # The membership CSV has one train/test row per sample for every fold.
    # Each sample belongs to exactly one test fold; use that row as its fold.
    test_membership = membership[membership["split"].astype(str).str.lower() == "test"]
    duplicate_test_paths = test_membership["relative_path"].duplicated(keep=False)
    if duplicate_test_paths.any():
        examples = test_membership.loc[duplicate_test_paths, "relative_path"].head(3).tolist()
        raise ValueError(f"CIF paths appear in multiple test folds: {examples}")
    fold_by_path = dict(zip(test_membership["relative_path"], test_membership["fold"], strict=True))
    known_paths = list(fold_by_path)
    folds: list[int | None] = []
    for source in table["source_cif"]:
        relative = source_to_relative(str(source), data_root)
        fold = fold_by_path.get(relative)
        if fold is None:
            suffix_matches = [path for path in known_paths if relative.endswith("/" + path) or relative == path]
            if len(suffix_matches) == 1:
                fold = fold_by_path[suffix_matches[0]]
        folds.append(int(fold) if fold is not None else None)
    result = table.copy()
    result["fold"] = folds
    missing = int(result["fold"].isna().sum())
    if missing:
        examples = result.loc[result["fold"].isna(), "source_cif"].head(3).tolist()
        raise ValueError(f"Could not match {missing} feature rows to fold membership; examples: {examples}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the fixed-parameter LightGBM demo model.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="features_all.csv from feature extraction")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--data-root", type=Path, help="CIF root used to create the feature table")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir / "train.log")
    started = time.perf_counter()

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise SystemExit("lightgbm is required. Install project dependencies with: pip install -e .") from exc

    summary_path = args.split_root / "summary.json"
    membership_path = args.split_root / "fold_membership.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    membership = pd.read_csv(membership_path)
    table = pd.read_csv(args.features)
    target = target_column(table)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")
    source_sg_columns = {"source_cif_space_group_number", "source_json_space_group_number"}
    missing_sg_columns = source_sg_columns - set(table.columns)
    if missing_sg_columns:
        raise ValueError(
            "Feature CSV lacks source CIF/JSON space-group columns. "
            "Regenerate features_all.csv with the current extractor; missing: "
            f"{sorted(missing_sg_columns)}"
        )
    cif_space_groups = pd.to_numeric(table["source_cif_space_group_number"], errors="coerce")
    json_space_groups = pd.to_numeric(table["source_json_space_group_number"], errors="coerce")
    supported_cif = cif_space_groups.isin(SUPPORTED_SPACE_GROUP_NUMBERS)
    supported_json = json_space_groups.isna() | json_space_groups.isin(SUPPORTED_SPACE_GROUP_NUMBERS)
    unsupported = ~(supported_cif & supported_json)
    logger.info("Excluded %d rows with unsupported or unknown space group", int(unsupported.sum()))
    table = table.loc[~unsupported].copy()

    table = attach_folds(table, membership, args.data_root)
    selected = FEATURE_COLUMNS + [target]
    before = len(table)
    table = table.dropna(subset=selected).copy()
    logger.info("Dataset: %d rows usable (%d dropped due to missing selected values)", len(table), before - len(table))
    fold_counts = table["fold"].value_counts().reindex(sorted(int(value) for value in membership["fold"].unique()), fill_value=0)
    logger.info("Usable rows per test fold: %s", fold_counts.to_dict())
    empty_folds = fold_counts[fold_counts == 0].index.tolist()
    if empty_folds:
        raise ValueError(
            "No usable test rows remain in fold(s) "
            f"{empty_folds} after removing rows with missing selected features or target. "
            "Check feature missingness and ensure the complete dataset was extracted."
        )
    logger.info("Features: %s", ", ".join(FEATURE_COLUMNS))
    logger.info("Target: %s", target)
    logger.info("Split summary: %d samples, %d folds", summary["n_samples"], len(summary["folds"]))

    params = {
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 8,
        "learning_rate": 0.05,
        "colsample_bytree": 0.7,
        "n_estimators": 150,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "objective": "regression",
        "random_state": 42,
        "verbosity": -1,
    }
    logger.info("Model parameters: %s", params)

    fold_results: list[dict[str, float | int]] = []
    all_predictions: list[pd.DataFrame] = []
    feature_importances: list[np.ndarray] = []
    for fold in sorted(int(value) for value in membership["fold"].unique()):
        train = table[table["fold"] != fold]
        test = table[table["fold"] == fold]
        model = LGBMRegressor(**params)
        model.fit(train[FEATURE_COLUMNS], train[target])
        feature_importances.append(np.asarray(model.feature_importances_, dtype=float))
        train_pred = model.predict(train[FEATURE_COLUMNS])
        test_pred = model.predict(test[FEATURE_COLUMNS])
        result = {
            "fold": fold,
            "train_r2": float(r2_score(train[target], train_pred)),
            "test_r2": float(r2_score(test[target], test_pred)),
            "test_rmse": float(math.sqrt(mean_squared_error(test[target], test_pred))),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }
        fold_results.append(result)
        all_predictions.append(
            pd.DataFrame(
                {
                    "source_cif": test["source_cif"].values,
                    "fold": fold,
                    "target": test[target].values,
                    "prediction": test_pred,
                }
            )
        )
        logger.info(
            "Fold %d | train R2=%.4f | test R2=%.4f | test RMSE=%.4f | train=%d test=%d",
            fold,
            result["train_r2"],
            result["test_r2"],
            result["test_rmse"],
            result["n_train"],
            result["n_test"],
        )

    train_r2 = float(np.mean([result["train_r2"] for result in fold_results]))
    test_r2 = float(np.mean([result["test_r2"] for result in fold_results]))
    mean_rmse = float(np.mean([result["test_rmse"] for result in fold_results]))
    elapsed = time.perf_counter() - started
    logger.info("Mean train R2: %.4f", train_r2)
    logger.info("Mean test R2: %.4f", test_r2)
    logger.info("Mean RMSE: %.4f", mean_rmse)
    logger.info("Total elapsed time: %.3f s", elapsed)

    pd.DataFrame(fold_results).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": np.mean(feature_importances, axis=0),
            "importance_std": np.std(feature_importances, axis=0),
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    report = {
        "features": FEATURE_COLUMNS,
        "target": target,
        "parameters": params,
        "mean_train_r2": train_r2,
        "mean_test_r2": test_r2,
        "mean_rmse": mean_rmse,
        "total_elapsed_seconds": elapsed,
        "fold_metrics": fold_results,
        "n_usable_rows": int(len(table)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
