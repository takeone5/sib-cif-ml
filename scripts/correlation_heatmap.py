#!/usr/bin/env python3
"""Extract features from a CIF directory and plot a Pearson heatmap.

This script is intentionally sequential. The local-structure extractor is
CPU-only and the default workflow avoids multiprocessing memory spikes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sib_features import FeatureExtractor


SAMPLE_ID_RE = re.compile(r"^(?P<id>\d+_S\d+(?:_\d+)?)")
DEFAULT_HEATMAP_EXCLUDED_PREFIXES = (
    "004_TM_layer_element_fractions",
    "005_Na_layer_element_fractions",
    "104_O_TM_O_distance_A",
    "105_O_Na_O_distance_A",
    "106_O_Na_O_to_O_TM_O_ratio",
)


def sample_id(path: Path) -> str | None:
    match = SAMPLE_ID_RE.match(path.stem)
    return match.group("id") if match else None


def make_json_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*.json"):
        index.setdefault(path.stem, []).append(path)
        identifier = sample_id(path)
        if identifier:
            index.setdefault(identifier, []).append(path)
    return index


def match_json(cif_path: Path, json_index: dict[str, list[Path]]) -> Path | None:
    exact = json_index.get(cif_path.stem, [])
    if exact:
        return sorted(set(exact), key=lambda path: len(path.name))[0]
    identifier = sample_id(cif_path)
    candidates = json_index.get(identifier or "", [])
    if not candidates:
        return None
    # Prefer a JSON in the same sample directory, then the shortest filename.
    return sorted(set(candidates), key=lambda path: (path.parent != cif_path.parent, len(path.name)))[0]


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested feature dictionaries into numeric dataframe columns."""
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}__{key}" if prefix else str(key)
            flattened.update(flatten_numeric(child, child_prefix))
        return flattened
    if isinstance(value, bool):
        return {prefix: float(value)}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def extract_dataset(
    root: Path,
    limit: int | None = None,
    json_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cif_paths = sorted(root.rglob("*.cif"))
    if limit is not None:
        cif_paths = cif_paths[:limit]
    json_index = make_json_index(json_root or root)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total = len(cif_paths)
    for number, cif_path in enumerate(cif_paths, start=1):
        json_path = match_json(cif_path, json_index)
        try:
            result = FeatureExtractor.from_files(cif_path, json_path).extract()
            row = {"source_cif": str(cif_path), "source_json": str(json_path) if json_path else ""}
            row.update(flatten_numeric(result))
            rows.append(row)
        except Exception as exc:  # Keep one malformed structure from stopping a batch.
            errors.append({"source_cif": str(cif_path), "error": repr(exc)})
        if number == 1 or number % 25 == 0 or number == total:
            print(f"[{number}/{total}] processed; success={len(rows)}, errors={len(errors)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(errors, columns=["source_cif", "error"])


def plot_heatmap(correlation: pd.DataFrame, output: Path, figsize: tuple[float, float]) -> None:
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(correlation.columns)), labels=correlation.columns, rotation=90, fontsize=5)
    axis.set_yticks(range(len(correlation.index)), labels=correlation.index, fontsize=5)
    axis.set_title("Pearson correlation of extracted features")
    figure.colorbar(image, ax=axis, label="Pearson r", fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def select_heatmap_features(table: pd.DataFrame, exclude_element_vector: bool = False) -> pd.DataFrame:
    """Select numeric features for correlation without changing extracted data."""
    numeric = table.select_dtypes(include="number")
    excluded_prefixes = DEFAULT_HEATMAP_EXCLUDED_PREFIXES
    if exclude_element_vector:
        excluded_prefixes = excluded_prefixes + ("004_TM_layer_element_fractions__", "005_Na_layer_element_fractions__")
    keep = ~numeric.columns.to_series().apply(
        lambda column: any(column == prefix or column.startswith(prefix + "__") for prefix in excluded_prefixes)
    )
    numeric = numeric.loc[:, keep.to_numpy()]
    numeric = numeric.dropna(axis=1, how="all")
    return numeric.loc[:, numeric.nunique(dropna=True) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CIF features and create a Pearson correlation heatmap.")
    parser.add_argument("data_root", type=Path, help="Directory recursively containing CIF files.")
    parser.add_argument(
        "--json-root",
        type=Path,
        help="Directory recursively containing JSON metadata; defaults to data_root.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("correlation_output"))
    parser.add_argument("--limit", type=int, help="Process only the first N sorted CIF files for a pilot run.")
    parser.add_argument("--min-periods", type=int, default=3, help="Minimum paired samples for each correlation.")
    parser.add_argument("--exclude-element-vector", action="store_true", help="Also exclude expanded element-fraction columns from the heatmap (already excluded by default).")
    parser.add_argument("--fig-width", type=float, default=28.0)
    parser.add_argument("--fig-height", type=float, default=28.0)
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"data_root is not a directory: {args.data_root}")
    if args.json_root is not None and not args.json_root.is_dir():
        parser.error(f"json_root is not a directory: {args.json_root}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, errors = extract_dataset(args.data_root, args.limit, args.json_root)
    if table.empty:
        print("No CIF produced a feature row.", file=sys.stderr)
        errors.to_csv(args.output_dir / "extraction_errors.csv", index=False)
        raise SystemExit(1)
    table.to_csv(args.output_dir / "features_all.csv", index=False)
    errors.to_csv(args.output_dir / "extraction_errors.csv", index=False)

    numeric = select_heatmap_features(table, args.exclude_element_vector)
    correlation = numeric.corr(method="pearson", min_periods=args.min_periods)
    correlation.to_csv(args.output_dir / "pearson_correlation.csv")
    plot_heatmap(correlation, args.output_dir / "pearson_heatmap.png", (args.fig_width, args.fig_height))
    metadata = {
        "data_root": str(args.data_root),
        "json_root": str(args.json_root or args.data_root),
        "cif_count_requested": args.limit,
        "successful_rows": int(len(table)),
        "error_rows": int(len(errors)),
        "numeric_columns_in_heatmap": int(len(numeric.columns)),
        "excluded_feature_prefixes": list(DEFAULT_HEATMAP_EXCLUDED_PREFIXES),
        "oxidation_states_included": False,
        "min_periods": args.min_periods,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
