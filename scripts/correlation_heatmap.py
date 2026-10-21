#!/usr/bin/env python3
"""Extract Fold 1 train features and plot a Spearman heatmap.

This script is intentionally sequential. The local-structure extractor is
CPU-only and the default workflow avoids multiprocessing memory spikes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sib_features import FeatureExtractor
from sib_features.io import warning_log


SAMPLE_ID_RE = re.compile(r"^(?P<id>\d+_S\d+(?:_\d+)?)")
DEFAULT_DATA_ROOT = Path("split_data_v1/fold_1/train")
DEFAULT_OUTPUT_DIR = Path("correlation_output/fold_1_train_spearman")
EXCLUDED_HEATMAP_FEATURE_IDS = {101, 102, 103, 104, 105, 106}


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
    warning_log_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cif_paths = sorted(root.rglob("*.cif"))
    if limit is not None:
        cif_paths = cif_paths[:limit]
    json_index = make_json_index(json_root or root)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total = len(cif_paths)
    with warning_log(warning_log_path):
        for number, cif_path in enumerate(cif_paths, start=1):
            json_path = match_json(cif_path, json_index)
            status = "OK"
            try:
                result = FeatureExtractor.from_files(cif_path, json_path).extract()
                row = {"source_cif": str(cif_path), "source_json": str(json_path) if json_path else ""}
                row.update(flatten_numeric(result))
                rows.append(row)
            except Exception as exc:  # Keep one malformed structure from stopping a batch.
                status = "FAIL"
                errors.append({"source_cif": str(cif_path), "error": repr(exc)})
            print(f"\r[{number}/{total}] {cif_path.name} ; {status}", end="", flush=True)
    if total:
        print()
    return pd.DataFrame(rows), pd.DataFrame(errors, columns=["source_cif", "error"])


def plot_heatmap(correlation: pd.DataFrame, output: Path, figsize: tuple[float, float]) -> None:
    figure, axis = plt.subplots(figsize=figsize)
    values = correlation.to_numpy(dtype=float)
    lower_triangle = np.tril(np.ones(values.shape, dtype=bool), k=-1)
    masked_values = np.ma.masked_where(lower_triangle | np.isnan(values), values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    image = axis.imshow(masked_values, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="none")

    # Scale labels to the available cell size.  This keeps labels prominent for
    # small matrices while preventing adjacent row labels from colliding when
    # the feature count grows.
    row_count = max(len(correlation.index), 1)
    col_count = max(len(correlation.columns), 1)
    row_cell_points = figsize[1] * 72.0 / row_count
    col_cell_points = figsize[0] * 72.0 / col_count
    label_fontsize = min(
        16.0,
        max(8.0, row_cell_points * 0.72),
        max(8.0, col_cell_points * 0.34),
    )
    axis.set_xticks(range(col_count), labels=correlation.columns, rotation=90, fontsize=label_fontsize)
    axis.set_yticks(range(row_count), labels=correlation.index, fontsize=label_fontsize)
    axis.tick_params(axis="both", which="major", pad=3)
    axis.set_title("Spearman correlation of Fold 1 train features", fontsize=14, pad=14)
    figure.colorbar(image, ax=axis, label="Spearman ρ", fraction=0.046, pad=0.04)
    # Let tight_layout reserve the larger tick-label margins rather than
    # allowing labels to be clipped or overlap the colorbar/title.
    figure.tight_layout(pad=1.2)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_interactive_heatmap(correlation: pd.DataFrame, output: Path) -> bool:
    """Save an optional Plotly heatmap with hover values and an abs(ρ) slider."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        warnings.warn(
            "Plotly is not installed; skipping the interactive heatmap. "
            "The static PNG was still generated.",
            UserWarning,
            stacklevel=2,
        )
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    values = correlation.to_numpy(dtype=float)
    lower_triangle = np.tril(np.ones(values.shape, dtype=bool), k=-1)
    values[lower_triangle] = np.nan
    labels = [str(label) for label in correlation.columns]

    def values_for_cutoff(cutoff: float) -> list[list[float | None]]:
        masked = values.copy()
        masked[np.abs(masked) < cutoff] = np.nan
        return [
            [None if np.isnan(value) else float(value) for value in row]
            for row in masked
        ]

    initial_cutoff = 0.0
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=values_for_cutoff(initial_cutoff),
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar={"title": "Spearman ρ"},
                hovertemplate="x: %{x}<br>y: %{y}<br>ρ: %{z:.3f}<extra></extra>",
            )
        ]
    )
    cutoffs = np.arange(0.0, 1.01, 0.05)
    figure.update_layout(
        title="Interactive Spearman correlation heatmap",
        # The slider is deliberately placed well below the rotated x labels.
        # The extra bottom margin is needed in both browser and notebook output.
        margin={"l": 170, "r": 40, "t": 90, "b": 250},
        xaxis={"tickangle": 90, "automargin": True, "ticklabelstandoff": 10},
        yaxis={"autorange": "reversed"},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Minimum |ρ|: "},
                "y": -0.30,
                "pad": {"t": 18},
                "steps": [
                    {
                        "label": f"{cutoff:.2f}",
                        "method": "restyle",
                        "args": [{"z": [values_for_cutoff(float(cutoff))]}],
                    }
                    for cutoff in cutoffs
                ],
            }
        ],
    )
    figure.write_html(output, include_plotlyjs=True)

    # In Colab/Jupyter, display the same interactive figure inline as well.
    try:
        get_ipython  # type: ignore[name-defined]
    except NameError:
        return True
    try:
        from IPython.display import display

        display(figure)
    except Exception as exc:
        warnings.warn(f"Could not display the interactive heatmap inline: {exc}", UserWarning, stacklevel=2)
    return True


def select_heatmap_features(table: pd.DataFrame) -> pd.DataFrame:
    """Select composition features and 113-114 while excluding 101-106."""
    numeric = table.select_dtypes(include="number")
    feature_columns = []
    for column in numeric.columns:
        match = re.match(r"^(\d{3})_", str(column))
        if match and 1 <= int(match.group(1)) <= 114 and int(match.group(1)) not in EXCLUDED_HEATMAP_FEATURE_IDS:
            feature_columns.append(column)
    numeric = numeric.loc[:, feature_columns]
    numeric = numeric.dropna(axis=1, how="all")
    return numeric.loc[:, numeric.nunique(dropna=True) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Fold 1 train CIF features and create a Spearman heatmap.")
    parser.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA_ROOT,
        help="Directory recursively containing CIF files (default: Fold 1 train set).",
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        help="Directory recursively containing JSON metadata; defaults to data_root.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--interactive-output",
        type=Path,
        help="Interactive Plotly HTML path (default: <output-dir>/spearman_heatmap.html).",
    )
    parser.add_argument(
        "--warning-log",
        type=Path,
        help="Warning log path (default: <output-dir>/warnings.log).",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N sorted CIF files for a pilot run.")
    parser.add_argument("--min-periods", type=int, default=3, help="Minimum paired samples for each correlation.")
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
    warning_log_path = args.warning_log or args.output_dir / "warnings.log"
    table, errors = extract_dataset(args.data_root, args.limit, args.json_root, warning_log_path)
    if table.empty:
        print("No CIF produced a feature row.", file=sys.stderr)
        errors.to_csv(args.output_dir / "extraction_errors.csv", index=False)
        raise SystemExit(1)
    table.to_csv(args.output_dir / "features_all.csv", index=False)
    errors.to_csv(args.output_dir / "extraction_errors.csv", index=False)

    numeric = select_heatmap_features(table)
    correlation = numeric.corr(method="spearman", min_periods=args.min_periods)
    correlation.to_csv(args.output_dir / "spearman_correlation.csv")
    plot_heatmap(correlation, args.output_dir / "spearman_heatmap.png", (args.fig_width, args.fig_height))
    interactive_output = args.interactive_output or args.output_dir / "spearman_heatmap.html"
    with warning_log(warning_log_path):
        try:
            interactive_created = plot_interactive_heatmap(correlation, interactive_output)
        except Exception as exc:
            warnings.warn(
                f"Interactive heatmap could not be created: {exc}. The static PNG was still generated.",
                UserWarning,
                stacklevel=2,
            )
            interactive_created = False
    metadata = {
        "data_root": str(args.data_root),
        "json_root": str(args.json_root or args.data_root),
        "cif_count_requested": args.limit,
        "successful_rows": int(len(table)),
        "error_rows": int(len(errors)),
        "warning_log": str(warning_log_path),
        "interactive_heatmap": str(interactive_output) if interactive_created else None,
        "numeric_columns_in_heatmap": int(len(numeric.columns)),
        "correlation_method": "spearman",
        "feature_id_range": "001-112, 113, 114",
        "excluded_feature_ids": sorted(EXCLUDED_HEATMAP_FEATURE_IDS),
        "min_periods": args.min_periods,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
