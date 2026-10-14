"""Create a leakage-free, stratified grouped 5-fold CIF dataset split.

The grouping key is ``reduced_formula_is_Prismatic_is_2Layer``. CIFs with the
same composition and phase stay in one fold, while different phases of the
same composition may be assigned to different folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure


SPACE_GROUP_LABELS = {160: (1, 0), 166: (0, 0), 194: (1, 1)}
DEFAULT_SEED = 260718
OCCUPANCY_TOLERANCE = 1.00000001


@dataclass(frozen=True)
class Sample:
    source: Path
    relative_path: str
    reduced_formula: str
    space_group: int
    is_prismatic: int
    is_2Layer: int
    group_key: str


def _space_group_number(path: Path, structure: Structure) -> int | None:
    """Read the CIF's declared IT number, falling back to pymatgen analysis."""
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"_symmetry_Int_Tables_number\s+['\"]?([0-9]+)",
        r"_space_group_IT_number\s+['\"]?([0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    try:
        return int(structure.get_space_group_info()[1])
    except Exception:
        return None


def collect_samples(input_root: Path) -> tuple[list[Sample], list[dict[str, str]]]:
    samples: list[Sample] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".cif"):
        relative = path.relative_to(input_root)
        if "cod_cif" in relative.parts:
            excluded.append({"path": str(relative), "reason": "cod_cif"})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                structure = Structure.from_file(
                    str(path), occupancy_tolerance=OCCUPANCY_TOLERANCE
                )
            sg_number = _space_group_number(path, structure)
            if sg_number not in SPACE_GROUP_LABELS:
                excluded.append({"path": str(relative), "reason": f"space_group:{sg_number}"})
                continue
            is_prismatic, is_2layer = SPACE_GROUP_LABELS[sg_number]
            reduced_formula = structure.composition.reduced_formula
            group_key = f"{reduced_formula}_{is_prismatic}_{is_2layer}"
            samples.append(
                Sample(
                    source=path,
                    relative_path=str(relative),
                    reduced_formula=reduced_formula,
                    space_group=sg_number,
                    is_prismatic=is_prismatic,
                    is_2Layer=is_2layer,
                    group_key=group_key,
                )
            )
        except Exception as exc:
            excluded.append({"path": str(relative), "reason": f"parse_error:{type(exc).__name__}:{exc}"})
    return samples, excluded


def assign_folds(samples: list[Sample], n_splits: int, seed: int) -> dict[str, int]:
    from sklearn.model_selection import StratifiedGroupKFold
    import numpy as np

    indices = np.arange(len(samples))
    y = np.array([sample.is_prismatic for sample in samples])
    groups = np.array([sample.group_key for sample in samples])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments: dict[str, int] = {}
    for fold, (_, test_indices) in enumerate(splitter.split(indices, y, groups)):
        for index in test_indices:
            group_key = samples[int(index)].group_key
            previous = assignments.setdefault(group_key, fold)
            if previous != fold:
                raise AssertionError(f"Group assigned to multiple folds: {group_key}")
    return assignments


def validate_assignments(samples: list[Sample], assignments: dict[str, int], n_splits: int) -> None:
    sample_groups = {sample.group_key for sample in samples}
    if sample_groups != set(assignments):
        raise AssertionError("Every sample group must have exactly one fold assignment")
    for fold in range(n_splits):
        test_groups = {group for group, assigned in assignments.items() if assigned == fold}
        train_groups = sample_groups - test_groups
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Data leakage in fold {fold + 1}: {sorted(overlap)}")


def copy_split(samples: list[Sample], assignments: dict[str, int], output_root: Path, n_splits: int) -> None:
    for fold in range(n_splits):
        test_root = output_root / f"fold_{fold + 1}" / "test"
        train_root = output_root / f"fold_{fold + 1}" / "train"
        for sample in samples:
            destination_root = test_root if assignments[sample.group_key] == fold else train_root
            destination = destination_root / sample.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.source, destination)


def write_reports(
    samples: list[Sample],
    excluded: list[dict[str, str]],
    assignments: dict[str, int],
    output_root: Path,
    n_splits: int,
) -> None:
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["assigned_test_fold", "group_key", "relative_path", "reduced_formula", "space_group", "is_Prismatic", "is_2Layer"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            fold = assignments[sample.group_key]
            writer.writerow(
                {
                    "assigned_test_fold": fold + 1,
                    "group_key": sample.group_key,
                    "relative_path": sample.relative_path,
                    "reduced_formula": sample.reduced_formula,
                    "space_group": sample.space_group,
                    "is_Prismatic": sample.is_prismatic,
                    "is_2Layer": sample.is_2Layer,
                }
            )

    # The fold-specific train/test membership is written separately so the
    # manifest remains one row per CIF rather than five duplicated rows.
    memberships = output_root / "fold_membership.csv"
    with memberships.open("w", newline="", encoding="utf-8") as handle:
        fields = ["fold", "split", "group_key", "relative_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fold in range(n_splits):
            for sample in samples:
                test = assignments[sample.group_key] == fold
                writer.writerow({"fold": fold + 1, "split": "test" if test else "train", "group_key": sample.group_key, "relative_path": sample.relative_path})

    summary: dict[str, object] = {
        "n_samples": len(samples),
        "n_groups": len(assignments),
        "n_excluded": len(excluded),
        "group_leakage_validation": {
            "status": "PASS",
            "train_test_group_overlap_per_fold": 0,
            "group_assigned_to_multiple_folds": 0,
        },
        "excluded": excluded,
        "folds": [],
    }
    for fold in range(n_splits):
        test = [sample for sample in samples if assignments[sample.group_key] == fold]
        train = [sample for sample in samples if assignments[sample.group_key] != fold]
        fold_summary = {"fold": fold + 1}
        for split, split_samples in (("train", train), ("test", test)):
            fold_summary[split] = {
                "n_samples": len(split_samples),
                "n_unique_groups": len({sample.group_key for sample in split_samples}),
                "n_unique_reduced_formulas": len({sample.reduced_formula for sample in split_samples}),
                "space_group_counts": dict(sorted(Counter(sample.space_group for sample in split_samples).items())),
                "is_Prismatic_counts": dict(sorted(Counter(sample.is_prismatic for sample in split_samples).items())),
                "is_2Layer_counts": dict(sorted(Counter(sample.is_2Layer for sample in split_samples).items())),
            }
        summary["folds"].append(fold_summary)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(samples: list[Sample], assignments: dict[str, int], n_splits: int) -> None:
    print(f"Included CIFs: {len(samples)}")
    print(f"Unique groups: {len(assignments)}")
    print("Group leakage validation: PASS (train/test overlap = 0 for every fold)")
    for fold in range(n_splits):
        test = [sample for sample in samples if assignments[sample.group_key] == fold]
        train = [sample for sample in samples if assignments[sample.group_key] != fold]
        print(f"Fold {fold + 1}: train={len(train)}, test={len(test)}")
        for split, split_samples in (("train", train), ("test", test)):
            sg = dict(sorted(Counter(sample.space_group for sample in split_samples).items()))
            phase = dict(sorted(Counter(sample.is_prismatic for sample in split_samples).items()))
            formulas = len({sample.reduced_formula for sample in split_samples})
            print(f"  {split}: unique_reduced_formulas={formulas}, space_groups={sg}, is_Prismatic={phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("origin_data_v4"))
    parser.add_argument("--output-root", type=Path, default=Path("split_data_v1"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    samples, excluded = collect_samples(args.input_root)
    if not samples:
        raise SystemExit("No eligible CIF files found.")
    assignments = assign_folds(samples, args.n_splits, args.seed)
    validate_assignments(samples, assignments, args.n_splits)
    copy_split(samples, assignments, args.output_root, args.n_splits)
    write_reports(samples, excluded, assignments, args.output_root, args.n_splits)
    print_summary(samples, assignments, args.n_splits)
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
