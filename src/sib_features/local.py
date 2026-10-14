"""Local coordination features [201], [206]-[208], [213]-[220], [225]-[228]."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from pymatgen.core import Structure

from .chemistry import is_transition_metal


OXYGEN = {"O", "F"}
NEIGHBOR_CUTOFF_A = 3.2
COORDINATION_NUMBER = 6


def _stats(values: list[tuple[float, float]]) -> dict[str, float | None]:
    """Weighted descriptive statistics; weights are fractional occupancies."""
    pairs = [(float(v), float(w)) for v, w in values if w > 0 and math.isfinite(v)]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return {k: None for k in ("mean", "harmonic", "geometric", "max", "min", "std", "variance", "skew")}
    if len({v for v, _ in pairs}) == 1:
        # Preserve exact equality after the requested distance quantization;
        # do not reintroduce floating-point noise through repeated sums.
        value = pairs[0][0]
        return {
            "mean": value,
            "harmonic": value,
            "geometric": value,
            "max": value,
            "min": value,
            "std": 0.0,
            "variance": 0.0,
            "skew": 0.0,
        }
    mean = sum(v * w for v, w in pairs) / total
    variance = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    std = math.sqrt(variance)
    positive = [(v, w) for v, w in pairs if v > 0]
    harmonic = total / sum(w / v for v, w in positive) if len(positive) == len(pairs) else None
    geometric = math.exp(sum(w * math.log(v) for v, w in pairs) / total) if len(positive) == len(pairs) else None
    skew = sum(w * (v - mean) ** 3 for v, w in pairs) / total / std**3 if std > 0 else 0.0
    return {
        "mean": mean,
        "harmonic": harmonic,
        "geometric": geometric,
        "max": max(v for v, _ in pairs),
        "min": min(v for v, _ in pairs),
        "std": std,
        "variance": variance,
        "skew": skew,
    }


def _species_occupancy(site, symbols: set[str]) -> float:
    return sum(float(occ) for sp, occ in site.species.items() if sp.symbol in symbols)


def _oxygen_neighbors(structure: Structure, center_index: int) -> list[tuple[np.ndarray, float, float]]:
    center = structure[center_index]
    neighbors = []
    for neighbor in structure.get_neighbors(center, NEIGHBOR_CUTOFF_A):
        occupancy = _species_occupancy(neighbor, OXYGEN)
        if occupancy > 0:
            neighbors.append((np.asarray(neighbor.coords, dtype=float), occupancy, float(neighbor.nn_distance)))
    neighbors.sort(key=lambda x: x[2])
    # All local features are defined on a six-coordinate shell.  Do not
    # silently calculate a different feature from a partial shell.
    if len(neighbors) < COORDINATION_NUMBER:
        return []
    return neighbors[:COORDINATION_NUMBER]


def _center_indices(structure: Structure, kind: str) -> list[tuple[int, float]]:
    centers = []
    for index, site in enumerate(structure):
        if kind == "Na":
            na_occupancy = _species_occupancy(site, {"Na"})
            if na_occupancy <= 0:
                continue
            # Treat Na and all species sharing its crystallographic site as
            # one center. Vacancy is not a center and is therefore excluded.
            occupancy = sum(float(occ) for occ in site.species.values())
        else:
            occupancy = sum(float(occ) for sp, occ in site.species.items() if is_transition_metal(sp.symbol))
        if occupancy > 0:
            centers.append((index, occupancy))
    return centers


def _bond_values(structure: Structure, kind: str) -> list[tuple[float, float]]:
    values = []
    for index, center_occupancy in _center_indices(structure, kind):
        for _, oxygen_occupancy, distance in _oxygen_neighbors(structure, index):
            # Bond-length features describe geometry of a fully occupied
            # crystallographic center and O/F shell, without occupancy weights.
            # Report/compare distances at the requested 0.001 Angstrom
            # resolution so floating-point noise cannot create false max/min
            # differences in symmetry-equivalent bonds.
            values.append((round(distance, 3), 1.0))
    return values


def _angle_values(structure: Structure, kind: str) -> list[tuple[float, float]]:
    values = []
    for index, center_occupancy in _center_indices(structure, kind):
        center = np.asarray(structure[index].coords, dtype=float)
        neighbors = _oxygen_neighbors(structure, index)
        for (a, wa, _), (b, wb, _) in combinations(neighbors, 2):
            va, vb = a - center, b - center
            cosine = np.dot(va, vb) / np.linalg.norm(va) / np.linalg.norm(vb)
            angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            if kind in {"Na", "TM"}:
                # [201]-[214] describe the geometry of a fully occupied
                # crystallographic center; do not weight angles by species,
                # oxygen occupancy, or vacancy.
                values.append((angle, 1.0))
            else:
                values.append((angle, center_occupancy * wa * wb))
    return values


def _merge_selected(
    prefix: str,
    stats: dict[str, float | None],
    definitions: tuple[tuple[int, str, str], ...],
) -> dict[str, float | None]:
    """Merge only the requested statistics, preserving their feature numbers."""
    return {f"{number:03d}_{prefix}_{suffix}": stats[stat_name] for number, suffix, stat_name in definitions}


def _round_stats(stats: dict[str, float | None], decimals: int) -> dict[str, float | None]:
    """Round numeric statistics for feature outputs at their physical resolution."""
    return {key: (None if value is None else round(float(value), decimals)) for key, value in stats.items()}


def extract_local_features(structure: Structure) -> dict[str, float | None]:
    """Extract all currently specified local features."""
    na_angles = _round_stats(_stats(_angle_values(structure, "Na")), 2)
    tm_angles = _round_stats(_stats(_angle_values(structure, "TM")), 2)
    na_bonds = _round_stats(_stats(_bond_values(structure, "Na")), 3)
    tm_bonds = _round_stats(_stats(_bond_values(structure, "TM")), 3)
    result = {}
    result.update(
        _merge_selected(
            "O_Na_O_angle_deg",
            na_angles,
            (
                (201, "arithmetic_mean", "mean"),
                (206, "std", "std"),
                (207, "skew", "skew"),
            ),
        )
    )
    result.update(
        _merge_selected(
            "O_TM_O_angle_deg",
            tm_angles,
            (
                (208, "arithmetic_mean", "mean"),
                (213, "std", "std"),
                (214, "skew", "skew"),
            ),
        )
    )
    result.update(
        _merge_selected(
            "Na_O_bond_length_A",
            na_bonds,
            (
                (215, "arithmetic_mean", "mean"),
                (217, "harmonic_mean", "harmonic"),
                (219, "geometric_mean", "geometric"),
                (225, "std", "std"),
                (227, "skew", "skew"),
            ),
        )
    )
    result.update(
        _merge_selected(
            "TM_O_bond_length_A",
            tm_bonds,
            (
                (216, "arithmetic_mean", "mean"),
                (218, "harmonic_mean", "harmonic"),
                (220, "geometric_mean", "geometric"),
                (226, "std", "std"),
                (228, "skew", "skew"),
            ),
        )
    )
    return result
