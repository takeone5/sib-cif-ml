"""Local coordination features [201]-[228]."""

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
            values.append((distance, 1.0))
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


def _merge(prefix: str, stats: dict[str, float | None], feature_numbers: tuple[int, ...]) -> dict[str, float | None]:
    suffixes = (
        "arithmetic_mean",
        "harmonic_mean",
        "geometric_mean",
        "max",
        "min",
        "std",
        "skew",
    )
    stat_names = ("mean", "harmonic", "geometric", "max", "min", "std", "skew")
    return {
        f"{number:03d}_{prefix}_{suffix}": stats[stat_name]
        for number, suffix, stat_name in zip(feature_numbers, suffixes, stat_names)
    }


def extract_local_features(structure: Structure) -> dict[str, float | None]:
    """Extract all currently specified local features."""
    na_angles = _stats(_angle_values(structure, "Na"))
    tm_angles = _stats(_angle_values(structure, "TM"))
    na_bonds = _stats(_bond_values(structure, "Na"))
    tm_bonds = _stats(_bond_values(structure, "TM"))
    result = {}
    result.update(_merge("O_Na_O_angle_deg", na_angles, (201, 202, 203, 204, 205, 206, 207)))
    result.update(_merge("O_TM_O_angle_deg", tm_angles, (208, 209, 210, 211, 212, 213, 214)))
    result.update(_merge("Na_O_bond_length_A", na_bonds, (215, 217, 219, 221, 223, 225, 227)))
    result.update(_merge("TM_O_bond_length_A", tm_bonds, (216, 218, 220, 222, 224, 226, 228)))
    return result
