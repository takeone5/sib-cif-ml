"""Occupancy-aware bond-valence GII features without oxidation-state guessing."""

from __future__ import annotations

import math
from typing import Any

from pymatgen.analysis.bond_valence import BV_PARAMS
from pymatgen.core import Element, Structure

from .chemistry import is_transition_metal

BOND_B = 0.37
NEIGHBOR_RADIUS = 3.0


def _element_parameters(symbol: str) -> Any:
    """Return the BV_PARAMS entry for a symbol across pymatgen key formats."""
    direct = BV_PARAMS.get(symbol)
    if direct is not None:
        return direct
    for key, value in BV_PARAMS.items():
        if getattr(key, "symbol", str(key)) == symbol:
            return value
    return None


def _first_number(value: Any) -> float | None:
    """Find the first numeric value in older nested oxidation-state tables."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for child in value.values():
            result = _first_number(child)
            if result is not None:
                return result
    return None


def _r0_to_oxygen(symbol: str) -> float:
    """Get a baseline element–oxygen R0 from pymatgen's bond-valence table.

    Legacy pymatgen tables contain element/oxygen/oxidation-state entries; the
    first listed oxygen value is selected as a baseline. Newer tables expose
    element-specific ``c`` and ``r`` parameters, from which the pair R0 is
    derived. No oxidation state is inferred or attached to a structure site.
    """
    data = _element_parameters(symbol)
    if not isinstance(data, dict):
        raise KeyError(f"No BV_PARAMS entry for {symbol}")
    oxygen = _element_parameters("O")
    if isinstance(oxygen, dict) and all(k in data for k in ("c", "r")) and all(
        k in oxygen for k in ("c", "r")
    ):
        ci, ri = float(data["c"]), float(data["r"])
        cj, rj = float(oxygen["c"]), float(oxygen["r"])
        denominator = ri * ci + rj * cj
        if denominator <= 0:
            raise ValueError(f"Invalid BV_PARAMS c/r values for {symbol}-O")
        return ri + rj - ri * rj * (math.sqrt(ci) - math.sqrt(cj)) ** 2 / denominator

    # Select the oxygen sub-entry when present (old versions commonly use O2-).
    for key, child in data.items():
        if str(key).startswith("O"):
            result = _first_number(child)
            if result is not None:
                return result
    raise KeyError(f"No element–oxygen R0 found for {symbol}")


def _symbol(specie: Any) -> str:
    """Return the bare element symbol for Element and oxidation-state Species."""
    return getattr(specie, "symbol", str(specie))


def _site_bvs(structure: Structure, index: int) -> float:
    """Sum pairwise bond valences, weighted by center and neighbor occupancy."""
    site = structure[index]
    center_total = sum(float(amount) for amount in site.species.values())
    if center_total <= 0:
        raise ValueError(f"Site {index} has zero occupancy")
    bvs = 0.0
    for neighbor in structure.get_neighbors(site, r=NEIGHBOR_RADIUS):
        distance = float(neighbor.nn_distance)
        for center, center_occ in site.species.items():
            center_symbol = _symbol(center)
            for neighbor_species, neighbor_occ in neighbor.species.items():
                neighbor_symbol = _symbol(neighbor_species)
                if center_symbol == "O" and neighbor_symbol == "O":
                    continue
                if center_symbol == "O":
                    pair_r0 = _r0_to_oxygen(neighbor_symbol)
                elif neighbor_symbol == "O":
                    pair_r0 = _r0_to_oxygen(center_symbol)
                else:
                    continue
                bvs += (
                    float(center_occ)
                    * float(neighbor_occ)
                    * math.exp((pair_r0 - distance) / BOND_B)
                )
    # Normalize center occupancy so this is the BVS expected per occupied atom.
    return bvs / center_total


def calculate_bvs_gii_features(structure: Structure) -> dict[str, float | None]:
    """Return global instability index and charge-neutrality mean TM target.

    Composition-derived target valence is shared across all TM sites. Na is
    assigned target +1 and oxygen target magnitude 2. The two returned values
    are ``None`` for a structure that lacks a TM sublattice or has unsupported
    bond-valence parameters, allowing the batch feature extraction to continue.
    """
    amounts: dict[str, float] = {}
    for specie, amount in structure.composition.items():
        symbol = _symbol(specie)
        amounts[symbol] = amounts.get(symbol, 0.0) + float(amount)
    tm_amount = sum(amount for symbol, amount in amounts.items() if is_transition_metal(symbol))
    oxygen_amount = amounts.get("O", 0.0)
    if tm_amount <= 0 or oxygen_amount <= 0:
        return {"229_GII": None, "230_TM_average_target_valence": None}

    tm_target = (2.0 * oxygen_amount - amounts.get("Na", 0.0)) / tm_amount
    squared_error = 0.0
    try:
        for index, site in enumerate(structure):
            # Mixed-occupancy sites receive the occupancy-weighted average of
            # their applicable Na, O, and TM target values.
            target_parts: list[tuple[float, float]] = []
            for specie, occupancy in site.species.items():
                symbol = _symbol(specie)
                if symbol == "Na":
                    target_parts.append((1.0, float(occupancy)))
                elif symbol == "O":
                    target_parts.append((2.0, float(occupancy)))
                elif is_transition_metal(symbol):
                    target_parts.append((tm_target, float(occupancy)))
            target_weight = sum(weight for _, weight in target_parts)
            if target_weight <= 0:
                # No target was specified for non-Na, non-TM cations such as Li.
                continue
            target = sum(value * weight for value, weight in target_parts) / target_weight
            difference = _site_bvs(structure, index) - target
            squared_error += difference * difference
    except (KeyError, ValueError, OverflowError):
        return {"229_GII": None, "230_TM_average_target_valence": tm_target}

    # The requested GII denominator is all crystallographic sites in the cell.
    gii = math.sqrt(squared_error / len(structure)) if len(structure) else None
    return {"229_GII": gii, "230_TM_average_target_valence": tm_target}
