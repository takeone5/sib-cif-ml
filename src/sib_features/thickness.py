"""Layer-thickness features [104]-[106].

This follows the coordinate-based equations used by the supplied spreadsheet
calculator. It deliberately uses the representative anion z coordinate from
the Rietveld model; partial occupancies do not alter the geometric distance.
"""

from __future__ import annotations

import math
from typing import Any

from pymatgen.core import Structure


def _site_rows(structure: Structure, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if metadata.get("sites"):
        return [
            {
                "wyckoff": row.get("wyckoff"),
                "species": {s["element"]: float(s.get("occ", 0.0)) for s in row.get("species", [])},
                "coords": row.get("coords", []),
            }
            for row in metadata["sites"]
        ]
    return [
        {
            "wyckoff": None,
            "species": {str(sp.symbol): float(occ) for sp, occ in site.species.items()},
            "coords": list(site.frac_coords),
        }
        for site in structure
    ]


def _has_element(row: dict[str, Any], symbols: set[str]) -> bool:
    return any(symbol in symbols and occupancy > 0 for symbol, occupancy in row["species"].items())


def _representative_anion_z(rows: list[dict[str, Any]]) -> float:
    for row in rows:
        if _has_element(row, {"O", "F"}):
            coords = row["coords"]
            if len(coords) >= 3:
                return float(coords[2])
    raise ValueError("No O/F representative site with a z coordinate was found")


def _r3m_3a_anion_zs(rows: list[dict[str, Any]]) -> list[float]:
    """Return z coordinates of O/F sites at the R3m 3a origin line."""
    zs = []
    for row in rows:
        coords = row.get("coords", [])
        wyckoff = str(row.get("wyckoff") or "").lower().replace(" ", "")
        if wyckoff and wyckoff != "3a":
            continue
        if len(coords) < 3 or not (math.isclose(float(coords[0]), 0.0, abs_tol=1e-5) and math.isclose(float(coords[1]), 0.0, abs_tol=1e-5)):
            continue
        if _has_element(row, {"O", "F"}):
            zs.append(float(coords[2]))
    return zs


def _r3m_3a_thicknesses(rows: list[dict[str, Any]], c: float) -> tuple[float, float]:
    """Calculate slab distances from the two O/F 3a coordinates in R3m."""
    zs = _r3m_3a_anion_zs(rows)
    if len(zs) != 2:
        raise ValueError(f"R3m requires exactly two O/F 3a z coordinates, found {len(zs)}")

    z1, z2 = zs
    opposite_sign = math.isclose(z1, -z2, abs_tol=1e-5)
    if opposite_sign:
        z = max(z1, z2)
        if z <= 1.0 / 3.0:
            raise ValueError(f"R3m opposite-sign O/F z coordinate must be greater than 1/3: {z}")
        d_tm = 2 * (z - 1.0 / 3.0) * c
    else:
        # The two sites may appear in either order in the input file.
        z1, z2 = max(zs), min(zs)
    if not opposite_sign and z1 > 0 and z2 > 0 and math.isclose(z1 + z2, 1.0, abs_tol=1e-5):
        d_tm = (z1 - z2) * c
    elif not opposite_sign:
        raise ValueError(f"Unsupported R3m O/F 3a z coordinates: {z1}, {z2}")
    return d_tm, c / 3.0 - d_tm


def _space_group(metadata: dict[str, Any]) -> str:
    space_group = metadata.get("space_group", {})
    symbol = space_group.get("symbol")
    if symbol:
        return symbol
    return {160: "R3m", 166: "R-3m", 194: "P63/mmc"}.get(space_group.get("number"), "")


def calculate_layer_thicknesses(
    structure: Structure | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """Return O-Na-O and O-TM-O slab distances and their ratio.

    Geometry inputs come only from JSON metadata so symmetry-expanded CIF
    coordinates cannot be mistaken for representative site coordinates. The
    structure argument remains optional for compatibility with existing callers
    but is intentionally not used. Missing JSON geometry returns null features.
    """
    metadata = metadata or {}
    missing = {
        "104_O_TM_O_distance_A": None,
        "105_O_Na_O_distance_A": None,
        "106_O_Na_O_to_O_TM_O_ratio": None,
    }
    lattice_parameters = metadata.get("lattice_parameters")
    if not isinstance(lattice_parameters, dict) or lattice_parameters.get("c") is None:
        return missing
    if not metadata.get("sites") or not isinstance(metadata.get("space_group"), dict):
        return missing

    rows = _site_rows(structure, metadata)
    c = float(lattice_parameters["c"])
    zo = _representative_anion_z(rows)
    sg_symbol = _space_group(metadata)

    if sg_symbol == "P63/mmc":
        if abs(zo - 0.41) <= 0.05:
            d_tm = 2 * (0.5 - zo) * c
        elif abs(zo - 0.91) <= 0.05:
            d_tm = 2 * (1.0 - zo) * c
        elif abs(zo - 0.08) <= 0.05:
            d_tm = 2 * zo * c
        elif abs(zo - 0.58) <= 0.05:
            d_tm = 2 * (zo - 0.5) * c
        else:
            raise ValueError(f"Unsupported P63/mmc anion z coordinate: {zo}")
        d_na = c / 2.0 - d_tm
    elif sg_symbol == "R-3m":
        na_rows = [row for row in rows if _has_element(row, {"Na"})]
        if not na_rows:
            raise ValueError("R-3m structure has no Na site")
        # A CIF parser expands the asymmetric unit into symmetry-equivalent
        # sites. The first expanded Na may therefore be at z=1/6 or 5/6 even
        # when the representative CIF coordinate is z=0 or 1/2. Prefer one
        # of the representative coordinates used by the equations below.
        na_zs = [float(row["coords"][2]) for row in na_rows if len(row.get("coords", [])) >= 3]
        representative_zs = [
            z for z in na_zs
            if math.isclose(z, 0.0, abs_tol=1e-5)
            or math.isclose(z, 1.0, abs_tol=1e-5)
            or math.isclose(z, 0.5, abs_tol=1e-5)
        ]
        if not representative_zs:
            raise ValueError(f"R-3m Na has no representative z coordinate (found: {na_zs})")
        na_z = representative_zs[0]
        if math.isclose(na_z, 0.0, abs_tol=1e-5) or math.isclose(na_z, 1.0, abs_tol=1e-5):
            d_na = 2 * (1.0 / 3.0 - zo) * c
        elif math.isclose(na_z, 0.5, abs_tol=1e-5):
            d_na = 2 * (zo - 1.0 / 6.0) * c
        else:
            raise ValueError(f"R-3m Na z coordinate is not 0 or 0.5: {na_z}")
        d_tm = c / 3.0 - d_na
    elif sg_symbol.replace(" ", "") == "R3m" or metadata.get("space_group", {}).get("number") == 160:
        d_tm, d_na = _r3m_3a_thicknesses(rows, c)
    else:
        raise ValueError(f"Unsupported space group for layer thickness: {sg_symbol}")

    return {
        "104_O_TM_O_distance_A": d_tm,
        "105_O_Na_O_distance_A": d_na,
        "106_O_Na_O_to_O_TM_O_ratio": d_na / d_tm if d_tm else None,
    }
