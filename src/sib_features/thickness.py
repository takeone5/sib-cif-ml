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
                "species": {s["element"]: float(s.get("occ", 0.0)) for s in row.get("species", [])},
                "coords": row.get("coords", []),
            }
            for row in metadata["sites"]
        ]
    return [
        {"species": {str(sp.symbol): float(occ) for sp, occ in site.species.items()}, "coords": list(site.frac_coords)}
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


def _space_group(metadata: dict[str, Any], structure: Structure) -> str:
    symbol = metadata.get("space_group", {}).get("symbol")
    return symbol or structure.get_space_group_info()[0]


def calculate_layer_thicknesses(structure: Structure, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    """Return O-Na-O and O-TM-O slab distances and their ratio.

    Supported space groups and equations intentionally match the existing
    spreadsheet: ``P63/mmc`` and ``R-3m`` only.
    """
    metadata = metadata or {}
    rows = _site_rows(structure, metadata)
    c = float(structure.lattice.c)
    zo = _representative_anion_z(rows)
    sg_symbol = _space_group(metadata, structure)

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
        na_z = float(na_rows[0]["coords"][2])
        if math.isclose(na_z, 0.0, abs_tol=1e-5) or math.isclose(na_z, 1.0, abs_tol=1e-5):
            d_na = 2 * (1.0 / 3.0 - zo) * c
        elif math.isclose(na_z, 0.5, abs_tol=1e-5):
            d_na = 2 * (zo - 1.0 / 6.0) * c
        else:
            raise ValueError(f"R-3m Na z coordinate is not 0 or 0.5: {na_z}")
        d_tm = c / 3.0 - d_na
    else:
        raise ValueError(f"Unsupported space group for layer thickness: {sg_symbol}")

    return {
        "104_O_TM_O_distance_A": d_tm,
        "105_O_Na_O_distance_A": d_na,
        "106_O_Na_O_to_O_TM_O_ratio": d_na / d_tm if d_tm else None,
    }
