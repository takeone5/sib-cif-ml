#!/usr/bin/env python3
"""Calculate occupancy-weighted bond-valence sums and GII from CIF files.

Oxidation states are never assigned to CIF sites.  Bond-valence R0 values use
one representative entry from pymatgen's tabulated parameters as a baseline;
target TM valence is instead inferred from Na(+1), O(-2), and charge neutrality.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from pymatgen.analysis.bond_valence import BV_PARAMS
from pymatgen.core import Element, Structure

BOND_B = 0.37  # Empirical Brown-Altermatt decay parameter, in angstroms.
NEIGHBOR_RADIUS = 3.0
OXYGEN_SYMBOL = "O"


def _find_numeric_parameter(tree: Any) -> float | None:
    """Find a numeric R0 value within a possibly nested BV_PARAMS entry."""
    if isinstance(tree, (int, float)):
        return float(tree)
    if isinstance(tree, dict):
        # Prefer the first stable/insertion-ordered representative value.
        for value in tree.values():
            found = _find_numeric_parameter(value)
            if found is not None:
                return found
    return None


def _parameter_entry(symbol: str) -> Any:
    """Find a BV_PARAMS key by element symbol across pymatgen versions."""
    direct = BV_PARAMS.get(symbol)
    if direct is not None:
        return direct
    for key, value in BV_PARAMS.items():
        if getattr(key, "symbol", str(key)) == symbol:
            return value
    return None


def base_r0_for_element(symbol: str) -> float:
    """Return a baseline element–oxygen R0 without assigning site valences.

    pymatgen versions encode oxidation states at different levels. We first
    locate the element's table, prefer its oxygen entry (including O2-), and
    recursively select its first representative numeric R0. A clear error is
    raised when the installed table has no usable oxygen parameter.
    """
    element_data = _parameter_entry(symbol)
    if not isinstance(element_data, dict):
        raise KeyError(f"No BV_PARAMS entry for element {symbol!r}")
    # Recent pymatgen releases store Gagné–Hawthorne element parameters as
    # ``{"c": ..., "r": ...}`` instead of oxidation-state-specific R0s.
    # Derive the element–oxygen R0 from those tabulated c/r parameters.
    oxygen_data = _parameter_entry(OXYGEN_SYMBOL)
    if isinstance(oxygen_data, dict) and all(k in element_data for k in ("c", "r")) and all(
        k in oxygen_data for k in ("c", "r")
    ):
        ci, ri = float(element_data["c"]), float(element_data["r"])
        cj, rj = float(oxygen_data["c"]), float(oxygen_data["r"])
        return ri + rj - (ri * rj * (math.sqrt(ci) - math.sqrt(cj)) ** 2) / (ri * ci + rj * cj)
    oxygen_entries = [
        value for key, value in element_data.items()
        if str(key).startswith("O") or str(key).startswith("O2")
    ]
    # Some table layouts put oxygen below an oxidation-state key; recursively
    # search those layouts as a compatibility fallback.
    for entry in oxygen_entries:
        result = _find_numeric_parameter(entry)
        if result is not None:
            return result
    for key, value in element_data.items():
        if str(key).lower() in {"o", "o2-", "oxygen"}:
            result = _find_numeric_parameter(value)
            if result is not None:
                return result
    # Common pymatgen representation is element -> anion -> oxidation -> R0,
    # but in some releases the intermediate keys are ordered differently.
    for value in element_data.values():
        result = _find_numeric_parameter(value)
        if result is not None:
            return result
    raise KeyError(f"No element–oxygen R0 parameter found for {symbol!r}")


def _species_symbol(specie: Any) -> str:
    """Get a bare element symbol from Element or Species objects."""
    return getattr(specie, "symbol", str(specie))


def site_bvs(structure: Structure, site_index: int) -> float:
    """Compute one site's occupancy-weighted BVS using neighbors within 3 Å.

    Every center/neighbor species pair is visited. The bond contribution is
    multiplied by both occupancies. Center occupancy is then normalized out to
    report the expected BVS per occupied center atom; neighbor occupancy remains
    as the probability that the neighboring species is present.
    """
    site = structure[site_index]
    center_occupancy = sum(float(amount) for amount in site.species.values())
    if center_occupancy <= 0:
        raise ValueError(f"Site {site_index} has zero total occupancy")
    total = 0.0
    for neighbor in structure.get_neighbors(site, r=NEIGHBOR_RADIUS):
        distance = float(neighbor.nn_distance)
        for center_species, center_amount in site.species.items():
            center_symbol = _species_symbol(center_species)
            for neighbor_species, neighbor_amount in neighbor.species.items():
                # Use the element–oxygen baseline table for any pair involving O.
                # Non-oxygen pairs are ignored because this workflow targets
                # oxide cathodes and their dominant TM/Na–O bond valences.
                neighbor_symbol = _species_symbol(neighbor_species)
                if center_symbol == OXYGEN_SYMBOL and neighbor_symbol == OXYGEN_SYMBOL:
                    continue
                if center_symbol == OXYGEN_SYMBOL and neighbor_symbol != OXYGEN_SYMBOL:
                    r0 = base_r0_for_element(neighbor_symbol)
                elif neighbor_symbol != OXYGEN_SYMBOL:
                    continue
                else:
                    r0 = base_r0_for_element(center_symbol)
                contribution = math.exp((r0 - distance) / BOND_B)
                total += float(center_amount) * float(neighbor_amount) * contribution
    return total / center_occupancy


def target_valences(structure: Structure) -> tuple[list[float], float]:
    """Return per-site targets and the charge-neutrality average TM valence.

    Na sites target +1 and O sites target an absolute valence of 2. All other
    elements are treated as transition-metal (TM) sites; their total required
    positive charge is obtained from neutrality and divided by total TM amount.
    Thus mixed TM species share one composition-derived average target.
    """
    amounts: dict[str, float] = {}
    for specie, amount in structure.composition.items():
        symbol = _species_symbol(specie)
        amounts[symbol] = amounts.get(symbol, 0.0) + float(amount)
    na_amount = amounts.get("Na", 0.0)
    oxygen_amount = amounts.get("O", 0.0)
    tm_amount = sum(amount for symbol, amount in amounts.items() if symbol not in {"Na", "O"})
    if tm_amount <= 0:
        raise ValueError("Cannot infer TM target valence: no non-Na, non-O species found")
    tm_average = (2.0 * oxygen_amount - na_amount) / tm_amount
    targets: list[float] = []
    for site in structure:
        symbols = {_species_symbol(specie) for specie in site.species}
        if symbols == {"Na"}:
            targets.append(1.0)
        elif symbols == {"O"}:
            targets.append(2.0)
        else:
            targets.append(tm_average)
    return targets, tm_average


def calculate_structure(cif_path: Path) -> tuple[list[dict[str, Any]], float, float]:
    """Load one CIF and return its site records, GII, and average TM target."""
    structure = Structure.from_file(cif_path)
    targets, tm_target = target_valences(structure)
    records: list[dict[str, Any]] = []
    squared_error = 0.0
    for index, (site, target) in enumerate(zip(structure, targets)):
        bvs = site_bvs(structure, index)
        error = bvs - target
        squared_error += error * error
        records.append({
            "cif": str(cif_path),
            "site_index": index,
            "species": str(site.species),
            "fractional_coordinates": ";".join(f"{x:.6f}" for x in site.frac_coords),
            "BVS_calc": bvs,
            "target_valence": target,
            "difference": error,
        })
    gii = math.sqrt(squared_error / len(structure))
    for record in records:
        record["GII"] = gii
        record["TM_average_target"] = tm_target
    return records, gii, tm_target


def find_cif_files(input_path: Path) -> list[Path]:
    """Accept either a single CIF or a directory, recursively discovering CIFs."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".cif":
            raise ValueError(f"Input file is not a .cif: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".cif")


def main() -> None:
    """Parse CLI options and write one row per crystallographic site to CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="*", type=Path, default=[Path("origin_data_v4")],
                        help="One or more CIF files or directories (default: ./origin_data_v4)")
    parser.add_argument("-o", "--output", type=Path, default=Path("bvs_gii_results.csv"),
                        help="CSV output path")
    args = parser.parse_args()
    # Expand each provided file/directory independently, then deduplicate paths
    # so overlapping directory arguments do not calculate a CIF twice.
    cif_files = sorted({cif for input_path in args.input for cif in find_cif_files(input_path)})
    if not cif_files:
        raise SystemExit(f"No CIF files found under {', '.join(map(str, args.input))}")
    fields = ["cif", "site_index", "species", "fractional_coordinates", "BVS_calc",
              "target_valence", "difference", "GII", "TM_average_target"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cif_path in cif_files:
            try:
                records, gii, tm_target = calculate_structure(cif_path)
                writer.writerows(records)
                print(f"{cif_path}: GII={gii:.6f} v(TM)={tm_target:.4f}")
            except Exception as exc:
                # Keep processing the dataset and make failures visible in the
                # output rather than silently dropping a structure.
                print(f"ERROR {cif_path}: {exc}")
    print(f"Wrote site-level results to {args.output}")


if __name__ == "__main__":
    main()
