"""First-stage feature extractor for disordered Rietveld structures.

The JSON site list is used when available because it retains explicit Wyckoff
labels. Occupancies are never expanded into ordered configurations: every
calculation uses the fractional species occupancies on each crystallographic
site and includes the unoccupied fraction as ``Vacancy`` where applicable.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pymatgen.core import Element, Structure

from .chemistry import (
    ALKALI,
    ALKALINE_EARTH,
    ANIONS,
    is_alkali,
    is_alkaline_earth,
    is_transition_metal,
    shannon_entropy,
    weighted_moments,
)
from .thickness import calculate_layer_thicknesses
from .local import extract_local_features
from .fixed_charge import variable_tm_group_total_charge


PERIODIC_SYMBOLS = [Element.from_Z(z).symbol for z in range(1, 119)]


class FeatureExtractor:
    """Extract composition and global features from one JSON/CIF pair."""

    def __init__(self, structure: Structure, metadata: dict[str, Any] | None = None):
        self.structure = structure
        self.metadata = metadata or {}
        self.json_sites = self.metadata.get("sites", [])

    @classmethod
    def from_files(cls, cif_path: str | Path, json_path: str | Path | None = None):
        from .io import load_json

        metadata = load_json(json_path) if json_path else {}
        return cls(Structure.from_file(str(cif_path)), metadata)

    def _site_rows(self) -> list[dict[str, Any]]:
        """Return JSON rows, or CIF sites with an inferred row label."""
        if self.json_sites:
            rows = []
            for site in self.json_sites:
                species = {s["element"]: float(s.get("occ", 0.0)) for s in site.get("species", [])}
                rows.append({"wyckoff": site.get("wyckoff"), "species": species, "coords": site.get("coords")})
            return rows
        rows = []
        for site in self.structure:
            species = {str(sp.symbol): float(occ) for sp, occ in site.species.items()}
            rows.append({"wyckoff": None, "species": species, "coords": list(site.frac_coords)})
        return rows

    @staticmethod
    def _occupancy(row: dict[str, Any], symbol: str) -> float:
        return float(row["species"].get(symbol, 0.0))

    @staticmethod
    def _vacancy(row: dict[str, Any]) -> float:
        return max(0.0, 1.0 - sum(row["species"].values()))

    @staticmethod
    def _site_multiplicity(row: dict[str, Any]) -> int:
        """Return a JSON site's Wyckoff multiplicity, or one for expanded sites."""
        match = re.match(r"(\d+)", str(row.get("wyckoff", "")))
        return int(match.group(1)) if match else 1

    def _na_layer_composition(self, na_rows: list[dict[str, Any]]) -> dict[str, float]:
        """Return Na-layer species fractions per ``Na_y X_x TM O2`` formula."""
        anion_total = sum(self.structure.composition.get(a, 0.0) for a in ANIONS)
        formula_units = anion_total / 2.0 if anion_total else 1.0
        weights: dict[str, float] = defaultdict(float)
        for row in na_rows:
            scale = self._site_multiplicity(row) / formula_units
            for symbol, occ in row["species"].items():
                weights[symbol] += occ * scale
        occupied = sum(weights.values())
        weights["Vacancy"] = max(0.0, 1.0 - occupied)
        return weights

    @staticmethod
    def _element_fractions(composition: dict[str, float]) -> dict[str, float]:
        """Normalize occupied element amounts, excluding non-element entries."""
        occupied = sum(value for symbol, value in composition.items() if symbol != "Vacancy")
        if not occupied:
            return {}
        return {
            symbol: value / occupied
            for symbol, value in composition.items()
            if symbol != "Vacancy" and value
        }

    def _normalized_composition(self) -> dict[str, float]:
        comp = {str(k): float(v) for k, v in self.structure.composition.items()}
        oxygen = sum(comp.get(a, 0.0) for a in ANIONS)
        scale = 2.0 / oxygen if oxygen else 1.0
        return {k: v * scale for k, v in comp.items()}

    def _na_rows(self) -> list[dict[str, Any]]:
        return [r for r in self._site_rows() if self._occupancy(r, "Na") > 0]

    def _tm_rows(self) -> list[dict[str, Any]]:
        rows = []
        for row in self._site_rows():
            if any(is_transition_metal(s) for s in row["species"]):
                rows.append(row)
        return rows

    def _tm_composition(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in self._tm_rows():
            for symbol, occ in row["species"].items():
                if is_transition_metal(symbol) or is_alkali(symbol) or is_alkaline_earth(symbol):
                    totals[symbol] += occ
        total = sum(totals.values())
        return {s: v / total for s, v in totals.items()} if total else {}

    def _atomic_property_moments(self, property_name: str) -> dict[str, float | None]:
        fractions = self._tm_composition()
        values = []
        for symbol, fraction in fractions.items():
            value = getattr(Element(symbol), property_name, None)
            if value is not None:
                values.append((float(value), fraction))
        return weighted_moments(values)

    def extract(self) -> dict[str, Any]:
        rows = self._site_rows()
        na_rows = self._na_rows()
        tm_rows = self._tm_rows()
        na_weights = self._na_layer_composition(na_rows)
        tm_weights: dict[str, float] = defaultdict(float)
        for row in tm_rows:
            for symbol, occ in row["species"].items():
                if is_transition_metal(symbol) or is_alkali(symbol) or is_alkaline_earth(symbol):
                    tm_weights[symbol] += occ
        tm_fraction = self._tm_composition()
        na_fraction = self._element_fractions(na_weights)
        comp = self._normalized_composition()
        sg_number = self.metadata.get("space_group", {}).get("number", self.structure.get_space_group_info()[1])
        radius = self._atomic_property_moments("atomic_radius")
        eneg = self._atomic_property_moments("X")
        lattice = self.structure.lattice
        try:
            thickness = calculate_layer_thicknesses(self.structure, self.metadata)
        except (KeyError, TypeError, ValueError, IndexError):
            thickness = {
                "104_O_TM_O_distance_A": None,
                "105_O_Na_O_distance_A": None,
                "106_O_Na_O_to_O_TM_O_ratio": None,
        }
        features: dict[str, Any] = {
            "001_Na_layer_entropy": shannon_entropy(na_weights),
            "002_TM_layer_entropy": shannon_entropy(tm_weights),
            "003_Na_content_per_O2": comp.get("Na", 0.0),
            "004_TM_layer_element_fractions": {s: tm_fraction.get(s, 0.0) for s in PERIODIC_SYMBOLS},
            "005_Na_layer_element_fractions": {s: na_fraction.get(s, 0.0) for s in PERIODIC_SYMBOLS},
            "006_Na_TM_entropy_product": shannon_entropy(na_weights) * shannon_entropy(tm_weights),
            "007_Li_fraction_in_TM_layer": tm_fraction.get("Li", 0.0),
            "008_alkali_fraction_in_TM_layer": sum(tm_fraction.get(s, 0.0) for s in ALKALI),
            "009_Mg_fraction_in_TM_layer": tm_fraction.get("Mg", 0.0),
            "010_alkaline_earth_fraction_in_TM_layer": sum(tm_fraction.get(s, 0.0) for s in ALKALINE_EARTH),
            "011_alkali_plus_alkaline_earth_fraction_in_TM_layer": sum(
                tm_fraction.get(s, 0.0) for s in ALKALI | ALKALINE_EARTH
            ),
            "012_TM_atomic_radius_mean": radius["mean"],
            "013_TM_atomic_radius_std": radius["std"],
            "014_TM_electronegativity_mean": eneg["mean"],
            "015_TM_electronegativity_std": eneg["std"],
            "016_Ni_Co_Mn_Fe_required_charge": variable_tm_group_total_charge(self.structure),
            "101_space_group_number": sg_number,
            "102_cell_c_over_a": lattice.c / lattice.a,
            "103_cell_volume_A3": lattice.volume,
        } 
        features.update(thickness)
        features.update(extract_local_features(self.structure))
        return features
