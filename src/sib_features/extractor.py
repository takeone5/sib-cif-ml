"""First-stage feature extractor for disordered Rietveld structures.

The JSON site list is used when available because it retains explicit Wyckoff
labels. Occupancies are never expanded into ordered configurations: every
calculation uses the fractional species occupancies on each crystallographic
site and includes the unoccupied fraction as ``Vacancy`` where applicable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _matminer_data():
    from matminer.utils.data import CohesiveEnergyData, MagpieData

    return MagpieData(), CohesiveEnergyData()


def _matminer_property_moments(
    fractions: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    """Return TM-layer weighted means/stds for the requested Matminer data."""
    magpie, cohesive = _matminer_data()
    property_sources = {
        "AtomicRadius": lambda symbol: magpie.get_elemental_property(Element(symbol), "AtomicRadius"),
        "Electronegativity": lambda symbol: magpie.get_elemental_property(
            Element(symbol), "Electronegativity"
        ),
        "NdValence": lambda symbol: magpie.get_elemental_property(Element(symbol), "NdValence"),
        "CohesiveEnergy": lambda symbol: cohesive.get_elemental_property(symbol),
    }
    result: dict[str, dict[str, float | None]] = {}
    for name, get_value in property_sources.items():
        values = []
        for symbol, fraction in fractions.items():
            try:
                values.append((float(get_value(symbol)), fraction))
            except (KeyError, TypeError, ValueError):
                continue
        moments = weighted_moments(values)
        if moments["mean"] is None:
            moments["avg_dev"] = None
        else:
            total = sum(weight for _, weight in values)
            moments["avg_dev"] = (
                sum(weight * abs(value - moments["mean"]) for value, weight in values) / total
                if total > 0
                else None
            )
        result[name] = moments
    return result


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
        comp = self._normalized_composition()
        na_entropy = shannon_entropy(na_weights)
        tm_entropy = shannon_entropy(tm_weights)
        sg_number = self.metadata.get("space_group", {}).get("number", self.structure.get_space_group_info()[1])
        try:
            is_prismatic = {160: 1, 166: 0, 194: 1}.get(int(sg_number))
            is_2layer = {160: 0, 166: 0, 194: 1}.get(int(sg_number))
        except (TypeError, ValueError):
            is_prismatic = None
            is_2layer = None
        matminer_properties = _matminer_property_moments(tm_fraction)
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
            "001_Na_layer_entropy": round(na_entropy, 3),
            "002_TM_layer_entropy": round(tm_entropy, 3),
            "003_Na_content_per_O2": comp.get("Na", 0.0),
            "006_Na_TM_entropy_sum": round(na_entropy + tm_entropy, 3),
            "007_Li_fraction_in_TM_layer": tm_fraction.get("Li", 0.0),
            "008_alkali_fraction_in_TM_layer": sum(tm_fraction.get(s, 0.0) for s in ALKALI),
            "009_Mg_fraction_in_TM_layer": tm_fraction.get("Mg", 0.0),
            "010_alkaline_earth_fraction_in_TM_layer": sum(tm_fraction.get(s, 0.0) for s in ALKALINE_EARTH),
            "016_Ni_Co_Mn_Fe_required_charge": variable_tm_group_total_charge(self.structure),
            "101_space_group_number": sg_number,
            "102_cell_c_over_a": lattice.c / lattice.a,
            "103_cell_volume_A3": lattice.volume,
        } 
        features.update(thickness)
        features.update(
            {
                "107_AtomicRadius_mean": matminer_properties["AtomicRadius"]["mean"],
                "108_AtomicRadius_avg_dev": matminer_properties["AtomicRadius"]["avg_dev"],
                "109_Electronegativity_mean": matminer_properties["Electronegativity"]["mean"],
                "110_Electronegativity_avg_dev": matminer_properties["Electronegativity"]["avg_dev"],
                "111_NdValence": matminer_properties["NdValence"]["mean"],
                "112_CohesiveEnergy": matminer_properties["CohesiveEnergy"]["mean"],
                "113_is_Prismatic": is_prismatic,
                "114_is_2Layer": is_2layer,
            }
        )
        features.update(extract_local_features(self.structure))
        return features
