"""Chemical classifications and weighted statistics used by the extractor."""

from __future__ import annotations

import math
from collections.abc import Iterable

from pymatgen.core import Element

ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
ANIONS = {"O", "F"}
LANTHANIDES = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}


def is_transition_metal(symbol: str) -> bool:
    """Return whether an element is a d-block transition metal or lanthanide TM-like center."""
    try:
        return bool(Element(symbol).is_transition_metal) or symbol in LANTHANIDES
    except (ValueError, TypeError):
        return False


def is_alkali(symbol: str) -> bool:
    return symbol in ALKALI


def is_alkaline_earth(symbol: str) -> bool:
    return symbol in ALKALINE_EARTH


def shannon_entropy(weights: dict[str, float]) -> float:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in weights.values() if v > 0)


def weighted_moments(values: Iterable[tuple[float, float]]) -> dict[str, float | None]:
    pairs = [(float(x), float(w)) for x, w in values if x is not None and w > 0]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return {"mean": None, "variance": None, "std": None}
    mean = sum(x * w for x, w in pairs) / total
    variance = sum(w * (x - mean) ** 2 for x, w in pairs) / total
    return {"mean": mean, "variance": variance, "std": math.sqrt(variance)}
