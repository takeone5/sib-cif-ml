"""Fixed-oxidation-state charge feature [016]."""

from __future__ import annotations

from pymatgen.core import Structure

from .chemistry import ANIONS

VARIABLE_TM_GROUP = frozenset({"Ni", "Co", "Mn", "Fe"})
FIXED_OXIDATION_STATES = {
    "Ti": 4,
    "Cu": 2,
    "Li": 1,
    "Mg": 2,
    "Al": 3,
    "Ca": 2,
    "Zn": 2,
    "K": 1,
    "Sn": 4,
    "Cr": 3,
    "La": 3,
    "Ce": 3,
    "F": -1,
    "Ta": 5,
    "B": 3,
    "Zr": 4,
    "Sc": 3,
    "Y": 3,
    "Ru": 4,
    "Ga": 3,
    "O": -2,
}


def _normalized_composition(structure: Structure) -> dict[str, float]:
    # ``Structure`` may retain oxidation states from CIF type symbols such as
    # ``O2-`` and ``Fe3+``.  The fixed-charge tables are keyed by bare element
    # symbols, so use ``symbol`` rather than ``str(specie)`` here.
    composition: dict[str, float] = {}
    for specie, amount in structure.composition.items():
        symbol = specie.symbol
        composition[symbol] = composition.get(symbol, 0.0) + float(amount)
    anion_amount = sum(composition.get(symbol, 0.0) for symbol in ANIONS)
    if anion_amount <= 0:
        raise ValueError("Fixed-charge feature requires an O/F anion sublattice")
    scale = 2.0 / anion_amount
    return {element: amount * scale for element, amount in composition.items()}


def load_reference_oxidation_states() -> dict[str, int]:
    """Return the fixed oxidation states bundled with the feature code."""
    return dict(FIXED_OXIDATION_STATES)


def variable_tm_group_total_charge(
    structure: Structure,
    *,
    variable_symbols: frozenset[str] = VARIABLE_TM_GROUP,
) -> float | None:
    """Calculate the charge contribution required from Ni/Co/Mn/Fe."""
    composition = _normalized_composition(structure)
    fixed_states = load_reference_oxidation_states()
    # Na is absent from the bundled list but is fixed at +1 for SIB.
    fixed_states["Na"] = 1
    group_amount = sum(composition.get(symbol, 0.0) for symbol in variable_symbols)
    if group_amount <= 0:
        return None
    unknown = [
        symbol for symbol, amount in composition.items()
        if amount > 1e-12 and symbol not in variable_symbols and symbol not in fixed_states
    ]
    if unknown:
        return None
    fixed_charge = sum(
        composition[symbol] * fixed_states[symbol]
        for symbol in composition
        if symbol not in variable_symbols
    )
    return -fixed_charge
