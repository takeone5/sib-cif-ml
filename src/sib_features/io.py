"""Load the JSON extraction format and CIF structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pymatgen.core import Structure


def load_json(path: str | Path) -> dict[str, Any]:
    # Rietveld export files may contain a UTF-8 BOM (as in the supplied sample).
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_structure(path: str | Path) -> Structure:
    return Structure.from_file(str(path))
