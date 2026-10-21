"""Load the JSON extraction format and CIF structures."""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pymatgen.core import Structure


@contextmanager
def warning_log(path: str | Path | None):
    """Redirect Python warnings to a log file for the duration of a run."""
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    warning_logger = logging.getLogger("py.warnings")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    warning_logger.addHandler(handler)
    warning_logger.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    try:
        yield
    finally:
        warning_logger.removeHandler(handler)
        handler.close()
        logging.captureWarnings(False)


def load_json(path: str | Path) -> dict[str, Any]:
    # Rietveld export files may contain a UTF-8 BOM (as in the supplied sample).
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_structure(path: str | Path) -> Structure:
    return Structure.from_file(str(path))


def load_cif_space_group_number(path: str | Path) -> int | None:
    """Read an explicitly declared space-group number from a CIF header."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"_symmetry_Int_Tables_number\s+['\"]?([0-9]+)",
        r"_space_group_IT_number\s+['\"]?([0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
