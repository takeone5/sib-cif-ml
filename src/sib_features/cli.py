"""Command line interface for stage-one extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .extractor import FeatureExtractor
from .fixed_charge import fixed_charge_unknown_elements
from .io import warning_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract disordered SIB structure features.")
    parser.add_argument("cif", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path; stdout if omitted.")
    parser.add_argument(
        "--warning-log",
        type=Path,
        default=Path("warnings.log"),
        help="Warning log path (default: warnings.log).",
    )
    args = parser.parse_args()
    with warning_log(args.warning_log):
        extractor = FeatureExtractor.from_files(args.cif, args.json_path)
        result = extractor.extract()
    unknown_elements = fixed_charge_unknown_elements(extractor.structure)
    if unknown_elements:
        print(
            f"Warning: feature 016 is null for {args.cif.name}; no fixed oxidation state is defined for: "
            f"{', '.join(unknown_elements)}",
            file=sys.stderr,
        )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
