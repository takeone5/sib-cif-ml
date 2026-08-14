"""Command line interface for stage-one extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extractor import FeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract disordered SIB structure features.")
    parser.add_argument("cif", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path; stdout if omitted.")
    args = parser.parse_args()
    result = FeatureExtractor.from_files(args.cif, args.json_path).extract()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
