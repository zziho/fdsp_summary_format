#!/usr/bin/env python3
"""Extract Trainer-style metric dicts from a torchrun log into JSONL."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


METRIC_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in METRIC_RE.finditer(text):
        try:
            rows.append(ast.literal_eval(match.group(0)))
        except (SyntaxError, ValueError):
            continue
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} metric rows -> {args.output}")


if __name__ == "__main__":
    main()
