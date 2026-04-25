"""Print a side-by-side markdown table comparing eval reports.

Usage:
    uv run python scripts/compare_eval.py reports/era1_3small.json reports/era1_3large.json reports/era1_noise.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path)
    args = p.parse_args(argv)

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if not reports:
        return 1

    headers = [r.get("run_name", path.stem) for r, path in zip(reports, args.reports)]
    metric_names: list[str] = []
    for section in ("ir_metrics", "deepeval_metrics"):
        for r in reports:
            for name in r.get(section, {}):
                if name not in metric_names:
                    metric_names.append(name)

    if not metric_names:
        print("no metrics found in reports", file=sys.stderr)
        return 1

    print("| Metric | " + " | ".join(headers) + " |")
    print("|--------|" + "|".join(["---"] * len(headers)) + "|")
    for name in metric_names:
        row = [name]
        for r in reports:
            val = r.get("ir_metrics", {}).get(name) or r.get("deepeval_metrics", {}).get(name)
            row.append(f"{val:.4f}" if val is not None else "—")
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
