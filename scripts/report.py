# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Turn the ablation CSV into the frontier table the decision needs.

Usage:
    uv run scripts/report.py results/ablation.csv > results/REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

BUDGETS = (0.005, 0.01, 0.02, 0.05)
"""Scan budgets, as a fraction of the corpus, at which recall is compared."""

SEEDED = re.compile(r"-s\d+")


@dataclass(frozen=True)
class Point:
    """One measured operating point, as read back from the CSV."""

    encoder: str
    depth: int
    n_probes: int
    recall: float
    scan_mean: float
    scan_p95: float
    n_ranges: float

    @property
    def family(self) -> str:
        """Seeded variants collapse into one family so seeds can be pooled."""
        return SEEDED.sub("", self.encoder)


def read_points(path: Path) -> list[Point]:
    """Read every operating point out of the ablation CSV."""
    with path.open(newline="") as handle:
        return [
            Point(
                encoder=row["encoder"],
                depth=int(row["depth"]),
                n_probes=int(row["n_probes"]),
                recall=float(row["recall"]),
                scan_mean=float(row["scan_mean"]),
                scan_p95=float(row["scan_p95"]),
                n_ranges=float(row["n_ranges"]),
            )
            for row in csv.DictReader(handle)
        ]


def best_within(points: list[Point], budget: float) -> Point | None:
    """Return the highest-recall point whose *mean* scan stays inside a budget.

    The budget binds on the average query, not on every query, which is why the
    rendered cell carries the p95 alongside.
    """
    affordable = [p for p in points if p.scan_mean <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda p: (p.recall, -p.n_ranges))


def _cell(points: list[Point], budget: float) -> str:
    """Format one family's best point at one budget, pooling seeds."""
    per_seed = {}
    for point in points:
        per_seed.setdefault(point.encoder, []).append(point)
    bests = [best_within(group, budget) for group in per_seed.values()]
    found = [b for b in bests if b is not None]
    if not found:
        return "—"
    recalls = [b.recall for b in found]
    ranges = statistics.mean(b.n_ranges for b in found)
    tail = statistics.mean(b.scan_p95 for b in found)
    detail = f"({ranges:.0f}r · p95 {tail:.1%})"
    if len(found) == 1:
        return f"{recalls[0]:.3f} {detail}"
    spread = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
    return f"{statistics.mean(recalls):.3f}±{spread:.3f} {detail}"


def read_notes(path: Path) -> dict[str, float]:
    """Read the run's scalar record, if the run wrote one."""
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text())


def render(points: list[Point], notes: dict[str, float]) -> str:
    """Render the recall-within-budget table, with range counts in brackets."""
    families: dict[str, list[Point]] = {}
    for point in points:
        families.setdefault(point.family, []).append(point)
    header = " | ".join(f"mean scan ≤{b:.1%}" for b in BUDGETS)
    lines = [
        "# HILDA representation ablation",
        "",
        "recall@10 reachable while the *mean* per-query scan stays inside a",
        "budget. `Nr` is the mean number of separate B-tree ranges that scan",
        "takes; `p95` is the 95th-percentile per-query scan fraction, which is",
        "what a single unlucky query actually pays.",
        "",
        f"| encoder | {header} |",
        "|---|" + "---|" * len(BUDGETS),
    ]
    ordered = sorted(
        families.items(),
        key=lambda item: (
            -(best_within(item[1], BUDGETS[-1]) or Point("", 0, 0, 0, 0, 0)).recall
        ),
    )
    for name, group in ordered:
        cells = " | ".join(_cell(group, budget) for budget in BUDGETS)
        lines.append(f"| `{name}` | {cells} |")
    if notes:
        lines.extend(["", "## Run record", ""])
        lines.extend(f"- {key}: {value:.4f}" for key, value in sorted(notes.items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Render the report to stdout."""
    parser = argparse.ArgumentParser(description="Summarise the ablation CSV")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.write(render(read_points(args.csv), read_notes(args.csv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
