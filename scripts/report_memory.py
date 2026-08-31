# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Turn the memory-pressure JSON into the tables the write-up reports.

One table per corpus size, one row per plan and cache state, columns ordered so
the comparison the experiment exists to make -- HILDA against HNSW at a stated
recall, as the cap tightens -- reads left to right.

Several files can be given, so a sweep extended to tighter caps reads as one
table. Runs are merged only when they agree on the working set and on both
operating points; disagreement means the two halves are not the same
experiment and the merge is refused rather than papered over.

Usage:
    uv run scripts/report_memory.py results/memory_pressure.json [more.json ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MEBIBYTE = 1024 * 1024
"""The caps are set in mebibytes, so every size is reported in them too."""
HEADER = (
    "| plan | recall | cap | cap / working set | cache "
    "| p50 | p95 | shared read | read time |"
)
RULE = "|---|---|---|---|---|---|---|---|---|"


def _row(cell: dict[str, object]) -> str:
    """Render one measured cell as a table row."""
    return (
        f"| `{cell['plan']}` "
        f"| {cell['recall']:.3f} "
        f"| {cell['limit_mb']} MiB "
        f"| {cell['achieved_ratio']:.2f} "
        f"| {cell['cache']} "
        f"| {cell['latency_p50_ms']:.2f} ms "
        f"| {cell['latency_p95_ms']:.2f} ms "
        f"| {cell['shared_read']:.0f} "
        f"| {cell['read_ms']:.2f} ms |"
    )


def _order(cell: dict[str, object]) -> tuple[float, int, str]:
    """Sort caps from loosest to tightest, then plan, then cache state."""
    return (-float(cell["achieved_ratio"]), 0, str(cell["plan"]) + str(cell["cache"]))


def render_size(size: dict[str, object]) -> list[str]:
    """Render one corpus size: what it costs to hold, then what it costs to query."""
    indexes = size["index_bytes"]
    setting = size["hilda_setting"]
    lines = [
        f"### {size['rows']:,} rows",
        "",
        (
            f"Working set {size['working_set_bytes'] / MEBIBYTE:.0f} MiB: "
            f"table {size['table_bytes'] / MEBIBYTE:.0f} MiB, "
            f"`hnsw_idx` {indexes['hnsw_idx'] / MEBIBYTE:.0f} MiB, "
            f"`code_idx` {indexes['code_idx'] / MEBIBYTE:.1f} MiB. "
            f"HILDA at depth {setting['depth']}, {setting['probes']} probes; "
            f"HNSW at `ef_search` {size['hnsw_ef']}. "
            "Both chosen on validation queries."
        ),
        "",
        HEADER,
        RULE,
    ]
    lines.extend(_row(cell) for cell in sorted(size["cells"], key=_order))
    lines.append("")
    return lines


def render(payload: dict[str, object]) -> str:
    """Render every corpus size in the report."""
    lines = [
        "## Measured",
        "",
        (
            f"Machine memory {payload['machine_memory_bytes'] / MEBIBYTE:.0f} MiB, "
            f"`shared_buffers` {payload['shared_buffers_mb']} MB held fixed, "
            f"recall@{payload['top_k']} against exact cosine, "
            f"target recall {payload['target_recall']}."
        ),
        "",
    ]
    for size in payload["sizes"]:
        lines.extend(render_size(size))
    return "\n".join(lines)


MERGE_KEYS = ("working_set_bytes", "hilda_setting", "hnsw_ef", "table_bytes")
"""What two runs of the same corpus size must agree on to be one experiment."""


def merge(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Combine runs of the same corpus sizes into one report."""
    merged = dict(payloads[0])
    sizes: dict[int, dict[str, object]] = {}
    for payload in payloads:
        for size in payload["sizes"]:
            rows = int(size["rows"])
            if rows not in sizes:
                sizes[rows] = dict(size)
                sizes[rows]["cells"] = list(size["cells"])
                continue
            held = sizes[rows]
            differing = [key for key in MERGE_KEYS if held[key] != size[key]]
            if differing:
                message = (
                    f"runs at {rows} rows disagree on {', '.join(differing)}; "
                    "they are not the same experiment"
                )
                raise ValueError(message)
            held["cells"].extend(size["cells"])
    merged["sizes"] = [sizes[rows] for rows in sorted(sizes)]
    return merged


FRONTIER_HEADER = "| recall | HILDA probes | HILDA p50 | HNSW ef | HNSW p50 | ratio |"
FRONTIER_RULE = "|---|---|---|---|---|---|"


def _pair(points: list[dict[str, object]]) -> list[tuple[dict, dict]]:
    """Match each HILDA setting to the cheapest HNSW point that is not worse.

    Pairing upward rather than to the nearest recall is deliberate: it hands
    HILDA the most favourable comparison available, because the HNSW point it
    is measured against returns at least as many true neighbours.
    """
    hilda = sorted(
        (p for p in points if p["family"] == "hilda"), key=lambda p: p["recall"]
    )
    hnsw = sorted(
        (p for p in points if p["family"] == "hnsw"), key=lambda p: p["recall"]
    )
    pairs = []
    for point in hilda:
        matched = [h for h in hnsw if h["recall"] >= point["recall"]]
        if matched:
            pairs.append((point, min(matched, key=lambda h: h["p50_ms"])))
    return pairs


def render_frontier(payload: dict[str, object]) -> str:
    """Render both frontiers, matched on recall, one table per cap."""
    lines = ["## The frontier, matched on recall", ""]
    caps = sorted({int(p["limit_mb"]) for p in payload["points"]}, reverse=True)
    for cap in caps:
        points = [p for p in payload["points"] if p["limit_mb"] == cap]
        ratio = next(p["achieved_ratio"] for p in points)
        lines.extend([f"### {cap} MiB cap ({ratio:.2f} of the working set)", ""])
        exact = next(p for p in points if p["family"] == "exact")
        lines.append(f"Exact scan: {exact['p50_ms']:.2f} ms at recall 1.000.")
        lines.extend(["", FRONTIER_HEADER, FRONTIER_RULE])
        for left, right in _pair(points):
            lines.append(
                f"| {left['recall']:.3f} vs {right['recall']:.3f} "
                f"| {left['setting']} | {left['p50_ms']:.2f} ms "
                f"| {right['setting']} | {right['p50_ms']:.2f} ms "
                f"| {left['p50_ms'] / right['p50_ms']:.2f}x |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Read the benchmark JSON files and print their tables."""
    parser = argparse.ArgumentParser(description="Summarise the memory-pressure run")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--frontier", type=Path)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    print(render(merge(payloads)))  # noqa: T201
    if args.frontier:
        print(render_frontier(json.loads(args.frontier.read_text())))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
