# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Turn the memory-pressure JSON into the tables the write-up reports.

One table per corpus size, one row per plan and cache state, columns ordered so
the comparison the experiment exists to make -- HILDA against HNSW at a stated
recall, as the cap tightens -- reads left to right.

Usage:
    uv run scripts/report_memory.py results/memory_pressure.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MEGABYTE = 1e6
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
        f"| {cell['limit_mb']} MB "
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
            f"Working set {size['working_set_bytes'] / MEGABYTE:.0f} MB: "
            f"table {size['table_bytes'] / MEGABYTE:.0f} MB, "
            f"`hnsw_idx` {indexes['hnsw_idx'] / MEGABYTE:.0f} MB, "
            f"`code_idx` {indexes['code_idx'] / MEGABYTE:.1f} MB. "
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
            f"Machine memory {payload['machine_memory_bytes'] / MEGABYTE:.0f} MB, "
            f"`shared_buffers` {payload['shared_buffers_mb']} MB held fixed, "
            f"recall@{payload['top_k']} against exact cosine, "
            f"target recall {payload['target_recall']}."
        ),
        "",
    ]
    for size in payload["sizes"]:
        lines.extend(render_size(size))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Read the benchmark JSON and print its tables."""
    parser = argparse.ArgumentParser(description="Summarise the memory-pressure run")
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    print(render(json.loads(args.input.read_text())))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
