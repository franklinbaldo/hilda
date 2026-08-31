# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Postgres side of the operational benchmark.

The ablation counts ranges over a sorted array. This module puts the same
codes in a B-tree and asks the planner what they actually cost, next to
pgvector's own indexes on the same column of embeddings.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hilda_ablation.codes import IndexRange

TABLE = "documents"
"""Single table: the code column and the embedding live side by side."""


@dataclass(frozen=True)
class Timings:
    """Latency of one plan across the query set, in milliseconds."""

    p50: float
    p95: float
    maximum: float
    mean: float

    @classmethod
    def of(cls, samples: list[float]) -> Timings:
        """Summarise per-query latencies."""
        ordered = sorted(samples)
        return cls(
            p50=float(statistics.median(ordered)),
            p95=float(ordered[max(0, round(0.95 * (len(ordered) - 1)))]),
            maximum=float(ordered[-1]),
            mean=float(statistics.fmean(ordered)),
        )


def range_scan_sql(
    ranges: list[IndexRange], vector: str, limit: int
) -> tuple[str, list[object]]:
    """Build the HILDA plan: scan the code ranges, rerank the rows by cosine.

    One range is a plain BETWEEN so the planner sees an ordinary index scan;
    several become a VALUES join, which it can turn into a bitmap of the same
    index. Either way nothing but the B-tree narrows the candidate set.

    Returns the statement and its parameters in order, so the caller never has
    to know where the query vector sits among them.
    """
    if not ranges:
        message = "a range scan needs at least one range"
        raise ValueError(message)
    params: list[object] = []
    for span in ranges:
        params.extend((span.lo, span.hi))
    if len(ranges) == 1:
        where = "WHERE code BETWEEN %s AND %s"
    else:
        rows = ", ".join(["(%s, %s)"] * len(ranges))
        where = (
            f"JOIN (VALUES {rows}) AS spans(lo, hi) "
            "ON d.code BETWEEN spans.lo AND spans.hi"
        )
    params.extend((vector, limit))
    sql = (
        f"SELECT d.id FROM {TABLE} AS d {where} "  # noqa: S608 - no user input
        "ORDER BY d.embedding <=> %s LIMIT %s"
    )
    return sql, params
