# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "numpy", "psycopg[binary]"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Trace latency against held-out recall for both methods, under one memory cap.

The memory-pressure sweep freezes one operating point per method at a
pre-registered target recall. That answers "at 0.95, which plan is cheaper",
and it is the right shape for the question it was built for -- but the two
methods land at different held-out recalls, so it cannot say whose frontier
dominates. A latency ratio between plans that return different answers is not
a trade-off.

This walks several settings on each side instead, on one HNSW graph built once,
and reports latency beside the recall each setting actually reaches. The
question it answers:

    At comparable held-out recall, does the no-crossover conclusion hold?

Every point is reported. Nothing here selects, so test queries are the honest
set to score on: a frontier is the curve, not a choice made from it.

Usage:
    uv run scripts/run_frontier_benchmark.py --rows 299800
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psycopg

from hilda_ablation.database import Timings
from hilda_ablation.encoders import fit_hierarchical_kmeans
from hilda_ablation.evaluation import QuerySet, exact_neighbours, split_queries
from hilda_ablation.geometry import unit_norm
from hilda_ablation.memory import (
    Cgroup,
    MemoryLimit,
    PostgresServer,
    ServerSpec,
    limits_for,
    resolve_binaries,
)
from hilda_ablation.postgres_plans import (
    Plan,
    exact_plan,
    hilda_plan,
    hnsw_plan,
    load_table,
    measure_latency,
    measure_recall,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("frontier")

TOP_K = 10
LEVELS, BRANCHING = 4, 16
DEPTH = 4
"""The depth the sweep selected at this corpus size, held fixed across widths."""

EF_VALUES = (20, 40, 80, 160, 320, 640)
PROBE_VALUES = (128, 256, 512, 1024, 2048)
DEFAULT_RATIOS = (2.0, 0.18)
"""The resident control, and the tightest cap the sweep reached."""

SHARED_BUFFERS_MB = 32
FIXED_SETTINGS = (
    "-c max_parallel_workers_per_gather=0",
    "-c track_io_timing=on",
    "-c jit=off",
    "-c max_connections=8",
)
BINARY_CANDIDATES = (
    Path("/usr/lib/postgresql/16/bin"),
    Path("/usr/lib/postgresql/17/bin"),
    Path("/usr/local/pgsql/bin"),
)
GENEROUS_BYTES = 4 * 1024 * 1024 * 1024
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class Point:
    """One setting of one method: what it returns, and what it costs."""

    plan: str
    family: str
    setting: int
    recall: float
    limit_mb: int
    achieved_ratio: float
    latency: Timings
    shared_read: float
    read_ms: float

    def as_row(self) -> dict[str, object]:
        """Flatten to a JSON-friendly row."""
        return {
            "plan": self.plan,
            "family": self.family,
            "setting": self.setting,
            "recall": round(self.recall, 4),
            "limit_mb": self.limit_mb,
            "achieved_ratio": round(self.achieved_ratio, 3),
            "p50_ms": round(self.latency.p50, 3),
            "p95_ms": round(self.latency.p95, 3),
            "shared_read": round(self.shared_read, 1),
            "read_ms": round(self.read_ms, 3),
        }


def _variants(encoder: object) -> list[tuple[str, int, Plan]]:
    """Build every operating point on both frontiers, plus the exact scan."""
    variants: list[tuple[str, int, Plan]] = [
        ("hilda", probes, hilda_plan(encoder, depth=DEPTH, probes=probes))
        for probes in PROBE_VALUES
    ]
    variants.extend(("hnsw", ef, hnsw_plan(ef)) for ef in EF_VALUES)
    variants.append(("exact", 0, exact_plan()))
    return variants


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the frontier benchmark's command line."""
    parser = argparse.ArgumentParser(description="Latency against held-out recall")
    parser.add_argument("--input", type=Path, default=Path("data/wikipedia.npy"))
    parser.add_argument("--rows", type=int, default=299_800)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=5434)
    parser.add_argument("--datadir", type=Path, default=Path("/var/lib/hilda-pgdata"))
    parser.add_argument("--cgroup", default="hilda-frontier")
    parser.add_argument("--out", type=Path, default=Path("results/frontier.json"))
    return parser.parse_args(argv)


def _corpus(args: argparse.Namespace) -> tuple[np.ndarray, QuerySet]:
    """Load the corpus and the held-out half the frontier is scored on."""
    matrix = np.load(args.input, mmap_mode="r")
    required = args.rows + args.queries
    if matrix.ndim != MATRIX_DIMENSIONS or len(matrix) < required:
        message = f"{args.input} needs at least {required} rows; found {len(matrix)}"
        raise ValueError(message)
    queries = unit_norm(
        np.asarray(matrix[args.rows : args.rows + args.queries], dtype=np.float32)
    )
    documents = unit_norm(np.asarray(matrix[: args.rows], dtype=np.float32))
    truth = exact_neighbours(documents, queries, k=TOP_K)
    _, test = split_queries(QuerySet(queries=queries, truth=truth))
    return documents, test


def _fresh_server(args: argparse.Namespace, cgroup: Cgroup) -> PostgresServer:
    """Start an empty server under a cap generous enough to build in."""
    spec = ServerSpec(
        binaries=resolve_binaries(BINARY_CANDIDATES),
        datadir=args.datadir,
        port=args.port,
        shared_buffers_mb=SHARED_BUFFERS_MB,
        settings=FIXED_SETTINGS,
    )
    server = PostgresServer(spec, cgroup)
    server.stop()
    if args.datadir.exists():
        import shutil  # noqa: PLC0415 - only needed when a stale directory exists

        shutil.rmtree(args.datadir)
    server.initdb()
    cgroup.set_limit(GENEROUS_BYTES)
    server.start()
    return server


def _trace(
    spec: ServerSpec,
    variants: list[tuple[str, int, Plan]],
    recalls: dict[str, float],
    test: QuerySet,
    limit: MemoryLimit,
) -> list[Point]:
    """Time every operating point under one cap, warm.

    Warm is the state reported here because the sweep already established that
    at the tightest cap warm and cold coincide: nothing large enough to matter
    stays resident between queries either way.
    """
    points: list[Point] = []
    with psycopg.connect(spec.dsn, autocommit=True) as connection:
        for family, setting, plan in variants:
            measure_latency(connection, plan, test, top_k=TOP_K)
            sample = measure_latency(connection, plan, test, top_k=TOP_K)
            point = Point(
                plan=plan.name,
                family=family,
                setting=setting,
                recall=recalls[f"{family}-{setting}"],
                limit_mb=limit.limit_bytes // (1024 * 1024),
                achieved_ratio=limit.achieved_ratio,
                latency=Timings.of(sample.latencies),
                shared_read=float(np.mean(sample.reads)),
                read_ms=float(np.mean(sample.read_ms)),
            )
            logger.info("%s", point.as_row())
            points.append(point)
    return points


def main(argv: Sequence[str] | None = None) -> int:
    """Trace both frontiers under each cap and write them out."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    documents, test = _corpus(args)

    logger.info("fitting hkmeans-L%dxK%d on %d rows", LEVELS, BRANCHING, args.rows)
    encoder = fit_hierarchical_kmeans(
        documents, levels=LEVELS, branching=BRANCHING, seed=args.seed
    )
    codes = encoder.encode(documents)
    cgroup = Cgroup(args.cgroup)
    server = _fresh_server(args, cgroup)
    spec = server.spec
    variants = _variants(encoder)
    points: list[Point] = []
    try:
        with psycopg.connect(spec.dsn, autocommit=True) as connection:
            sizes = load_table(connection, codes, documents)
            working_set = sizes["table"] + sizes["hnsw_idx"] + sizes["code_idx"]
            logger.info("working set %.1f MiB", working_set / (1024 * 1024))
            recalls = {
                f"{family}-{setting}": measure_recall(
                    connection, plan, test, top_k=TOP_K
                )
                for family, setting, plan in variants
            }
        for name, value in recalls.items():
            logger.info("%s recall %.4f", name, value)

        for limit in limits_for(working_set, args.ratios):
            logger.info("cap %s (achieved %.2f)", limit.label, limit.achieved_ratio)
            server.stop()
            cgroup.set_limit(limit.limit_bytes)
            server.start()
            points.extend(_trace(spec, variants, recalls, test, limit))
    finally:
        server.stop()
        cgroup.set_limit(GENEROUS_BYTES)
        cgroup.destroy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "rows": args.rows,
                "working_set_bytes": working_set,
                "depth": DEPTH,
                "top_k": TOP_K,
                "points": [point.as_row() for point in points],
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
