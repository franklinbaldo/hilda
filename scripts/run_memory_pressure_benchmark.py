# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "numpy", "psycopg[binary]"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Ask whether memory pressure produces a crossover between HILDA and HNSW.

Every earlier number in this repository was measured with the whole working set
resident. That regime is settled: a vector index wins the query, and HILDA wins
the index. This asks the one question those runs could not:

    When the working set stops fitting in memory, is there a point where a
    B-tree range scan plus a cosine re-rank offers a better latency/recall
    trade-off than HNSW?

The independent variable is a total memory cap, not ``shared_buffers``: the
server runs inside a cgroup, so page cache read on a backend's behalf counts
against the same budget. Caps are set as ratios of the working set actually
measured after loading, and each is run twice -- once cold, with shared buffers
and page cache both emptied, and once warm.

Operating points are chosen on validation queries under the loosest cap and
then frozen, HNSW's ``ef_search`` no less than HILDA's depth and probe width,
so no plan gets to pick the setting that happens to look best on the queries
the table reports.

Usage:
    uv run scripts/run_memory_pressure_benchmark.py --input data/wikipedia.npy
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psycopg

from hilda_ablation.database import Timings
from hilda_ablation.encoders import fit_hierarchical_kmeans
from hilda_ablation.evaluation import (
    CodeIndex,
    QuerySet,
    Setting,
    exact_neighbours,
    measure,
    split_queries,
)
from hilda_ablation.geometry import unit_norm
from hilda_ablation.memory import (
    Cgroup,
    MemoryLimit,
    PostgresServer,
    ServerSpec,
    available_bytes,
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

logger = logging.getLogger("memory_pressure")

TOP_K = 10
LEVELS, BRANCHING = 4, 16
DEFAULT_SIZES = (100_000, 300_000)
DEFAULT_RATIOS = (2.0, 0.75, 0.35)
EF_LADDER = (10, 20, 40, 80, 160, 320)
PROBE_LADDER = (16, 32, 64, 128, 256, 512, 1024)
SHARED_BUFFERS_MB = 32
"""Small and fixed, so the cgroup cap is what moves between conditions."""

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
MATRIX_DIMENSIONS = 2
MIN_QUERIES = 2


@dataclass(frozen=True)
class Cell:
    """One plan, at one memory cap, in one cache state."""

    rows: int
    plan: str
    limit_mb: int
    target_ratio: float
    achieved_ratio: float
    cache: str
    recall: float
    latency: Timings
    shared_hit: float
    shared_read: float
    read_ms: float

    def as_row(self) -> dict[str, object]:
        """Flatten to a JSON-friendly row."""
        return {
            "rows": self.rows,
            "plan": self.plan,
            "limit_mb": self.limit_mb,
            "target_ratio": self.target_ratio,
            "achieved_ratio": round(self.achieved_ratio, 3),
            "cache": self.cache,
            "recall": round(self.recall, 4),
            **{f"latency_{k}_ms": round(v, 3) for k, v in asdict(self.latency).items()},
            "shared_hit": round(self.shared_hit, 1),
            "shared_read": round(self.shared_read, 1),
            "read_ms": round(self.read_ms, 3),
        }


@dataclass
class SizeReport:
    """Everything measured at one corpus size."""

    rows: int
    working_set_bytes: int
    table_bytes: int
    index_bytes: dict[str, int]
    hilda_setting: dict[str, int]
    hnsw_ef: int | None
    cells: list[Cell] = field(default_factory=list)


def _select_hilda(
    encoder: object, index: CodeIndex, validation: QuerySet, target: float
) -> tuple[int, int, float]:
    """Choose the cheapest depth and probe width reaching the target recall.

    Selection runs against the in-memory harness rather than the server: it is
    the same encoder and the same candidate set, and recall does not depend on
    where the bytes are.
    """
    best: tuple[float, int, int, float] | None = None
    for depth in range(1, LEVELS + 1):
        for probes in PROBE_LADDER:
            point = measure(
                encoder,
                index,
                validation,
                Setting(depth=depth, n_probes=probes),
                split="validation",
            )
            if point.recall < target:
                continue
            candidate = (point.scanned.mean, depth, probes, point.recall)
            if best is None or candidate[0] < best[0]:
                best = candidate
            break
    if best is None:
        message = f"no depth and probe width on the ladder reaches recall {target}"
        raise RuntimeError(message)
    _, depth, probes, recall = best
    logger.info(
        "hilda operating point: depth=%d probes=%d validation recall=%.4f",
        depth,
        probes,
        recall,
    )
    return depth, probes, recall


def _select_ef(
    connection: psycopg.Connection, validation: QuerySet, target: float
) -> int | None:
    """Choose the smallest ``ef_search`` reaching the target recall.

    Chosen on validation queries for the same reason HILDA's depth is: an
    ``ef_search`` picked for how it reads on the reported table is not an
    operating point, it is a result.
    """
    for ef in EF_LADDER:
        recall = measure_recall(connection, hnsw_plan(ef), validation, top_k=TOP_K)
        logger.info("ef_search=%d validation recall=%.4f", ef, recall)
        if recall >= target:
            return ef
    return None


@dataclass(frozen=True)
class Condition:
    """What is held fixed while one plan is measured at one cap."""

    server: PostgresServer
    spec: ServerSpec
    limit: MemoryLimit
    rows: int


def _run_cell(
    condition: Condition,
    plan: Plan,
    queries: QuerySet,
    recall: float,
    *,
    cold: bool,
) -> Cell:
    """Measure one plan once, after establishing the cache state it claims.

    A cold trial restarts the server so ``shared_buffers`` starts empty too; a
    warm trial runs the same queries once and discards them, so the reported
    pass measures a cache the plan itself established.
    """
    if cold:
        condition.server.restart_cold()
    with psycopg.connect(condition.spec.dsn, autocommit=True) as connection:
        if not cold:
            measure_latency(connection, plan, queries)
        sample = measure_latency(connection, plan, queries)
    return Cell(
        rows=condition.rows,
        plan=plan.name,
        limit_mb=condition.limit.limit_bytes // (1024 * 1024),
        target_ratio=condition.limit.target_ratio,
        achieved_ratio=condition.limit.achieved_ratio,
        cache="cold" if cold else "warm",
        recall=recall,
        latency=Timings.of(sample.latencies),
        shared_hit=float(np.mean(sample.hits)),
        shared_read=float(np.mean(sample.reads)),
        read_ms=float(np.mean(sample.read_ms)),
    )


def _run_size(
    matrix: np.ndarray,
    queries: np.ndarray,
    rows: int,
    args: argparse.Namespace,
    cgroup: Cgroup,
) -> SizeReport:
    """Load one corpus, freeze the operating points, then sweep the caps."""
    documents = unit_norm(np.asarray(matrix[:rows], dtype=np.float32))
    truth = exact_neighbours(documents, queries, k=TOP_K)
    validation, test = split_queries(QuerySet(queries=queries, truth=truth))

    logger.info("fitting hkmeans-L%dxK%d on %d rows", LEVELS, BRANCHING, rows)
    encoder = fit_hierarchical_kmeans(
        documents, levels=LEVELS, branching=BRANCHING, seed=args.seed
    )
    codes = encoder.encode(documents)
    depth, probes, _ = _select_hilda(
        encoder, CodeIndex(codes), validation, args.target_recall
    )

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
        _remove_tree(args.datadir)
    server.initdb()

    generous = 4 * 1024 * 1024 * 1024
    cgroup.set_limit(generous)
    server.start()

    with psycopg.connect(spec.dsn, autocommit=True) as connection:
        sizes = load_table(connection, codes, documents)
        working_set = sizes["table"] + sizes["hnsw_idx"] + sizes["code_idx"]
        logger.info("working set %.1f MiB", working_set / (1024 * 1024))
        ef = _select_ef(connection, validation, args.target_recall)
        plans = [exact_plan(), hilda_plan(encoder, depth=depth, probes=probes)]
        if ef is not None:
            plans.append(hnsw_plan(ef))
        recalls = {
            plan.name: measure_recall(connection, plan, test, top_k=TOP_K)
            for plan in plans
        }
    for name, value in recalls.items():
        logger.info("%s test recall %.4f", name, value)

    report = SizeReport(
        rows=rows,
        working_set_bytes=working_set,
        table_bytes=sizes["table"],
        index_bytes={k: v for k, v in sizes.items() if k != "table"},
        hilda_setting={"depth": depth, "probes": probes},
        hnsw_ef=ef,
    )
    for limit in limits_for(working_set, args.ratios):
        logger.info(
            "cap %s (target %.2f, achieved %.2f of working set)",
            limit.label,
            limit.target_ratio,
            limit.achieved_ratio,
        )
        server.stop()
        cgroup.set_limit(limit.limit_bytes)
        server.start()
        condition = Condition(server=server, spec=spec, limit=limit, rows=rows)
        for plan in plans:
            for cold in (True, False):
                cell = _run_cell(condition, plan, test, recalls[plan.name], cold=cold)
                logger.info("%s", cell.as_row())
                report.cells.append(cell)
    server.stop()
    cgroup.set_limit(generous)
    return report


def _remove_tree(path: Path) -> None:
    """Delete a data directory left by an earlier run."""
    import shutil  # noqa: PLC0415 - only needed when a stale directory exists

    shutil.rmtree(path)


def _parse_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of corpus sizes."""
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        message = "sizes must be positive integers"
        raise argparse.ArgumentTypeError(message)
    return sizes


def _parse_ratios(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of working-set ratios."""
    ratios = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not ratios or any(ratio <= 0 for ratio in ratios):
        message = "ratios must be positive"
        raise argparse.ArgumentTypeError(message)
    return ratios


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the benchmark's command line."""
    parser = argparse.ArgumentParser(
        description="HILDA against HNSW under memory pressure"
    )
    parser.add_argument("--input", type=Path, default=Path("data/wikipedia.npy"))
    parser.add_argument("--sizes", type=_parse_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--ratios", type=_parse_ratios, default=DEFAULT_RATIOS)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=5434)
    parser.add_argument("--datadir", type=Path, default=Path("/var/lib/hilda-pgdata"))
    parser.add_argument("--cgroup", default="hilda-memory-pressure")
    parser.add_argument(
        "--out", type=Path, default=Path("results/memory_pressure.json")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep corpus sizes against memory caps and write the comparison."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.queries < MIN_QUERIES:
        message = "--queries must be at least 2 so validation and test are non-empty"
        raise ValueError(message)

    matrix = np.load(args.input, mmap_mode="r")
    max_rows = max(args.sizes)
    required = max_rows + args.queries
    if matrix.ndim != MATRIX_DIMENSIONS or len(matrix) < required:
        message = f"{args.input} needs at least {required} rows; found {len(matrix)}"
        raise ValueError(message)
    queries = unit_norm(
        np.asarray(matrix[max_rows : max_rows + args.queries], dtype=np.float32)
    )

    cgroup = Cgroup(args.cgroup)
    try:
        reports = [
            _run_size(matrix, queries, size, args, cgroup) for size in args.sizes
        ]
    finally:
        cgroup.destroy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine_memory_bytes": available_bytes(),
        "shared_buffers_mb": SHARED_BUFFERS_MB,
        "top_k": TOP_K,
        "target_recall": args.target_recall,
        "sizes": [
            {
                "rows": report.rows,
                "working_set_bytes": report.working_set_bytes,
                "table_bytes": report.table_bytes,
                "index_bytes": report.index_bytes,
                "hilda_setting": report.hilda_setting,
                "hnsw_ef": report.hnsw_ef,
                "cells": [cell.as_row() for cell in report.cells],
            }
            for report in reports
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
