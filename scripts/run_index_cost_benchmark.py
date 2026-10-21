# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "numpy", "psycopg[binary]"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Measure what it costs to *keep* HILDA's scalar index versus HNSW.

This benchmark deliberately does not ask which query is faster. PR #3/#4 own
that question. Here the independent variable is the index strategy maintained
on the same embeddings:

* no secondary index -- baseline append cost;
* HILDA's scalar ``code bigint`` B-tree;
* pgvector HNSW over the embedding.

For each strategy it records physical index bytes, build time, build WAL,
append throughput, and append WAL. New semantic codes are computed before the
timed database append and their encoding time is reported separately, because
HNSW also receives already-computed embeddings: mixing model-side work into one
Postgres insert path would make the index-maintenance comparison ambiguous.

Usage:
    uv run scripts/run_index_cost_benchmark.py \
      --dsn postgresql:///postgres --input data/wikipedia.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psycopg

from hilda_ablation.encoders import fit_hierarchical_kmeans
from hilda_ablation.geometry import unit_norm
from hilda_ablation.index_cost import IndexCost, relative_cost

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("index_cost")

LEVELS, BRANCHING = 4, 16
DEFAULT_ROWS = 250_000
DEFAULT_INSERT_ROWS = 25_000
VECTOR_PRECISION = 6


def _vector(row: np.ndarray) -> str:
    """Render a pgvector literal without changing the measured data."""
    return "[" + ",".join(f"{value:.{VECTOR_PRECISION}f}" for value in row) + "]"


def _wal_lsn(cursor: psycopg.Cursor) -> str:
    cursor.execute("SELECT pg_current_wal_insert_lsn()")
    return str(cursor.fetchone()[0])


def _wal_bytes(cursor: psycopg.Cursor, before: str, after: str) -> int:
    cursor.execute("SELECT pg_wal_lsn_diff(%s, %s)", [after, before])
    return int(cursor.fetchone()[0])


def _create_table(connection: psycopg.Connection, table: str, dimension: int) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608
        cursor.execute(
            f"CREATE TABLE {table} "  # noqa: S608
            f"(id bigint PRIMARY KEY, code bigint, embedding vector({dimension}))"
        )


def _copy_rows(
    connection: psycopg.Connection,
    table: str,
    ids: range,
    codes: np.ndarray,
    documents: np.ndarray,
) -> float:
    started = time.perf_counter()
    with connection.cursor() as cursor:
        with cursor.copy(
            f"COPY {table} (id, code, embedding) FROM STDIN"  # noqa: S608
        ) as copy:
            for row_id, code, embedding in zip(ids, codes, documents, strict=True):
                copy.write_row((row_id, int(code), _vector(embedding)))
    return time.perf_counter() - started


def _measure_strategy(
    connection: psycopg.Connection,
    strategy: str,
    base_codes: np.ndarray,
    base_documents: np.ndarray,
    insert_codes: np.ndarray,
    insert_documents: np.ndarray,
) -> IndexCost:
    table = f"hilda_cost_{strategy}"
    _create_table(connection, table, base_documents.shape[1])
    _copy_rows(
        connection,
        table,
        range(len(base_documents)),
        base_codes,
        base_documents,
    )

    with connection.cursor() as cursor:
        if strategy == "hnsw":
            cursor.execute("SET maintenance_work_mem = '2GB'")
            index_sql = (
                f"CREATE INDEX {table}_idx ON {table} "  # noqa: S608
                "USING hnsw (embedding vector_cosine_ops)"
            )
        elif strategy == "btree":
            index_sql = f"CREATE INDEX {table}_idx ON {table} (code)"  # noqa: S608
        elif strategy == "baseline":
            index_sql = None
        else:
            message = f"unknown strategy: {strategy}"
            raise ValueError(message)

        if index_sql is None:
            build_seconds = 0.0
            build_wal = 0
            size_bytes = 0
        else:
            before = _wal_lsn(cursor)
            started = time.perf_counter()
            cursor.execute(index_sql)
            build_seconds = time.perf_counter() - started
            after = _wal_lsn(cursor)
            build_wal = _wal_bytes(cursor, before, after)
            cursor.execute("SELECT pg_relation_size(%s)", [f"{table}_idx"])
            size_bytes = int(cursor.fetchone()[0])

        before_insert = _wal_lsn(cursor)

    insert_seconds = _copy_rows(
        connection,
        table,
        range(len(base_documents), len(base_documents) + len(insert_documents)),
        insert_codes,
        insert_documents,
    )

    with connection.cursor() as cursor:
        after_insert = _wal_lsn(cursor)
        insert_wal = _wal_bytes(cursor, before_insert, after_insert)

    return IndexCost(
        strategy=strategy,
        rows=len(base_documents),
        size_bytes=size_bytes,
        build_seconds=build_seconds,
        build_wal_bytes=build_wal,
        insert_rows=len(insert_documents),
        insert_seconds=insert_seconds,
        insert_wal_bytes=insert_wal,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-tree versus HNSW ownership cost")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/wikipedia.npy"))
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--insert-rows", type=int, default=DEFAULT_INSERT_ROWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/index_cost.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.rows <= 0 or args.insert_rows <= 0:
        message = "--rows and --insert-rows must be positive"
        raise ValueError(message)

    matrix = np.load(args.input, mmap_mode="r")
    required = args.rows + args.insert_rows
    if matrix.ndim != 2 or len(matrix) < required:
        message = f"{args.input} needs at least {required} rows; found {len(matrix)}"
        raise ValueError(message)

    base = unit_norm(np.asarray(matrix[: args.rows], dtype=np.float32))
    inserts = unit_norm(
        np.asarray(matrix[args.rows : required], dtype=np.float32)
    )

    logger.info("fitting hkmeans-L%dxK%d on %d rows", LEVELS, BRANCHING, args.rows)
    encoder = fit_hierarchical_kmeans(
        base, levels=LEVELS, branching=BRANCHING, seed=args.seed
    )
    base_codes = encoder.encode(base)
    encode_started = time.perf_counter()
    insert_codes = encoder.encode(inserts)
    insert_encode_seconds = time.perf_counter() - encode_started

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        costs = [
            _measure_strategy(
                connection,
                strategy,
                base_codes,
                base,
                insert_codes,
                inserts,
            )
            for strategy in ("baseline", "btree", "hnsw")
        ]

    by_name = {cost.strategy: cost for cost in costs}
    payload = {
        "rows": args.rows,
        "insert_rows": args.insert_rows,
        "embedding_dimension": int(base.shape[1]),
        "semantic_code_insert_encode_seconds": round(insert_encode_seconds, 3),
        "semantic_code_insert_rows_per_second": round(
            args.insert_rows / insert_encode_seconds, 3
        ),
        "strategies": [cost.as_row() for cost in costs],
        "hnsw_over_btree": relative_cost(by_name["hnsw"], by_name["btree"]),
        "append_over_baseline": {
            name: {
                "time_ratio": cost.insert_seconds
                / by_name["baseline"].insert_seconds,
                "wal_ratio": cost.insert_wal_bytes
                / by_name["baseline"].insert_wal_bytes,
            }
            for name, cost in by_name.items()
            if name != "baseline"
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", args.out)
    for row in payload["strategies"]:
        logger.info("%s", row)
    logger.info("HNSW / B-tree: %s", payload["hnsw_over_btree"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
