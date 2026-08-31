# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The three query plans, and one way to measure any of them.

A plan is whatever turns a query vector into a statement, plus the session
settings that force the planner to actually use it. Holding them behind one
interface is what makes the comparison fair: the same loop, the same
measurement, the same query set, and only the plan changing.

Latency here is the server's own execution time, read out of ``EXPLAIN
(ANALYZE, BUFFERS)`` rather than timed around the round trip. Under memory
pressure the quantity of interest is what the server spends, and a single
statement then yields latency, buffer counts and I/O time together -- so no
query is executed twice, which a cold-cache trial could not survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from hilda_ablation.codes import merge_ranges
from hilda_ablation.database import TABLE, range_scan_sql

if TYPE_CHECKING:
    from collections.abc import Callable

    import psycopg

    from hilda_ablation.evaluation import QuerySet

VECTOR_PRECISION = 6
IO_READ_KEYS = ("I/O Read Time", "Shared I/O Read Time")
"""PostgreSQL 17 renamed this field; both spellings mean the same thing."""


class Encoder(Protocol):
    """The part of an encoder a range-scan plan needs."""

    layout: object

    def probe(
        self, query: np.ndarray, depth: int, n_probes: int
    ) -> list[tuple[int, ...]]:
        """Return the cells nearest the query, nearest first."""


@dataclass(frozen=True)
class Plan:
    """One way of answering a top-k query, and how to make the planner use it."""

    name: str
    build: Callable[[np.ndarray, int], tuple[str, list[object]]]
    settings: tuple[str, ...] = ()


@dataclass
class Sample:
    """Per-query measurements of one plan over one query set."""

    latencies: list[float] = field(default_factory=list)
    hits: list[float] = field(default_factory=list)
    reads: list[float] = field(default_factory=list)
    read_ms: list[float] = field(default_factory=list)


def render_vector(row: np.ndarray) -> str:
    """Render one embedding in the literal syntax pgvector parses."""
    return "[" + ",".join(f"{value:.{VECTOR_PRECISION}f}" for value in row) + "]"


def exact_plan() -> Plan:
    """Build the sequential scan a deployment without a vector index runs.

    Index scans are disabled rather than sequential scans enabled: with an HNSW
    index on the table the planner would otherwise answer this query from it,
    and the plan would stop being the exact one.
    """

    def build(query: np.ndarray, limit: int) -> tuple[str, list[object]]:
        sql = f"SELECT id FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s"  # noqa: S608
        return sql, [render_vector(query), limit]

    return Plan(
        name="postgres-seqscan-exact",
        build=build,
        settings=("SET enable_indexscan = off", "SET enable_bitmapscan = off"),
    )


def hnsw_plan(ef_search: int) -> Plan:
    """Build pgvector's graph-index plan at one search width."""

    def build(query: np.ndarray, limit: int) -> tuple[str, list[object]]:
        sql = f"SELECT id FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s"  # noqa: S608
        return sql, [render_vector(query), limit]

    return Plan(
        name=f"pgvector-hnsw-ef{ef_search}",
        build=build,
        settings=(
            "SET enable_indexscan = on",
            f"SET hnsw.ef_search = {ef_search}",
        ),
    )


def hilda_plan(encoder: Encoder, depth: int, probes: int) -> Plan:
    """Build the B-tree range scan over semantic codes, re-ranked by cosine.

    Ranges are merged before they reach the statement, so the plan scans the
    same intervals the ablation counted rather than a longer list holding the
    same rows.
    """

    def build(query: np.ndarray, limit: int) -> tuple[str, list[object]]:
        cells = encoder.probe(query, depth=depth, n_probes=probes)
        ranges = merge_ranges([encoder.layout.prefix_range(cell) for cell in cells])
        return range_scan_sql(ranges, vector=render_vector(query), limit=limit)

    return Plan(
        name="hilda-btree+rerank",
        build=build,
        settings=("SET enable_indexscan = on",),
    )


def _apply(cursor: psycopg.Cursor, plan: Plan) -> None:
    """Apply the session settings the plan needs."""
    for setting in plan.settings:
        cursor.execute(setting)


def _io_read_ms(node: dict[str, object]) -> float:
    """Read the node's I/O read time under whichever key this server uses."""
    for key in IO_READ_KEYS:
        if key in node:
            return float(node[key])  # type: ignore[arg-type]
    return 0.0


def measure_latency(
    connection: psycopg.Connection, plan: Plan, queries: QuerySet, top_k: int = 10
) -> Sample:
    """Execute every query once, recording time, buffers and I/O together."""
    sample = Sample()
    with connection.cursor() as cursor:
        _apply(cursor, plan)
        for query in queries.queries:
            sql, params = plan.build(query, top_k)
            cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
            result = cursor.fetchone()[0][0]
            node = result["Plan"]
            sample.latencies.append(float(result["Execution Time"]))
            sample.hits.append(float(node.get("Shared Hit Blocks", 0)))
            sample.reads.append(float(node.get("Shared Read Blocks", 0)))
            sample.read_ms.append(_io_read_ms(node))
    return sample


def measure_recall(
    connection: psycopg.Connection, plan: Plan, queries: QuerySet, top_k: int = 10
) -> float:
    """Share of the exact top-k the plan returns, averaged over the query set.

    Recall is a property of the plan and its operating point, not of where the
    bytes happen to live, so it is measured once and carried across the memory
    caps rather than remeasured under each.
    """
    scores: list[float] = []
    with connection.cursor() as cursor:
        _apply(cursor, plan)
        for i, query in enumerate(queries.queries):
            sql, params = plan.build(query, top_k)
            cursor.execute(sql, params)
            found = {row[0] for row in cursor.fetchall()}
            truth = set(queries.truth[i].tolist())
            scores.append(len(found & truth) / len(truth))
    return float(np.mean(scores))


def load_table(
    connection: psycopg.Connection, codes: np.ndarray, documents: np.ndarray
) -> dict[str, int]:
    """Rebuild the table, index it both ways, and report what each costs.

    ``maintenance_work_mem`` is raised for the build only. It is a build-time
    allowance, spent before any cap is applied and released before any query is
    timed, so it does not enter the memory regime under test -- and leaving it
    at the default would measure pgvector's out-of-memory build path instead of
    its index.
    """
    sizes: dict[str, int] = {}
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cursor.execute(
            f"CREATE TABLE {TABLE} "
            f"(id int PRIMARY KEY, code bigint, embedding vector({documents.shape[1]}))"
        )
        with cursor.copy(f"COPY {TABLE} (id, code, embedding) FROM STDIN") as copy:
            for index, (code, row) in enumerate(zip(codes, documents, strict=True)):
                copy.write_row((index, int(code), render_vector(row)))
        cursor.execute("SET maintenance_work_mem = '2GB'")
        cursor.execute(f"CREATE INDEX code_idx ON {TABLE} (code)")
        cursor.execute(
            f"CREATE INDEX hnsw_idx ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
        )
        cursor.execute(f"ANALYZE {TABLE}")
        for name in (TABLE, "code_idx", "hnsw_idx"):
            cursor.execute("SELECT pg_relation_size(%s)", [name])
            sizes["table" if name == TABLE else name] = int(cursor.fetchone()[0])
    return sizes
