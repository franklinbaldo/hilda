# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "sentence-transformers", "torch", "psycopg[binary]"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Measure the HILDA plan against pgvector on a real Postgres.

The ablation counts ranges over a sorted array in memory. This asks the
planner instead: same codes in a B-tree, same embeddings under HNSW and
IVFFlat, one query set, three plans.

Usage:
    uv run scripts/run_postgres_benchmark.py --dsn "postgresql://...:5433/postgres"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psycopg

from hilda_ablation.corpus import load_corpus
from hilda_ablation.database import TABLE, Timings, range_scan_sql
from hilda_ablation.encoders import fit_hierarchical_kmeans
from hilda_ablation.evaluation import QuerySet, exact_neighbours, split_queries
from hilda_ablation.geometry import unit_norm

logger = logging.getLogger("postgres_benchmark")

TOP_K = 10
LEVELS, BRANCHING = 4, 16
DEPTH, PROBES = 3, 64
"""The hkmeans operating point the ablation selected at a 5% mean scan."""


@dataclass(frozen=True)
class IndexCost:
    """What an index costs to build and to keep.

    The B-tree is the whole point of the HILDA thesis: if it is not far cheaper
    than a vector index, the approach has nothing left to offer.
    """

    index: str
    build_seconds: float
    size_bytes: int

    def as_row(self) -> dict[str, object]:
        """Flatten to a JSON-friendly row."""
        return {
            "index": self.index,
            "build_seconds": round(self.build_seconds, 3),
            "size_mb": round(self.size_bytes / 1e6, 2),
        }


@dataclass(frozen=True)
class PlanResult:
    """What one plan cost and recovered."""

    plan: str
    recall: float
    latency: Timings
    candidates: float
    shared_hit: float
    shared_read: float

    def as_row(self) -> dict[str, object]:
        """Flatten to a JSON-friendly row."""
        return {
            "plan": self.plan,
            "recall": round(self.recall, 4),
            **{f"latency_{k}_ms": round(v, 3) for k, v in asdict(self.latency).items()},
            "candidates": round(self.candidates, 1),
            "shared_hit": round(self.shared_hit, 1),
            "shared_read": round(self.shared_read, 1),
        }


def _vector(row: np.ndarray) -> str:
    """Render one embedding in the literal syntax pgvector parses."""
    return "[" + ",".join(f"{value:.6f}" for value in row) + "]"


def _buffers(
    cursor: psycopg.Cursor, sql: str, params: list[object]
) -> tuple[float, float]:
    """Read shared buffer hits and reads for one statement."""
    cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
    plan = cursor.fetchone()[0][0]["Plan"]
    return float(plan.get("Shared Hit Blocks", 0)), float(
        plan.get("Shared Read Blocks", 0)
    )


def _recall(found: list[int], truth: np.ndarray) -> float:
    """Share of the exact top-k the plan returned."""
    return len(set(found) & set(truth.tolist())) / len(truth)


def load(
    connection: psycopg.Connection, codes: np.ndarray, documents: np.ndarray
) -> None:
    """Rebuild the table and copy the corpus in."""
    with connection.cursor() as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cursor.execute(
            f"CREATE TABLE {TABLE} "
            f"(id int PRIMARY KEY, code bigint, embedding vector({documents.shape[1]}))"
        )
        with cursor.copy(f"COPY {TABLE} (id, code, embedding) FROM STDIN") as copy:
            for index, (code, row) in enumerate(zip(codes, documents, strict=True)):
                copy.write_row((index, int(code), _vector(row)))
        cursor.execute(f"ANALYZE {TABLE}")
    connection.commit()


def measure_hilda(
    connection: psycopg.Connection, encoder: object, queries: QuerySet
) -> PlanResult:
    """Time the range scan plus cosine rerank, one query at a time."""
    latencies: list[float] = []
    recalls: list[float] = []
    candidates: list[int] = []
    hits: list[float] = []
    reads: list[float] = []
    with connection.cursor() as cursor:
        for i, query in enumerate(queries.queries):
            cells = encoder.probe(query, depth=DEPTH, n_probes=PROBES)
            ranges = [encoder.layout.prefix_range(cell) for cell in cells]
            sql, params = range_scan_sql(ranges, vector=_vector(query), limit=TOP_K)
            started = time.perf_counter()
            cursor.execute(sql, params)
            found = [row[0] for row in cursor.fetchall()]
            latencies.append(1000 * (time.perf_counter() - started))
            recalls.append(_recall(found, queries.truth[i]))
            if i % 20 == 0:
                hit, read = _buffers(cursor, sql, params)
                hits.append(hit)
                reads.append(read)
            counted = ", ".join(f"({s.lo}, {s.hi})" for s in ranges)
            cursor.execute(
                f"SELECT count(*) FROM {TABLE} AS d "  # noqa: S608 - integers only
                f"JOIN (VALUES {counted}) AS spans(lo, hi) "
                "ON d.code BETWEEN spans.lo AND spans.hi"
            )
            candidates.append(cursor.fetchone()[0])
    return PlanResult(
        plan="hilda-btree+rerank",
        recall=float(np.mean(recalls)),
        latency=Timings.of(latencies),
        candidates=float(np.mean(candidates)),
        shared_hit=float(np.mean(hits)),
        shared_read=float(np.mean(reads)),
    )


def measure_vector_index(
    connection: psycopg.Connection, queries: QuerySet, plan: str, setting: str
) -> PlanResult:
    """Time pgvector's own plan under the given search setting."""
    latencies: list[float] = []
    recalls: list[float] = []
    hits: list[float] = []
    reads: list[float] = []
    sql = f"SELECT id FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s"  # noqa: S608
    with connection.cursor() as cursor:
        cursor.execute(setting)
        for i, query in enumerate(queries.queries):
            params: list[object] = [_vector(query), TOP_K]
            started = time.perf_counter()
            cursor.execute(sql, params)
            found = [row[0] for row in cursor.fetchall()]
            latencies.append(1000 * (time.perf_counter() - started))
            recalls.append(_recall(found, queries.truth[i]))
            if i % 20 == 0:
                hit, read = _buffers(cursor, sql, params)
                hits.append(hit)
                reads.append(read)
    return PlanResult(
        plan=plan,
        recall=float(np.mean(recalls)),
        latency=Timings.of(latencies),
        candidates=float("nan"),
        shared_hit=float(np.mean(hits)),
        shared_read=float(np.mean(reads)),
    )


def build_index(connection: psycopg.Connection, name: str, sql: str) -> IndexCost:
    """Create one index, timing the build and reading back its size."""
    with connection.cursor() as cursor:
        started = time.perf_counter()
        cursor.execute(sql)
        elapsed = time.perf_counter() - started
        cursor.execute("SELECT pg_relation_size(%s)", [name])
        size = cursor.fetchone()[0]
    return IndexCost(index=name, build_seconds=elapsed, size_bytes=int(size))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the benchmark's command line."""
    parser = argparse.ArgumentParser(description="HILDA against pgvector on Postgres")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--corpus-size", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/postgres.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the three plans and write the comparison."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = load_corpus(
        size=args.corpus_size,
        n_queries=args.queries,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    documents, probes = unit_norm(corpus.documents), unit_norm(corpus.queries)
    truth = exact_neighbours(documents, probes, k=TOP_K)
    _, test = split_queries(QuerySet(queries=probes, truth=truth))

    logger.info("fitting hkmeans-L%dxK%d", LEVELS, BRANCHING)
    encoder = fit_hierarchical_kmeans(
        documents, levels=LEVELS, branching=BRANCHING, seed=args.seed
    )
    codes = encoder.encode(documents)

    results: list[PlanResult] = []
    costs: list[IndexCost] = []
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        logger.info("loading %d rows", len(documents))
        load(connection, codes, documents)
        costs.append(
            build_index(
                connection, "code_idx", f"CREATE INDEX code_idx ON {TABLE} (code)"
            )
        )
        with connection.cursor() as cursor:
            cursor.execute(f"ANALYZE {TABLE}")
        results.append(measure_hilda(connection, encoder, test))

        logger.info("building hnsw")
        costs.append(
            build_index(
                connection,
                "hnsw_idx",
                f"CREATE INDEX hnsw_idx ON {TABLE} "
                "USING hnsw (embedding vector_cosine_ops)",
            )
        )
        results.extend(
            measure_vector_index(
                connection,
                test,
                f"pgvector-hnsw-ef{ef}",
                f"SET hnsw.ef_search = {ef}",
            )
            for ef in (40, 100, 200)
        )
        with connection.cursor() as cursor:
            cursor.execute("DROP INDEX hnsw_idx")
        logger.info("building ivfflat")
        costs.append(
            build_index(
                connection,
                "ivf_idx",
                f"CREATE INDEX ivf_idx ON {TABLE} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
            )
        )
        results.extend(
            measure_vector_index(
                connection,
                test,
                f"pgvector-ivfflat-probes{probe}",
                f"SET ivfflat.probes = {probe}",
            )
            for probe in (1, 10, 30)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "rows": len(documents),
        "plans": [r.as_row() for r in results],
        "indexes": [c.as_row() for c in costs],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["plans"] + report["indexes"]:
        logger.info("%s", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
