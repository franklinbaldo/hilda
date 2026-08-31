# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "numpy"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Measure how HILDA's candidate budget grows with corpus size.

The input is one NPY matrix built by ``build_wikipedia_corpus.py``. Corpus
rungs are nested prefixes. Held-out queries are the final rows after the
largest rung, so no query belongs to any measured corpus.

For each rung the encoder is refit with the same shape. Validation queries pick
the depth and the smallest per-query candidate budget that reaches the target
recall; the held-out half then reports recall at that frozen operating point.
The final ``alpha`` fits ``budget ~= N**alpha``. Alpha below one is the evidence
needed for the claim that the range-scan advantage widens with scale.

Usage:
    uv run scripts/run_scaling_benchmark.py --input data/wikipedia.npy
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
from hilda_ablation.scaling import budget_for_recall, scaling_exponent

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("scaling_benchmark")

TOP_K = 10
LEVELS, BRANCHING = 4, 16
DEFAULT_RUNGS = (10_000, 30_000, 100_000, 300_000)
MIN_RUNGS = 2
"""A log-log slope needs two rungs before it means anything."""

MIN_QUERIES = 2
"""Halving the query set needs at least one query on each side."""

MATRIX_DIMENSIONS = 2
"""The cache is one row per document, one column per embedding dimension."""


@dataclass(frozen=True)
class ScaleRow:
    """One corpus-size observation after validation chooses the operating point."""

    rows: int
    depth: int | None
    budget: int | None
    budget_fraction: float | None
    validation_recall: float | None
    test_recall: float | None
    test_ranges: float | None


def _parse_rungs(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, strictly increasing list of corpus sizes."""
    try:
        rungs = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        message = "rungs must be comma-separated integers"
        raise argparse.ArgumentTypeError(message) from exc
    if len(rungs) < MIN_RUNGS or any(rung <= 0 for rung in rungs):
        message = "rungs must contain at least two positive sizes"
        raise argparse.ArgumentTypeError(message)
    if any(left >= right for left, right in itertools.pairwise(rungs)):
        message = "rungs must be strictly increasing"
        raise argparse.ArgumentTypeError(message)
    return rungs


def _recall_probe(
    encoder: object, index: CodeIndex, validation: QuerySet, depth: int
) -> Callable[[int], float]:
    """Return a memoised recall-at-budget function bound to one depth.

    The depth is bound as a parameter rather than captured from a loop, so the
    closure cannot drift if its call is ever deferred.
    """
    cache: dict[int, float] = {}

    def recall_at(budget: int) -> float:
        if budget not in cache:
            point = measure(
                encoder,
                index,
                validation,
                Setting(depth=depth, budget=budget),
                split="validation",
            )
            cache[budget] = point.recall
        return cache[budget]

    return recall_at


def _best_setting(
    encoder: object,
    index: CodeIndex,
    validation: QuerySet,
    target: float,
    ceiling: int,
) -> tuple[int, int, float] | None:
    """Choose the depth needing the fewest candidates to reach ``target``."""
    choices: list[tuple[int, int, float]] = []
    for depth in range(1, LEVELS + 1):
        recall_at = _recall_probe(encoder, index, validation, depth)
        budget = budget_for_recall(recall_at, target=target, ceiling=ceiling)
        if budget is not None:
            choices.append((budget, depth, recall_at(budget)))
    if not choices:
        return None
    budget, depth, recall = min(choices)
    return depth, budget, recall


def _run_rung(
    matrix: np.ndarray,
    queries: np.ndarray,
    rows: int,
    target: float,
    seed: int,
) -> ScaleRow:
    """Fit and evaluate one nested corpus prefix."""
    documents = unit_norm(np.asarray(matrix[:rows], dtype=np.float32))
    truth = exact_neighbours(documents, queries, k=TOP_K)
    validation, test = split_queries(QuerySet(queries=queries, truth=truth))

    logger.info("fitting hkmeans-L%dxK%d on %,d rows", LEVELS, BRANCHING, rows)
    encoder = fit_hierarchical_kmeans(
        documents, levels=LEVELS, branching=BRANCHING, seed=seed
    )
    index = CodeIndex(encoder.encode(documents))
    chosen = _best_setting(encoder, index, validation, target=target, ceiling=rows)
    if chosen is None:
        logger.info("%,d rows: target %.3f unreachable", rows, target)
        return ScaleRow(rows, None, None, None, None, None, None)

    depth, budget, validation_recall = chosen
    test_point = measure(
        encoder,
        index,
        test,
        Setting(depth=depth, budget=budget),
        split="test",
    )
    logger.info(
        "%,d rows: depth=%d budget=%d (%.3f%%) validation=%.4f test=%.4f",
        rows,
        depth,
        budget,
        100 * budget / rows,
        validation_recall,
        test_point.recall,
    )
    return ScaleRow(
        rows=rows,
        depth=depth,
        budget=budget,
        budget_fraction=budget / rows,
        validation_recall=validation_recall,
        test_recall=test_point.recall,
        test_ranges=test_point.n_ranges,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the scale ladder command line."""
    parser = argparse.ArgumentParser(
        description="Measure HILDA candidate-budget scaling"
    )
    parser.add_argument("--input", type=Path, default=Path("data/wikipedia.npy"))
    parser.add_argument(
        "--rungs",
        type=_parse_rungs,
        default=DEFAULT_RUNGS,
        help="comma-separated nested corpus sizes",
    )
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/scaling.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the nested scale ladder and write its power-law exponent."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.queries < MIN_QUERIES:
        message = "--queries must be at least 2 so validation and test are non-empty"
        raise ValueError(message)
    if not 0.0 < args.target_recall <= 1.0:
        message = "--target-recall must fall in (0, 1]"
        raise ValueError(message)

    matrix = np.load(args.input, mmap_mode="r")
    max_rows = max(args.rungs)
    required = max_rows + args.queries
    if matrix.ndim != MATRIX_DIMENSIONS or len(matrix) < required:
        message = f"{args.input} needs at least {required} rows; found {len(matrix)}"
        raise ValueError(message)
    queries = unit_norm(
        np.asarray(matrix[max_rows : max_rows + args.queries], dtype=np.float32)
    )

    rows = [
        _run_rung(matrix, queries, rung, args.target_recall, args.seed)
        for rung in args.rungs
    ]
    successful = [row for row in rows if row.budget is not None]
    alpha = None
    if len(successful) >= MIN_RUNGS:
        alpha = scaling_exponent(
            [row.rows for row in successful],
            [row.budget for row in successful if row.budget is not None],
        )

    report = {
        "encoder": f"hkmeans-L{LEVELS}xK{BRANCHING}",
        "target_recall": args.target_recall,
        "queries": args.queries,
        "alpha": alpha,
        "interpretation": (
            None if alpha is None else "sublinear" if alpha < 1.0 else "linear-or-worse"
        ),
        "rungs": [asdict(row) for row in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("alpha=%s", "unavailable" if alpha is None else f"{alpha:.4f}")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
