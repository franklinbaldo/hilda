# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Depth is chosen on validation queries and reported on test queries."""

import numpy as np
import pytest

from hilda_ablation.evaluation import QuerySet, split_queries

HALF = 3
TOTAL = 6


@pytest.fixture
def queries() -> np.ndarray:
    """Six distinguishable query vectors."""
    return np.arange(TOTAL * 2, dtype=np.float64).reshape(TOTAL, 2)


def test_split_halves_the_held_out_queries(queries: np.ndarray) -> None:
    """Split halves the held out queries."""
    truth = np.zeros((TOTAL, 4), dtype=np.int64)
    validation, test = split_queries(QuerySet(queries=queries, truth=truth))
    assert len(validation.queries) == HALF
    assert len(test.queries) == HALF


def test_the_two_halves_share_no_query(queries: np.ndarray) -> None:
    """The two halves share no query."""
    truth = np.zeros((TOTAL, 4), dtype=np.int64)
    validation, test = split_queries(QuerySet(queries=queries, truth=truth))
    seen = {tuple(row) for row in validation.queries}
    assert not any(tuple(row) in seen for row in test.queries)


def test_each_half_keeps_its_own_ground_truth(queries: np.ndarray) -> None:
    """Each half keeps its own ground truth."""
    truth = np.arange(TOTAL * 4, dtype=np.int64).reshape(TOTAL, 4)
    validation, test = split_queries(QuerySet(queries=queries, truth=truth))
    assert np.array_equal(validation.truth, truth[:HALF])
    assert np.array_equal(test.truth, truth[HALF:])


def test_an_odd_count_keeps_every_query(queries: np.ndarray) -> None:
    """An odd count keeps every query."""
    odd = queries[:5]
    truth = np.zeros((5, 4), dtype=np.int64)
    validation, test = split_queries(QuerySet(queries=odd, truth=truth))
    assert len(validation.queries) + len(test.queries) == len(odd)
