# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The SQL a range-scan plan sends, and how its timings are summarised."""

import pytest

from hilda_ablation.codes import IndexRange
from hilda_ablation.database import Timings, range_scan_sql

VECTOR = "[0.1,0.2]"
TWO_RANGES = 2
MEDIAN = 2.5
P95 = 10.0


def test_a_single_range_becomes_one_between() -> None:
    """A single range becomes one between."""
    sql, params = range_scan_sql([IndexRange(4, 9)], vector=VECTOR, limit=10)
    assert "code BETWEEN" in sql
    assert params[:2] == [4, 9]


def test_several_ranges_become_a_values_join() -> None:
    """Several ranges become a values join."""
    sql, params = range_scan_sql(
        [IndexRange(0, 1), IndexRange(8, 9)], vector=VECTOR, limit=10
    )
    assert sql.count("(%s, %s)") == TWO_RANGES
    assert params[:4] == [0, 1, 8, 9]


def test_the_plan_reranks_by_cosine_and_limits() -> None:
    """The plan reranks by cosine and limits."""
    sql, params = range_scan_sql([IndexRange(0, 1)], vector=VECTOR, limit=7)
    assert "<=>" in sql
    assert "LIMIT" in sql
    assert params[-2:] == [VECTOR, 7]


def test_every_placeholder_is_positional() -> None:
    """Every placeholder is positional.

    psycopg refuses a statement that mixes positional and named placeholders,
    and the query vector sits between the range bounds and the limit.
    """
    sql, params = range_scan_sql(
        [IndexRange(0, 1), IndexRange(4, 5)], vector=VECTOR, limit=10
    )
    assert "%(" not in sql
    assert sql.count("%s") == len(params)


def test_the_rerank_sits_behind_an_optimisation_fence() -> None:
    """The re-rank sits behind an optimisation fence.

    Without ``OFFSET 0`` a table carrying a vector index lets the planner
    answer the sort from that index and demote the codes to a filter, which
    measures the vector index under a different name.
    """
    sql, _ = range_scan_sql([IndexRange(0, 1)], vector=VECTOR, limit=10)
    assert "OFFSET 0" in sql
    assert sql.index("OFFSET 0") < sql.index("<=>")


def test_no_range_is_not_a_query() -> None:
    """No range is not a query."""
    with pytest.raises(ValueError, match="at least one range"):
        range_scan_sql([], vector=VECTOR, limit=10)


def test_timings_report_the_tail() -> None:
    """Timings report the tail."""
    summary = Timings.of([1.0, 2.0, 3.0, 10.0])
    assert summary.p50 == MEDIAN
    assert summary.p95 <= P95
    assert summary.maximum == P95
