# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The measurement itself: recall, scan fraction and range count."""

import numpy as np

from hilda_ablation.codes import IndexRange
from hilda_ablation.evaluation import CodeIndex, exact_neighbours

FOUR_MEMBERS = 4
THREE_MEMBERS = 3
TWO_SEEKS = 2


def test_exact_neighbours_ranks_by_cosine() -> None:
    """Exact neighbours ranks by cosine."""
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    queries = np.array([[1.0, 0.02]])
    assert exact_neighbours(corpus, queries, k=2).tolist() == [[0, 2]]


def test_index_reports_members_and_scan_cost() -> None:
    """Index reports members and scan cost."""
    index = CodeIndex(codes=np.array([10, 3, 7, 42, 7]))
    hit = index.scan([IndexRange(3, 7), IndexRange(8, 10)])
    assert sorted(hit.members.tolist()) == [0, 1, 2, 4]
    assert hit.n_scanned == FOUR_MEMBERS
    assert hit.n_ranges == 1  # 3..7 and 8..10 are adjacent, so one seek


def test_disjoint_ranges_are_counted_separately() -> None:
    """Disjoint ranges are counted separately."""
    index = CodeIndex(codes=np.array([1, 2, 3, 100]))
    hit = index.scan([IndexRange(1, 2), IndexRange(100, 100)])
    assert hit.n_ranges == TWO_SEEKS
    assert hit.n_scanned == THREE_MEMBERS


def test_empty_probe_set_scans_nothing() -> None:
    """Empty probe set scans nothing."""
    index = CodeIndex(codes=np.array([1, 2, 3]))
    hit = index.scan([])
    assert hit.n_scanned == 0
    assert hit.n_ranges == 0
    assert hit.members.size == 0
