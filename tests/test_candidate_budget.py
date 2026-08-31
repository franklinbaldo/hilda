# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""A budget imposed per query, not met on average."""

import numpy as np

from hilda_ablation.codes import CodeLayout, IndexRange
from hilda_ablation.evaluation import CodeIndex, take_budget

BUDGET = 5
CORPUS = 16


class _FakeEncoder:
    """An encoder whose cells are known, so budget truncation is checkable."""

    name = "fake"
    layout = CodeLayout(digit_bits=(2,))

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) integer codes."""
        return points[:, 0].astype(np.int64)

    def probe(
        self, query: np.ndarray, depth: int, n_probes: int
    ) -> list[tuple[int, ...]]:
        """Return the cells nearest the query, nearest first."""
        del query, depth
        return [(cell,) for cell in range(4)][:n_probes]


def test_budget_truncates_to_the_nearest_candidates() -> None:
    """Budget truncates to the nearest candidates."""
    codes = np.repeat(np.arange(4), 4)
    index = CodeIndex(codes=codes)
    plan = take_budget(_FakeEncoder(), index, np.zeros(2), depth=1, budget=BUDGET)
    assert plan.n_scanned == BUDGET
    assert set(plan.members.tolist()) <= set(range(8))


def test_a_budget_larger_than_the_corpus_scans_everything_reached() -> None:
    """A budget larger than the corpus scans everything reached."""
    index = CodeIndex(codes=np.repeat(np.arange(4), 4))
    plan = take_budget(_FakeEncoder(), index, np.zeros(2), depth=1, budget=1000)
    assert plan.n_scanned == CORPUS


def test_ranges_count_only_the_cells_the_budget_paid_for() -> None:
    """Ranges count only the cells the budget paid for."""
    index = CodeIndex(codes=np.repeat(np.arange(4), 4))
    plan = take_budget(_FakeEncoder(), index, np.zeros(2), depth=1, budget=4)
    assert plan.n_ranges == 1


def test_the_index_still_reports_merged_ranges() -> None:
    """The index still reports merged ranges."""
    index = CodeIndex(codes=np.array([0, 1, 2, 3]))
    assert index.scan([IndexRange(0, 1), IndexRange(2, 3)]).n_ranges == 1
