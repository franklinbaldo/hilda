# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Finding the candidate budget a recall target needs."""

import pytest

from hilda_ablation.scaling import budget_for_recall, scaling_exponent

TARGET = 0.9
NEEDED = 100
AT_TARGET = 90


class _Encoder:
    """An encoder whose recall is a known function of the budget."""

    def __init__(self, needed: int) -> None:
        """Recall reaches 1.0 exactly at ``needed`` candidates."""
        self.needed = needed

    def recall_at(self, budget: int) -> float:
        """Rise linearly to 1.0 at ``needed``, then stay there."""
        return min(1.0, budget / self.needed)


def test_it_finds_the_smallest_budget_that_reaches_the_target() -> None:
    """It finds the exact first budget that reaches the target."""
    encoder = _Encoder(needed=NEEDED)
    found = budget_for_recall(encoder.recall_at, target=TARGET, ceiling=1000)
    assert found == AT_TARGET


def test_an_unreachable_target_reports_no_budget() -> None:
    """An unreachable target reports no budget."""
    encoder = _Encoder(needed=10_000)
    assert budget_for_recall(encoder.recall_at, target=TARGET, ceiling=100) is None


def test_a_target_met_immediately_returns_the_smallest_budget() -> None:
    """A target met immediately returns the smallest budget."""
    assert budget_for_recall(lambda _: 1.0, target=TARGET, ceiling=1000) == 1


def test_the_search_is_monotonic_in_the_target() -> None:
    """A stricter target cannot require fewer candidates."""
    encoder = _Encoder(needed=500)
    loose = budget_for_recall(encoder.recall_at, target=0.5, ceiling=2000)
    strict = budget_for_recall(encoder.recall_at, target=0.95, ceiling=2000)
    assert loose < strict


def test_invalid_target_or_ceiling_is_rejected() -> None:
    """The bisection rejects domains where its contract is undefined."""
    with pytest.raises(ValueError, match="target"):
        budget_for_recall(lambda _: 1.0, target=1.5, ceiling=10)
    with pytest.raises(ValueError, match="ceiling"):
        budget_for_recall(lambda _: 1.0, target=0.9, ceiling=0)


def test_scaling_exponent_recovers_sublinear_power_law() -> None:
    """The reported exponent recovers a known square-root growth law."""
    alpha = scaling_exponent([100, 400, 1600], [10, 20, 40])
    assert alpha == pytest.approx(0.5)


def test_scaling_exponent_rejects_bad_series() -> None:
    """A power-law fit needs paired positive observations."""
    with pytest.raises(ValueError, match="same length"):
        scaling_exponent([100], [10])
    with pytest.raises(ValueError, match="positive"):
        scaling_exponent([100, 200], [10, 0])
