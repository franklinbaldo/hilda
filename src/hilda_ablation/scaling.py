# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Measure how the candidate budget needed for fixed recall grows with corpus size."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MIN_BUDGET = 1
"""No plan can retrieve fewer than one candidate."""


def budget_for_recall(
    recall_at: Callable[[int], float], target: float, ceiling: int
) -> int | None:
    """Return the smallest candidate budget reaching ``target``, or ``None``.

    The search assumes recall is monotonic in candidate budget. This is true
    for the nested candidate sets produced by the benchmark's budgeted range
    scan. ``None`` means the target is not reachable below ``ceiling``.
    """
    if not 0.0 < target <= 1.0:
        message = f"target must fall in (0, 1], got {target}"
        raise ValueError(message)
    if ceiling < MIN_BUDGET:
        message = f"ceiling must be at least {MIN_BUDGET}, got {ceiling}"
        raise ValueError(message)
    if recall_at(ceiling) < target:
        return None
    low, high = MIN_BUDGET, ceiling
    while low < high:
        middle = (low + high) // 2
        if recall_at(middle) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def scaling_exponent(sizes: Sequence[int], budgets: Sequence[int]) -> float:
    """Fit ``budget ≈ N**alpha`` and return the log-log exponent ``alpha``."""
    if len(sizes) != len(budgets) or len(sizes) < 2:
        message = "sizes and budgets must have the same length and at least two points"
        raise ValueError(message)
    size_array = np.asarray(sizes, dtype=float)
    budget_array = np.asarray(budgets, dtype=float)
    if np.any(size_array <= 0) or np.any(budget_array <= 0):
        message = "sizes and budgets must be positive"
        raise ValueError(message)
    alpha, _ = np.polyfit(np.log(size_array), np.log(budget_array), deg=1)
    return float(alpha)
