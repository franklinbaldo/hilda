# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""How the candidate budget a recall target needs grows with the corpus.

The Postgres benchmark showed the range scan beating an exact scan at one
corpus size. Whether that advantage widens or closes with scale depends on one
curve: the candidates needed to hold a fixed recall, as a function of N. If it
grows sublinearly the advantage widens; if it tracks N, the range scan is just
a sequential scan with extra steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

MIN_BUDGET = 1
"""No plan can retrieve fewer than one candidate."""


def budget_for_recall(
    recall_at: Callable[[int], float], target: float, ceiling: int
) -> int | None:
    """Smallest candidate budget whose recall reaches `target`, or None.

    Recall rises with the budget, so this bisects rather than sweeping. None
    means the target is out of reach below `ceiling`, which is itself a result:
    the corpus has outgrown what the encoder can deliver at that depth.
    """
    if not 0.0 < target <= 1.0:
        message = f"target must fall in (0, 1], got {target}"
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
