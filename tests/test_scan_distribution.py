# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Per-query scan cost, not just its mean."""

import numpy as np
import pytest

from hilda_ablation.evaluation import ScanDistribution

P50 = 0.2
P95 = 0.8
MAXIMUM = 1.0
MEAN = 0.4


def test_distribution_reports_the_tail_not_only_the_mean() -> None:
    """Distribution reports the tail not only the mean."""
    fractions = np.array([0.2, 0.2, 0.2, 1.0])
    summary = ScanDistribution.of(fractions)
    assert summary.mean == MEAN
    assert summary.p50 == P50
    assert summary.p95 >= P95
    assert summary.maximum == MAXIMUM


def test_a_uniform_scan_has_no_spread() -> None:
    """A uniform scan has no spread."""
    summary = ScanDistribution.of(np.full(10, 0.3))
    assert summary.p50 == pytest.approx(summary.mean)
    assert summary.p95 == pytest.approx(summary.mean)
    assert summary.maximum == pytest.approx(summary.mean)
