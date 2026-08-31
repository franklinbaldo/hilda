# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Contract every encoder family must honour, whatever its internals."""

import numpy as np
import pytest

from hilda_ablation.encoders import (
    Encoder,
    fit_hierarchical_kmeans,
    fit_residual_kmeans,
    fit_sfc,
)
from hilda_ablation.projections import fit_pca, fit_random_projection

PROBE_BUDGET = 5


@pytest.fixture
def points() -> np.ndarray:
    """Clustered points, so cell structure is something to get right."""
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(6, 12)) * 4
    return np.repeat(centres, 60, axis=0) + rng.normal(size=(360, 12))


def _encoders(points: np.ndarray) -> list[Encoder]:
    """One fitted encoder per family under test."""
    return [
        fit_sfc(points, projection=fit_pca(points, dims=2), bits=5, curve="hilbert"),
        fit_sfc(points, projection=fit_pca(points, dims=3), bits=4, curve="morton"),
        fit_sfc(
            points,
            projection=fit_random_projection(points, dims=2, seed=1),
            bits=5,
            curve="hilbert",
        ),
        fit_hierarchical_kmeans(points, levels=3, branching=4, seed=0),
        fit_residual_kmeans(points, levels=3, branching=4, seed=0),
    ]


def test_codes_fit_the_declared_bit_budget(points: np.ndarray) -> None:
    """Codes fit the declared bit budget."""
    for encoder in _encoders(points):
        codes = encoder.encode(points)
        assert codes.min() >= 0
        assert codes.max() < 2**encoder.layout.total_bits


def test_first_probe_is_the_query_own_cell(points: np.ndarray) -> None:
    """The nearest cell to an indexed point must be the cell holding it."""
    for encoder in _encoders(points):
        codes = encoder.encode(points)
        for depth in (1, encoder.layout.depth):
            for i in (0, 100, 359):
                cell = encoder.probe(points[i], depth=depth, n_probes=1)[0]
                span = encoder.layout.prefix_range(cell)
                assert span.lo <= codes[i] <= span.hi


def test_probe_returns_distinct_cells_up_to_the_budget(points: np.ndarray) -> None:
    """Probe returns distinct cells up to the budget."""
    for encoder in _encoders(points):
        cells = encoder.probe(points[3], depth=2, n_probes=PROBE_BUDGET)
        assert len(cells) == len(set(cells))
        assert len(cells) <= PROBE_BUDGET


def test_probe_rejects_depth_beyond_the_layout(points: np.ndarray) -> None:
    """Probe rejects depth beyond the layout."""
    encoder = fit_hierarchical_kmeans(points, levels=2, branching=4, seed=0)
    with pytest.raises(ValueError, match="depth"):
        encoder.probe(points[0], depth=3, n_probes=1)
