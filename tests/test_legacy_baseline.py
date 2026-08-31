# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The legacy scaling must reproduce the min-max grid the repo benchmark uses."""

import numpy as np

from hilda_ablation.encoders import fit_sfc
from hilda_ablation.projections import fit_pca

BITS = 4


def _skewed_points() -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.concatenate([rng.normal(size=(300, 6)), rng.normal(size=(20, 6)) * 30])


def test_minmax_scaling_leaves_the_grid_unequally_filled() -> None:
    """Min-max scaling leaves the grid unequally filled."""
    points = _skewed_points()
    encoder = fit_sfc(
        points,
        projection=fit_pca(points, dims=2),
        bits=BITS,
        curve="hilbert",
        scaling="minmax",
    )
    occupancy = np.bincount(encoder.encode(points))
    assert occupancy.max() > len(points) / 4


def test_quantile_scaling_spreads_the_corpus_across_cells() -> None:
    """Quantile scaling spreads the corpus across cells."""
    points = _skewed_points()
    encoder = fit_sfc(
        points,
        projection=fit_pca(points, dims=2),
        bits=BITS,
        curve="hilbert",
        scaling="quantile",
    )
    occupancy = np.bincount(encoder.encode(points))
    assert occupancy.max() < len(points) / 4
