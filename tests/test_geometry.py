# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The geometry every stage of the benchmark must agree on."""

import numpy as np

from hilda_ablation.geometry import unit_norm

RADIUS = 1.0


def test_unit_norm_puts_every_row_on_the_sphere() -> None:
    """Unit norm puts every row on the sphere."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(50, 8)) * rng.uniform(0.1, 20, size=(50, 1))
    lengths = np.linalg.norm(unit_norm(points), axis=1)
    assert np.allclose(lengths, RADIUS)


def test_unit_norm_preserves_direction() -> None:
    """Unit norm preserves direction."""
    points = np.array([[3.0, 4.0]])
    assert np.allclose(unit_norm(points), [[0.6, 0.8]])


def test_unit_norm_leaves_a_zero_row_alone() -> None:
    """Unit norm leaves a zero row alone."""
    assert np.allclose(unit_norm(np.array([[0.0, 0.0]])), [[0.0, 0.0]])
