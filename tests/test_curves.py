# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Properties every space-filling curve in the ablation must satisfy."""

import itertools

import numpy as np
import pytest

from hilda_ablation.curves import HilbertCurve, MortonCurve, SpaceFillingCurve


def _curves(dims: int, bits: int) -> list[SpaceFillingCurve]:
    """Both curve families, at one grid size."""
    return [HilbertCurve(dims=dims, bits=bits), MortonCurve(dims=dims, bits=bits)]


@pytest.mark.parametrize("dims", [2, 3, 4])
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_index_is_a_bijection_over_the_grid(dims: int, bits: int) -> None:
    """Index is a bijection over the grid."""
    for curve in _curves(dims, bits):
        coords = np.array(list(itertools.product(range(2**bits), repeat=dims)))
        indices = curve.index(coords)
        assert sorted(indices.tolist()) == list(range(2 ** (dims * bits)))
        assert np.array_equal(curve.coords(indices), coords)


@pytest.mark.parametrize("dims", [2, 3, 4])
def test_hilbert_successive_indices_are_grid_neighbours(dims: int) -> None:
    """Hilbert successive indices are grid neighbours."""
    curve = HilbertCurve(dims=dims, bits=3)
    coords = curve.coords(np.arange(2 ** (dims * 3)))
    steps = np.abs(np.diff(coords, axis=0)).sum(axis=1)
    assert np.all(steps == 1)


@pytest.mark.parametrize("dims", [2, 3, 4])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_index_prefix_is_the_coarse_cell_index(dims: int, depth: int) -> None:
    """A depth-r prefix must address the enclosing level-r cell.

    This is what makes a prefix a contiguous B-tree range, so it is the
    property the whole sCIDR story rests on.
    """
    bits = 4
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 2**bits, size=(500, dims))
    for curve in _curves(dims, bits):
        prefix = curve.index(coords) >> (dims * (bits - depth))
        coarse = curve.at_depth(depth).index(coords >> (bits - depth))
        assert np.array_equal(prefix, coarse)
