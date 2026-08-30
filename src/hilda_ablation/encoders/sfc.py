# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""SFC family: project to few dimensions, quantise, order by a space-filling curve.

This is the current HILDA architecture, kept as the baseline it is.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from hilda_ablation.codes import CodeLayout
from hilda_ablation.curves import HilbertCurve, MortonCurve, SpaceFillingCurve
from hilda_ablation.encoders.protocol import validate_depth

if TYPE_CHECKING:
    from hilda_ablation.projections import Projection

CurveName = Literal["hilbert", "morton"]


def _build_curve(curve: CurveName, dims: int, bits: int) -> SpaceFillingCurve:
    """Build the requested curve over the projected grid."""
    if curve == "hilbert":
        return HilbertCurve(dims=dims, bits=bits)
    return MortonCurve(dims=dims, bits=bits)


@dataclass
class SfcEncoder:
    """Quantile grid + space-filling curve over a linear projection."""

    name: str
    projection: Projection
    curve: SpaceFillingCurve
    anchors: np.ndarray
    layout: CodeLayout = field(init=False)

    def __post_init__(self) -> None:
        """Validate the declared shape at construction."""
        self.layout = CodeLayout(digit_bits=(self.curve.dims,) * self.curve.bits)

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) integer codes."""
        return self.curve.index(self._grid(self.projection.apply(points)))

    def probe(
        self,
        query: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> list[tuple[int, ...]]:
        """Return the cells nearest the query, nearest first."""
        validate_depth(self.layout, depth)
        centre = self._fractional(self.projection.apply(query))[0] / 2 ** (
            self.curve.bits - depth
        )
        cells = self._nearest_cells(centre, depth=depth, n_probes=n_probes)
        indices = self.curve.at_depth(depth).index(cells)
        return [self._digits(index, depth) for index in indices]

    def _grid(self, projected: np.ndarray) -> np.ndarray:
        """Snap projected points onto integer grid cells."""
        return np.floor(self._fractional(projected)).astype(np.int64)

    def _fractional(self, projected: np.ndarray) -> np.ndarray:
        """Continuous grid position, so probe order reflects sub-cell placement.

        The quantile anchors are interpolated rather than materialised per
        cell: at 30 bits per axis there is no storing one edge per cell.
        """
        size = 2**self.curve.bits
        uniform = np.linspace(0.0, 1.0, self.anchors.shape[1])
        columns = [
            np.interp(projected[:, axis], self.anchors[axis], uniform)
            for axis in range(self.curve.dims)
        ]
        return np.clip(np.stack(columns, axis=1) * size, 0.0, size - 1e-9)

    def _nearest_cells(
        self,
        centre: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> np.ndarray:
        """Grow a neighbourhood until it holds enough cells, then rank by distance."""
        size = 2**depth
        home = np.floor(centre).astype(np.int64)
        radius = 1
        while True:
            offsets = np.array(
                list(
                    itertools.product(
                        range(-radius, radius + 1),
                        repeat=self.curve.dims,
                    ),
                ),
            )
            cells = home + offsets
            inside = np.all((cells >= 0) & (cells < size), axis=1)
            cells = cells[inside]
            if len(cells) >= n_probes or radius >= size:
                break
            radius += 1
        distances = np.linalg.norm(cells + 0.5 - centre, axis=1)
        return cells[np.argsort(distances, kind="stable")][:n_probes]

    def _digits(self, index: int, depth: int) -> tuple[int, ...]:
        """Split a curve index into one digit per bit-plane."""
        width = self.curve.dims
        mask = (1 << width) - 1
        return tuple(
            int((index >> (width * (depth - 1 - level))) & mask)
            for level in range(depth)
        )


ANCHOR_COUNT = 4097
"""Quantile anchors per axis: enough to flatten the marginal, cheap to store."""


def fit_sfc(
    points: np.ndarray,
    projection: Projection,
    bits: int,
    curve: CurveName,
) -> SfcEncoder:
    """Flatten each axis onto a uniform grid, the SFC family at its best.

    A quantile transform is the friendliest possible input to a space-filling
    curve: every cell holds a comparable share of the corpus, so the baseline
    loses nothing to a badly scaled axis.
    """
    projected = projection.apply(points)
    quantiles = np.linspace(0.0, 1.0, ANCHOR_COUNT)
    anchors = np.stack(
        [np.quantile(projected[:, axis], quantiles) for axis in range(projection.dims)],
    )
    return SfcEncoder(
        name=f"{curve}-{projection.name}-b{bits}",
        projection=projection,
        curve=_build_curve(curve, projection.dims, bits),
        anchors=anchors,
    )
