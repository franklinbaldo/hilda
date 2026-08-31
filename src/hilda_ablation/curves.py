# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Space-filling curves used as the ordering step of an SFC-family encoder.

Hilbert follows Skilling's transpose algorithm (AIP Conf. Proc. 707, 2004),
which generalises to any number of dimensions; Morton (Z-order) is the cheap
multidimensional control that isolates "does the curve matter" from "does the
dimension matter".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SpaceFillingCurve(Protocol):
    """A bijection between a 2**bits-per-axis grid and a linear order."""

    dims: int
    bits: int

    def index(self, coords: np.ndarray) -> np.ndarray:
        """Map an (n, dims) integer grid to (n,) curve positions."""
        ...

    def coords(self, indices: np.ndarray) -> np.ndarray:
        """Inverse of `index`."""
        ...

    def at_depth(self, depth: int) -> SpaceFillingCurve:
        """Return the same curve over the coarser grid of the top `depth` bits."""
        ...


def _validate(dims: int, bits: int) -> None:
    """Reject a degenerate grid."""
    if dims < 1 or bits < 1:
        msg = f"dims and bits must be positive, got dims={dims}, bits={bits}"
        raise ValueError(msg)


def _interleave(axes: np.ndarray, bits: int, dims: int) -> np.ndarray:
    """Pack axis bit-planes, most significant plane first."""
    indices = np.zeros(len(axes), dtype=np.int64)
    for plane in range(bits - 1, -1, -1):
        for axis in range(dims):
            indices = (indices << 1) | ((axes[:, axis] >> plane) & 1)
    return indices


def _deinterleave(indices: np.ndarray, bits: int, dims: int) -> np.ndarray:
    """Unpack axis bit-planes, most significant plane first."""
    axes = np.zeros((len(indices), dims), dtype=np.int64)
    for plane in range(bits - 1, -1, -1):
        for axis in range(dims):
            shift = plane * dims + (dims - 1 - axis)
            axes[:, axis] |= ((indices >> shift) & 1) << plane
    return axes


@dataclass(frozen=True)
class MortonCurve:
    """Z-order: plain bit interleaving."""

    dims: int
    bits: int

    def __post_init__(self) -> None:
        """Reject a degenerate grid."""
        _validate(self.dims, self.bits)

    def index(self, coords: np.ndarray) -> np.ndarray:
        """Map an (n, dims) integer grid to (n,) curve positions."""
        return _interleave(np.asarray(coords, dtype=np.int64), self.bits, self.dims)

    def coords(self, indices: np.ndarray) -> np.ndarray:
        """Inverse of `index`."""
        return _deinterleave(np.asarray(indices, dtype=np.int64), self.bits, self.dims)

    def at_depth(self, depth: int) -> MortonCurve:
        """Return the same curve over the coarser grid of the top `depth` bits."""
        return MortonCurve(dims=self.dims, bits=depth)


@dataclass(frozen=True)
class HilbertCurve:
    """Multidimensional Hilbert curve via Skilling's transpose."""

    dims: int
    bits: int

    def __post_init__(self) -> None:
        """Reject a degenerate grid."""
        _validate(self.dims, self.bits)

    def index(self, coords: np.ndarray) -> np.ndarray:
        """Map an (n, dims) integer grid to (n,) curve positions."""
        transposed = self._axes_to_transpose(np.array(coords, dtype=np.int64))
        return _interleave(transposed, self.bits, self.dims)

    def coords(self, indices: np.ndarray) -> np.ndarray:
        """Inverse of `index`."""
        transposed = _deinterleave(
            np.asarray(indices, dtype=np.int64),
            self.bits,
            self.dims,
        )
        return self._transpose_to_axes(transposed)

    def at_depth(self, depth: int) -> HilbertCurve:
        """Return the same curve over the coarser grid of the top `depth` bits."""
        return HilbertCurve(dims=self.dims, bits=depth)

    def _axes_to_transpose(self, axes: np.ndarray) -> np.ndarray:
        """Apply Skilling's forward transform in place."""
        high = 1 << (self.bits - 1)
        step = high
        while step > 1:
            lower = step - 1
            for axis in range(self.dims):
                flip = (axes[:, axis] & step) != 0
                axes[flip, 0] ^= lower
                keep = ~flip
                swap = (axes[keep, 0] ^ axes[keep, axis]) & lower
                axes[keep, 0] ^= swap
                axes[keep, axis] ^= swap
            step >>= 1
        for axis in range(1, self.dims):
            axes[:, axis] ^= axes[:, axis - 1]
        correction = np.zeros(len(axes), dtype=np.int64)
        step = high
        while step > 1:
            correction ^= np.where((axes[:, self.dims - 1] & step) != 0, step - 1, 0)
            step >>= 1
        for axis in range(self.dims):
            axes[:, axis] ^= correction
        return axes

    def _transpose_to_axes(self, axes: np.ndarray) -> np.ndarray:
        """Apply Skilling's inverse transform in place."""
        top = 2 << (self.bits - 1)
        correction = axes[:, self.dims - 1] >> 1
        for axis in range(self.dims - 1, 0, -1):
            axes[:, axis] ^= axes[:, axis - 1]
        axes[:, 0] ^= correction
        step = 2
        while step != top:
            lower = step - 1
            for axis in range(self.dims - 1, -1, -1):
                flip = (axes[:, axis] & step) != 0
                axes[flip, 0] ^= lower
                keep = ~flip
                swap = (axes[keep, 0] ^ axes[keep, axis]) & lower
                axes[keep, 0] ^= swap
                axes[keep, axis] ^= swap
            step <<= 1
        return axes
