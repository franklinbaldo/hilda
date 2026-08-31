# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Fixed-width digit codes and the B-tree ranges their prefixes address.

Every encoder in the ablation emits a code built the same way: a tuple of
digits, one per level, each of a fixed bit width, packed most-significant
level first. A depth-r prefix is then a contiguous integer range, which is
the only property an ordinary ordered index needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndexRange:
    """An inclusive range of code values, i.e. one B-tree range scan."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        """Reject inverted bounds at construction, not at scan time."""
        if self.lo > self.hi:
            msg = f"lo={self.lo} exceeds hi={self.hi}"
            raise ValueError(msg)

    def touches(self, other: IndexRange) -> bool:
        """Whether the two ranges overlap or sit back to back."""
        return other.lo <= self.hi + 1 and self.lo <= other.hi + 1

    def union(self, other: IndexRange) -> IndexRange:
        """Return the smallest range covering both."""
        return IndexRange(lo=min(self.lo, other.lo), hi=max(self.hi, other.hi))


def merge_ranges(ranges: list[IndexRange]) -> list[IndexRange]:
    """Collapse overlapping and adjacent ranges, so counts reflect real scans."""
    if not ranges:
        return []
    merged: list[IndexRange] = []
    for current in sorted(ranges, key=lambda r: (r.lo, r.hi)):
        if merged and merged[-1].touches(current):
            merged[-1] = merged[-1].union(current)
            continue
        merged.append(current)
    return merged


@dataclass(frozen=True)
class CodeLayout:
    """How many bits each level of a code occupies."""

    digit_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject a layout that could not address anything."""
        if not self.digit_bits or any(b <= 0 for b in self.digit_bits):
            msg = f"digit_bits must be positive and non-empty, got {self.digit_bits}"
            raise ValueError(msg)

    @property
    def depth(self) -> int:
        """Number of levels in the code."""
        return len(self.digit_bits)

    @property
    def total_bits(self) -> int:
        """Width of a full code."""
        return sum(self.digit_bits)

    def shift_after(self, depth: int) -> int:
        """Bits remaining below a depth-`depth` prefix."""
        return sum(self.digit_bits[depth:])

    def pack(self, digits: np.ndarray) -> np.ndarray:
        """Pack an (n, depth) digit matrix into (n,) integer codes."""
        if digits.shape[1] != self.depth:
            msg = f"expected {self.depth} digits per row, got {digits.shape[1]}"
            raise ValueError(msg)
        codes = np.zeros(len(digits), dtype=np.int64)
        for level, bits in enumerate(self.digit_bits):
            codes = (codes << bits) | digits[:, level].astype(np.int64)
        return codes

    def prefix_range(self, prefix: tuple[int, ...]) -> IndexRange:
        """Return the code range addressed by a partial (or full) digit prefix."""
        if len(prefix) > self.depth:
            msg = f"prefix of length {len(prefix)} exceeds depth {self.depth}"
            raise ValueError(msg)
        value = 0
        for level, digit in enumerate(prefix):
            value = (value << self.digit_bits[level]) | int(digit)
        shift = self.shift_after(len(prefix))
        lo = value << shift
        return IndexRange(lo=lo, hi=lo + (1 << shift) - 1)
