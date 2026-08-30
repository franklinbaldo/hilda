# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Code layout and range algebra."""

import numpy as np
import pytest

from hilda_ablation.codes import CodeLayout, IndexRange, merge_ranges

TWO_LEVELS_OF_TWO_BITS = 4


def test_layout_packs_digits_most_significant_level_first() -> None:
    """Layout packs digits most significant level first."""
    layout = CodeLayout(digit_bits=(2, 2))
    assert layout.total_bits == TWO_LEVELS_OF_TWO_BITS
    assert layout.pack(np.array([[1, 2]])).tolist() == [0b0110]


def test_prefix_range_covers_exactly_its_children() -> None:
    """Prefix range covers exactly its children."""
    layout = CodeLayout(digit_bits=(2, 2))
    assert layout.prefix_range((1,)) == IndexRange(lo=0b0100, hi=0b0111)
    assert layout.prefix_range((1, 2)) == IndexRange(lo=0b0110, hi=0b0110)


def test_full_depth_prefix_is_a_single_code() -> None:
    """Full depth prefix is a single code."""
    layout = CodeLayout(digit_bits=(3, 3))
    rng = layout.prefix_range((5, 5))
    assert rng.lo == rng.hi == layout.pack(np.array([[5, 5]]))[0]


def test_merge_joins_adjacent_and_overlapping_ranges() -> None:
    """Merge joins adjacent and overlapping ranges."""
    merged = merge_ranges(
        [IndexRange(4, 7), IndexRange(0, 3), IndexRange(20, 25), IndexRange(22, 30)],
    )
    assert merged == [IndexRange(0, 7), IndexRange(20, 30)]


def test_merge_of_nothing_is_nothing() -> None:
    """Merge of nothing is nothing."""
    assert merge_ranges([]) == []


def test_range_rejects_inverted_bounds() -> None:
    """Range rejects inverted bounds."""
    with pytest.raises(ValueError, match="lo"):
        IndexRange(lo=5, hi=4)
