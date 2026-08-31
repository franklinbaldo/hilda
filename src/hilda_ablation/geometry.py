# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The geometry the whole benchmark agrees on.

Ground truth is cosine, so every encoder is fitted, and every query probed, on
the same unit sphere. Fitting on raw embeddings while scoring against cosine
would let norm variation into the representation and out of the metric.
"""

from __future__ import annotations

import numpy as np


def unit_norm(points: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving zero rows untouched."""
    rows = np.atleast_2d(np.asarray(points, dtype=np.float64))
    lengths = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(lengths == 0.0, 1.0, lengths)
