# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Linear maps from embedding space to the low-dimensional space an SFC orders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Projection:
    """A fitted affine projection, kept explicit so runs stay reproducible."""

    name: str
    mean: np.ndarray
    matrix: np.ndarray
    explained_variance_ratio: float | None

    @property
    def dims(self) -> int:
        """Width of the projected space."""
        return self.matrix.shape[1]

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Project (n, d) embeddings down to (n, dims)."""
        return (np.atleast_2d(points) - self.mean) @ self.matrix


def fit_pca(points: np.ndarray, dims: int) -> Projection:
    """Fit the principal axes of the corpus."""
    pca = PCA(n_components=dims).fit(points)
    return Projection(
        name=f"pca{dims}",
        mean=pca.mean_,
        matrix=pca.components_.T,
        explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
    )


def fit_random_projection(points: np.ndarray, dims: int, seed: int) -> Projection:
    """Fit a Gaussian random projection, the control for what PCA adds."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(points.shape[1], dims)) / np.sqrt(dims)
    return Projection(
        name=f"rp{dims}-s{seed}",
        mean=points.mean(axis=0),
        matrix=matrix,
        explained_variance_ratio=None,
    )
