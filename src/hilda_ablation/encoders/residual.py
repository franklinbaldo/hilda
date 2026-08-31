# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Residual family: one shared codebook per level, each quantising the residual.

Unlike the tree, digit 2 addresses a correction to digit 1 rather than a
subcategory of it. Whether that costs prefix semantics is exactly what the
ablation is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from hilda_ablation.codes import CodeLayout
from hilda_ablation.encoders.protocol import validate_depth


@dataclass
class ResidualKMeansEncoder:
    """Codes are residual vector quantisation stages, packed coarse to fine."""

    name: str
    codebooks: list[np.ndarray]
    branching: int
    layout: CodeLayout = field(init=False)

    def __post_init__(self) -> None:
        """Validate the declared shape at construction."""
        width = max(1, int(np.ceil(np.log2(self.branching))))
        self.layout = CodeLayout(digit_bits=(width,) * len(self.codebooks))

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) integer codes."""
        residual = np.atleast_2d(points).astype(np.float64)
        digits = np.zeros((len(residual), len(self.codebooks)), dtype=np.int64)
        for level, codebook in enumerate(self.codebooks):
            chosen = np.argmin(
                ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2),
                axis=1,
            )
            digits[:, level] = chosen
            residual = residual - codebook[chosen]
        return self.layout.pack(digits)

    def reconstruct(self, points: np.ndarray) -> np.ndarray:
        """Sum the codewords each point selects, stage by stage."""
        residual = np.atleast_2d(points).astype(np.float64)
        total = np.zeros_like(residual)
        for codebook in self.codebooks:
            chosen = np.argmin(
                ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2),
                axis=1,
            )
            total = total + codebook[chosen]
            residual = residual - codebook[chosen]
        return total

    def probe(
        self,
        query: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> list[tuple[int, ...]]:
        """Return the cells nearest the query, nearest first."""
        validate_depth(self.layout, depth)
        point = np.asarray(query, dtype=np.float64).reshape(-1)
        beam: list[tuple[float, tuple[int, ...], np.ndarray]] = [(0.0, (), point)]
        for level in range(depth):
            beam = self._expand(self.codebooks[level], beam, n_probes)
        return [path for _, path, _ in beam[:n_probes]]

    def _expand(
        self,
        codebook: np.ndarray,
        beam: list[tuple[float, tuple[int, ...], np.ndarray]],
        n_probes: int,
    ) -> list[tuple[float, tuple[int, ...], np.ndarray]]:
        """Extend every beam path by one residual stage."""
        expanded: list[tuple[float, tuple[int, ...], np.ndarray]] = []
        for _, path, residual in beam:
            distances = np.linalg.norm(codebook - residual, axis=1)
            expanded.extend(
                (float(distances[code]), (*path, int(code)), residual - codebook[code])
                for code in np.argsort(distances, kind="stable")[:n_probes]
            )
        expanded.sort(key=lambda item: (item[0], item[1]))
        return expanded[:n_probes]


def fit_residual_kmeans(
    points: np.ndarray,
    levels: int,
    branching: int,
    seed: int,
) -> ResidualKMeansEncoder:
    """Fit the residual kmeans variant on the corpus."""
    residual = np.asarray(points, dtype=np.float64)
    codebooks: list[np.ndarray] = []
    for level in range(levels):
        clusters = min(branching, len(residual))
        kmeans = KMeans(n_clusters=clusters, n_init=4, random_state=seed + level).fit(
            residual,
        )
        codebooks.append(kmeans.cluster_centers_)
        residual = residual - kmeans.cluster_centers_[kmeans.labels_]
    return ResidualKMeansEncoder(
        name=f"rvq-L{levels}xK{branching}",
        codebooks=codebooks,
        branching=branching,
    )
