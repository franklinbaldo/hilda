# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Tree family: hierarchical k-means, where a prefix is literally a subtree.

This is the purest reading of sCIDR: /1 is a root cluster, /2 a subcluster,
and so on down the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from hilda_ablation.codes import CodeLayout
from hilda_ablation.encoders.protocol import validate_depth


@dataclass
class _Node:
    centroids: np.ndarray
    children: list[_Node | None]

    def nearest(self, query: np.ndarray) -> np.ndarray:
        """Child indices ordered by distance from the query to their centroid."""
        distances = np.linalg.norm(self.centroids - query, axis=1)
        return np.argsort(distances, kind="stable")


def _fit_node(points: np.ndarray, levels: int, branching: int, seed: int) -> _Node:
    """Cluster this node, then recurse into each child."""
    clusters = min(branching, len(points))
    kmeans = KMeans(n_clusters=clusters, n_init=4, random_state=seed).fit(points)
    children: list[_Node | None] = []
    for label in range(clusters):
        members = points[kmeans.labels_ == label]
        if levels <= 1 or len(members) < branching:
            # Fewer points than clusters is not a split but a relabelling: the
            # node becomes a leaf and the remaining digits stay zero.
            children.append(None)
            continue
        children.append(_fit_node(members, levels - 1, branching, seed))
    return _Node(centroids=kmeans.cluster_centers_, children=children)


@dataclass
class HierarchicalKMeansEncoder:
    """Codes are root-to-leaf paths through a k-means tree."""

    name: str
    root: _Node
    levels: int
    branching: int
    layout: CodeLayout = field(init=False)

    def __post_init__(self) -> None:
        """Validate the declared shape at construction."""
        width = max(1, int(np.ceil(np.log2(self.branching))))
        self.layout = CodeLayout(digit_bits=(width,) * self.levels)

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) integer codes."""
        rows = np.atleast_2d(points)
        digits = np.array([self._path(row) for row in rows])
        return self.layout.pack(digits)

    def probe(
        self,
        query: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> list[tuple[int, ...]]:
        """Return the cells nearest the query, nearest first."""
        validate_depth(self.layout, depth)
        beam: list[tuple[float, tuple[int, ...], _Node | None]] = [(0.0, (), self.root)]
        for _ in range(depth):
            beam = self._expand(np.asarray(query), beam, n_probes)
        return [path for _, path, _ in beam[:n_probes]]

    def _expand(
        self,
        query: np.ndarray,
        beam: list[tuple[float, tuple[int, ...], _Node | None]],
        n_probes: int,
    ) -> list[tuple[float, tuple[int, ...], _Node | None]]:
        """Extend every beam path by one level of the tree."""
        expanded: list[tuple[float, tuple[int, ...], _Node | None]] = []
        for score, path, node in beam:
            if node is None:
                expanded.append((score, (*path, 0), None))
                continue
            distances = np.linalg.norm(node.centroids - query, axis=1)
            expanded.extend(
                (float(distances[child]), (*path, int(child)), node.children[child])
                for child in node.nearest(query)[:n_probes]
            )
        expanded.sort(key=lambda item: (item[0], item[1]))
        return expanded[:n_probes]

    def _path(self, point: np.ndarray) -> list[int]:
        """Walk one point from root to leaf."""
        digits: list[int] = []
        node: _Node | None = self.root
        for _ in range(self.levels):
            if node is None:
                digits.append(0)
                continue
            child = int(np.argmin(np.linalg.norm(node.centroids - point, axis=1)))
            digits.append(child)
            node = node.children[child]
        return digits


def fit_hierarchical_kmeans(
    points: np.ndarray,
    levels: int,
    branching: int,
    seed: int,
) -> HierarchicalKMeansEncoder:
    """Fit the hierarchical kmeans variant on the corpus."""
    return HierarchicalKMeansEncoder(
        name=f"hkmeans-L{levels}xK{branching}",
        root=_fit_node(points, levels, branching, seed),
        levels=levels,
        branching=branching,
    )
