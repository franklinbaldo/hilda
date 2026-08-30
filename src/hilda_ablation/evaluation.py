# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Measure a code scheme the way a database would pay for it.

Three numbers travel together, because any one of them alone can flatter a
scheme: recall of the true neighbours, the fraction of the corpus scanned to
get it, and the number of separate ranges that scan takes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hilda_ablation.codes import IndexRange, merge_ranges

if TYPE_CHECKING:
    from hilda_ablation.encoders.protocol import Encoder


def exact_neighbours(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: top-k by cosine similarity, brute force."""
    normalised = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    probes = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    similarity = probes @ normalised.T
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(similarity, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


@dataclass(frozen=True)
class ScanResult:
    """What one range-scan plan touched."""

    members: np.ndarray
    n_scanned: int
    n_ranges: int


@dataclass
class CodeIndex:
    """A sorted code column, standing in for the B-tree."""

    codes: np.ndarray

    def __post_init__(self) -> None:
        """Validate the declared shape at construction."""
        self._order = np.argsort(self.codes, kind="stable")
        self._sorted = self.codes[self._order]

    def scan(self, ranges: list[IndexRange]) -> ScanResult:
        """Run a range-scan plan and report what it touched."""
        merged = merge_ranges(ranges)
        if not merged:
            return ScanResult(
                members=np.array([], dtype=np.int64),
                n_scanned=0,
                n_ranges=0,
            )
        slices = [
            self._order[
                np.searchsorted(self._sorted, span.lo, side="left") : np.searchsorted(
                    self._sorted,
                    span.hi,
                    side="right",
                )
            ]
            for span in merged
        ]
        members = np.concatenate(slices) if slices else np.array([], dtype=np.int64)
        return ScanResult(
            members=members,
            n_scanned=int(members.size),
            n_ranges=len(merged),
        )


@dataclass(frozen=True)
class OperatingPoint:
    """One (depth, probes) setting of one encoder, averaged over queries."""

    encoder: str
    depth: int
    n_probes: int
    recall: float
    recall_stderr: float
    fraction_scanned: float
    n_ranges: float

    def as_row(self) -> dict[str, str | int | float]:
        """Flatten to a CSV row."""
        return {
            "encoder": self.encoder,
            "depth": self.depth,
            "n_probes": self.n_probes,
            "recall": round(self.recall, 4),
            "recall_stderr": round(self.recall_stderr, 4),
            "fraction_scanned": round(self.fraction_scanned, 5),
            "n_ranges": round(self.n_ranges, 2),
        }


@dataclass(frozen=True)
class QuerySet:
    """Held-out queries paired with their exact-cosine ground truth."""

    queries: np.ndarray
    truth: np.ndarray

    @property
    def k(self) -> int:
        """Number of true neighbours each query is scored against."""
        return self.truth.shape[1]


@dataclass(frozen=True)
class Setting:
    """One operating point of an encoder: how deep to address, how wide to probe."""

    depth: int
    n_probes: int


def measure(
    encoder: Encoder,
    index: CodeIndex,
    queries: QuerySet,
    setting: Setting,
) -> OperatingPoint:
    """Run every query at one operating point and average the three costs."""
    corpus_size = len(index.codes)
    recalls = np.zeros(len(queries.queries))
    scanned = np.zeros(len(queries.queries))
    ranges = np.zeros(len(queries.queries))
    for i, query in enumerate(queries.queries):
        cells = encoder.probe(query, depth=setting.depth, n_probes=setting.n_probes)
        result = index.scan([encoder.layout.prefix_range(cell) for cell in cells])
        recalls[i] = np.isin(queries.truth[i], result.members).sum() / queries.k
        scanned[i] = result.n_scanned / corpus_size
        ranges[i] = result.n_ranges
    return OperatingPoint(
        encoder=encoder.name,
        depth=setting.depth,
        n_probes=setting.n_probes,
        recall=float(recalls.mean()),
        recall_stderr=float(recalls.std(ddof=1) / np.sqrt(len(recalls))),
        fraction_scanned=float(scanned.mean()),
        n_ranges=float(ranges.mean()),
    )
