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
from hilda_ablation.geometry import unit_norm

if TYPE_CHECKING:
    from hilda_ablation.encoders.protocol import Encoder


def exact_neighbours(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: top-k by cosine similarity, brute force."""
    similarity = unit_norm(queries) @ unit_norm(corpus).T
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


PROBE_CEILING = 256
"""How wide a probe the budget walk may open before giving up on filling it."""


def take_budget(
    encoder: Encoder,
    index: CodeIndex,
    query: np.ndarray,
    depth: int,
    budget: int,
) -> ScanResult:
    """Scan cells nearest-first and stop at `budget` candidates for this query.

    A budget met on average is not a budget: unbalanced cells let one query pay
    far more than the mean while the table still calls it cheap. Truncating
    nearest-first gives every encoder the same candidate count on every query.
    """
    cells = encoder.probe(query, depth=depth, n_probes=PROBE_CEILING)
    taken: list[np.ndarray] = []
    spent: list[IndexRange] = []
    remaining = budget
    for cell in cells:
        span = encoder.layout.prefix_range(cell)
        members = index.scan([span]).members
        if members.size == 0:
            continue
        taken.append(members[:remaining])
        spent.append(span)
        remaining -= min(members.size, remaining)
        if remaining <= 0:
            break
    if not taken:
        return ScanResult(members=np.array([], dtype=np.int64), n_scanned=0, n_ranges=0)
    members = np.concatenate(taken)
    return ScanResult(
        members=members,
        n_scanned=int(members.size),
        n_ranges=len(merge_ranges(spent)),
    )


@dataclass(frozen=True)
class ScanDistribution:
    """Per-query scan cost. A budget met on average is not a budget per query."""

    mean: float
    p50: float
    p95: float
    maximum: float

    @classmethod
    def of(cls, fractions: np.ndarray) -> ScanDistribution:
        """Summarise the per-query scan fractions of one operating point."""
        return cls(
            mean=float(fractions.mean()),
            p50=float(np.percentile(fractions, 50)),
            p95=float(np.percentile(fractions, 95)),
            maximum=float(fractions.max()),
        )


@dataclass(frozen=True)
class OperatingPoint:
    """One (depth, probes) setting of one encoder, averaged over queries."""

    encoder: str
    depth: int
    n_probes: int
    budgeted: bool
    recall: float
    recall_stderr: float
    scanned: ScanDistribution
    n_ranges: float

    def as_row(self) -> dict[str, str | int | float]:
        """Flatten to a CSV row."""
        return {
            "encoder": self.encoder,
            "depth": self.depth,
            "n_probes": self.n_probes,
            "budgeted": int(self.budgeted),
            "recall": round(self.recall, 4),
            "recall_stderr": round(self.recall_stderr, 4),
            "scan_mean": round(self.scanned.mean, 5),
            "scan_p50": round(self.scanned.p50, 5),
            "scan_p95": round(self.scanned.p95, 5),
            "scan_max": round(self.scanned.maximum, 5),
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
    """One operating point of an encoder: how deep to address, how wide to probe.

    `n_probes` is a probe width; `budget` instead caps candidates per query, and
    the two are alternatives, not a pair.
    """

    depth: int
    n_probes: int | None = None
    budget: int | None = None

    def __post_init__(self) -> None:
        """Reject a setting that names neither cost or both."""
        if (self.n_probes is None) == (self.budget is None):
            message = "set exactly one of n_probes and budget"
            raise ValueError(message)

    @property
    def width(self) -> int:
        """The cost knob's value, whichever knob this setting uses."""
        return self.n_probes if self.n_probes is not None else self.budget


def _plan(
    encoder: Encoder, index: CodeIndex, query: np.ndarray, setting: Setting
) -> ScanResult:
    """Run one query's scan plan, by probe width or by candidate budget."""
    if setting.budget is not None:
        return take_budget(
            encoder, index, query, depth=setting.depth, budget=setting.budget
        )
    cells = encoder.probe(query, depth=setting.depth, n_probes=setting.n_probes)
    return index.scan([encoder.layout.prefix_range(cell) for cell in cells])


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
        result = _plan(encoder, index, query, setting)
        recalls[i] = np.isin(queries.truth[i], result.members).sum() / queries.k
        scanned[i] = result.n_scanned / corpus_size
        ranges[i] = result.n_ranges
    return OperatingPoint(
        encoder=encoder.name,
        depth=setting.depth,
        n_probes=setting.width,
        budgeted=setting.budget is not None,
        recall=float(recalls.mean()),
        recall_stderr=float(recalls.std(ddof=1) / np.sqrt(len(recalls))),
        scanned=ScanDistribution.of(scanned),
        n_ranges=float(ranges.mean()),
    )
