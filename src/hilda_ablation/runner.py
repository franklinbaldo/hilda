# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Assemble the encoder roster and sweep it over one comparable axis.

Every family is swept by *prefix bits*, not by its own notion of depth, so a
point on one curve costs the same address space as a point on another.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hilda_ablation.encoders import (
    fit_hierarchical_kmeans,
    fit_residual_kmeans,
    fit_sfc,
)
from hilda_ablation.evaluation import (
    CodeIndex,
    QuerySet,
    Setting,
    exact_neighbours,
    measure,
)
from hilda_ablation.projections import fit_pca, fit_random_projection

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from hilda_ablation.corpus import Corpus
    from hilda_ablation.encoders import Encoder
    from hilda_ablation.evaluation import OperatingPoint

logger = logging.getLogger(__name__)

SEMANTIC_BITS = 60
"""The HILDA semantic field: 128 bits minus version, timestamp and tie-break."""


@dataclass(frozen=True)
class SweepGrid:
    """The comparable operating points every encoder is asked to hit."""

    prefix_bits: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20)
    probes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    k: int = 10

    def depths_for(self, encoder: Encoder) -> list[tuple[int, int]]:
        """Return the depths this layout can reach, with the bits they spend."""
        width = encoder.layout.digit_bits[0]
        reachable = []
        for bits in self.prefix_bits:
            depth, remainder = divmod(bits, width)
            if remainder or not 1 <= depth <= encoder.layout.depth:
                continue
            reachable.append((bits, depth))
        return reachable


@dataclass(frozen=True)
class RosterConfig:
    """Which variants to fit. Defaults mirror the ablation table in the plan."""

    sfc_dims: tuple[int, ...] = (2, 3, 4)
    random_projection_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    quantiser_shapes: tuple[tuple[int, int], ...] = ((4, 8), (6, 4))
    include_rqvae: bool = True
    seed: int = 0


def _sfc_family(points: np.ndarray, config: RosterConfig) -> list[Encoder]:
    """Project, then order by a curve: the current HILDA plus its controls."""
    encoders: list[Encoder] = []
    for dims in config.sfc_dims:
        projection = fit_pca(points, dims=dims)
        bits = SEMANTIC_BITS // dims
        encoders.extend(
            fit_sfc(points, projection=projection, bits=bits, curve=curve)
            for curve in ("hilbert", "morton")
        )
    encoders.extend(
        fit_sfc(
            points,
            projection=fit_random_projection(points, dims=2, seed=seed),
            bits=SEMANTIC_BITS // 2,
            curve="hilbert",
        )
        for seed in config.random_projection_seeds
    )
    return encoders


def _quantiser_family(points: np.ndarray, config: RosterConfig) -> list[Encoder]:
    """Fit the tree and residual quantisers, where a prefix is a cluster."""
    encoders: list[Encoder] = []
    for levels, branching in config.quantiser_shapes:
        encoders.append(
            fit_hierarchical_kmeans(
                points,
                levels=levels,
                branching=branching,
                seed=config.seed,
            ),
        )
        encoders.append(
            fit_residual_kmeans(
                points,
                levels=levels,
                branching=branching,
                seed=config.seed,
            ),
        )
    return encoders


def _fit_rqvae_or_skip(points: np.ndarray, config: RosterConfig) -> Encoder | None:
    """Fit the learned variant, or report that torch is absent."""
    from hilda_ablation.encoders.rqvae import (  # noqa: PLC0415
        RqVaeSpec,
        TorchMissingError,
        fit_rqvae,
    )

    levels, branching = config.quantiser_shapes[0]
    spec = RqVaeSpec(levels=levels, branching=branching, seed=config.seed)
    try:
        return fit_rqvae(points, spec)
    except TorchMissingError:
        logger.warning("skipping rqvae: torch is not installed")
        return None


def build_roster(points: np.ndarray, config: RosterConfig) -> list[Encoder]:
    """Fit every encoder under a shared 60-bit semantic budget."""
    encoders: list[Encoder | None] = []
    encoders.extend(_sfc_family(points, config))
    encoders.extend(_quantiser_family(points, config))
    if config.include_rqvae:
        encoders.append(_fit_rqvae_or_skip(points, config))
    return [encoder for encoder in encoders if encoder is not None]


@dataclass
class AblationResult:
    """Every operating point measured, plus the projections' variance record."""

    points: list[OperatingPoint] = field(default_factory=list)
    notes: dict[str, float] = field(default_factory=dict)

    def write_notes(self, path: Path) -> None:
        """Write the run's scalar record, so the report needs no log file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.notes, indent=2, sort_keys=True) + "\n")

    def write_csv(self, path: Path) -> None:
        """Write every operating point as one CSV row."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [point.as_row() for point in self.points]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_ablation(
    corpus: Corpus,
    roster: RosterConfig,
    grid: SweepGrid,
) -> AblationResult:
    """Fit the roster on the corpus and measure it against exact cosine truth."""
    queries = QuerySet(
        queries=corpus.queries,
        truth=exact_neighbours(corpus.documents, corpus.queries, k=grid.k),
    )
    result = AblationResult()
    for dims in roster.sfc_dims:
        projection = fit_pca(corpus.documents, dims=dims)
        result.notes[f"pca{dims}_explained_variance"] = (
            projection.explained_variance_ratio
        )
    for encoder in build_roster(corpus.documents, roster):
        index = CodeIndex(codes=encoder.encode(corpus.documents))
        logger.info("measuring %s", encoder.name)
        for bits, depth in grid.depths_for(encoder):
            for probes in grid.probes:
                point = measure(
                    encoder,
                    index,
                    queries,
                    Setting(depth=depth, n_probes=probes),
                )
                result.points.append(point)
                logger.debug(
                    "%s /%d x%d %.3f", encoder.name, bits, probes, point.recall
                )
    return result
