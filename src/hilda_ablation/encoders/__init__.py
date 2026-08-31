# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Encoder families under ablation, all speaking the same code protocol."""

from hilda_ablation.encoders.protocol import Encoder
from hilda_ablation.encoders.residual import fit_residual_kmeans
from hilda_ablation.encoders.sfc import fit_sfc
from hilda_ablation.encoders.tree import fit_hierarchical_kmeans

__all__ = [
    "Encoder",
    "fit_hierarchical_kmeans",
    "fit_residual_kmeans",
    "fit_sfc",
]
