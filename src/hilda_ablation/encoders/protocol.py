# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""What every encoder family must offer the evaluation loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from hilda_ablation.codes import CodeLayout


class Encoder(Protocol):
    """Maps embeddings to prefix-sortable codes and picks cells to scan."""

    name: str
    layout: CodeLayout

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) integer codes."""
        ...

    def probe(
        self,
        query: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> list[tuple[int, ...]]:
        """Return the `n_probes` cells nearest the query, nearest first."""
        ...


def validate_depth(layout: CodeLayout, depth: int) -> None:
    """Reject a depth the layout cannot address."""
    if not 1 <= depth <= layout.depth:
        msg = f"depth must be in 1..{layout.depth}, got depth={depth}"
        raise ValueError(msg)
