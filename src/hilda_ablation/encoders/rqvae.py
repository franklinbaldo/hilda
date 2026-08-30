# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Learned family: an RQ-VAE, in the shape TIGER popularised for semantic IDs.

An MLP compresses the embedding, residual codebooks quantise the latent, and a
decoder pushes reconstruction back into the codebooks during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hilda_ablation.encoders.residual import ResidualKMeansEncoder, fit_residual_kmeans

if TYPE_CHECKING:
    from hilda_ablation.codes import CodeLayout


class TorchMissingError(RuntimeError):
    """Raised when the learned variant is requested without torch installed."""

    def __init__(self) -> None:
        """Record the structured detail the caller needs."""
        super().__init__("the rqvae variant needs the 'rqvae' extra (torch)")


@dataclass
class RqVaeEncoder:
    """A learned latent space with residual codebooks fitted inside it."""

    name: str
    weights: list[tuple[np.ndarray, np.ndarray]]
    quantiser: ResidualKMeansEncoder
    layout: CodeLayout = field(init=False)

    def __post_init__(self) -> None:
        """Validate the declared shape at construction."""
        self.layout = self.quantiser.layout

    def latent(self, points: np.ndarray) -> np.ndarray:
        """Run the trained encoder MLP forward in numpy."""
        activations = np.atleast_2d(points).astype(np.float64)
        for depth, (matrix, bias) in enumerate(self.weights):
            activations = activations @ matrix + bias
            if depth < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)
        return activations

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) integer codes."""
        return self.quantiser.encode(self.latent(points))

    def probe(
        self,
        query: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> list[tuple[int, ...]]:
        """Return the cells nearest the query, nearest first."""
        return self.quantiser.probe(
            self.latent(query)[0],
            depth=depth,
            n_probes=n_probes,
        )


@dataclass(frozen=True)
class RqVaeSpec:
    """Training knobs for the learned variant, kept together and on the record."""

    levels: int
    branching: int
    seed: int
    latent_dims: int = 32
    epochs: int = 40


def fit_rqvae(points: np.ndarray, spec: RqVaeSpec) -> RqVaeEncoder:
    """Train the autoencoder, then fit residual codebooks in its latent space."""
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TorchMissingError from exc

    torch.manual_seed(spec.seed)
    data = torch.tensor(np.asarray(points, dtype=np.float32))
    encoder = nn.Sequential(
        nn.Linear(data.shape[1], 256), nn.ReLU(), nn.Linear(256, spec.latent_dims)
    )
    decoder = nn.Sequential(
        nn.Linear(spec.latent_dims, 256), nn.ReLU(), nn.Linear(256, data.shape[1])
    )
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=256, shuffle=True
    )
    for _ in range(spec.epochs):
        for (batch,) in loader:
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(decoder(encoder(batch)), batch)
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        latents = encoder(data).numpy().astype(np.float64)
    weights = [
        (
            layer.weight.detach().numpy().T.astype(np.float64),
            layer.bias.detach().numpy().astype(np.float64),
        )
        for layer in encoder
        if isinstance(layer, nn.Linear)
    ]
    quantiser = fit_residual_kmeans(
        latents, levels=spec.levels, branching=spec.branching, seed=spec.seed
    )
    return RqVaeEncoder(
        name=f"rqvae-L{spec.levels}xK{spec.branching}",
        weights=weights,
        quantiser=quantiser,
    )
