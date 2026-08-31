# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Learned family: an RQ-VAE in the shape TIGER uses for semantic IDs.

The quantiser sits inside the training loop. An MLP compresses the embedding,
residual codebooks quantise the latent, the decoder reconstructs from the
*quantised* latent, and a straight-through estimator carries the gradient back
through the lookup, so encoder, decoder and codebooks train together.

Fitting codebooks after freezing the autoencoder is a different model, and a
weaker one; see Rajput et al., "Recommender Systems with Generative Retrieval"
(https://arxiv.org/abs/2305.05065), section 3.1.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hilda_ablation.encoders.residual import (
    ResidualKMeansEncoder,
    fit_residual_kmeans,
)

if TYPE_CHECKING:
    from hilda_ablation.codes import CodeLayout

COMMITMENT_WEIGHT = 0.25
"""How hard the encoder is pulled towards the code it selects."""

WARMUP_SHARE = 0.25
"""Share of epochs spent on reconstruction alone, before seeding the codebooks.

Codebooks seeded from noise never catch a latent space that is still growing,
and collapse. Seeding them by k-means on warmed-up latents is what the RQ-VAE
implementations do, and it leaves the bulk of training joint.
"""


class TorchMissingError(RuntimeError):
    """Raised when the learned variant is requested without torch installed."""

    def __init__(self) -> None:
        """Name the missing extra so the fix is obvious."""
        message = "the rqvae variant needs the 'rqvae' extra (torch)"
        super().__init__(message)


@dataclass
class RqVaeEncoder:
    """A learned latent space whose residual codebooks were trained with it."""

    name: str
    weights: list[tuple[np.ndarray, np.ndarray]]
    quantiser: ResidualKMeansEncoder
    layout: CodeLayout = field(init=False)

    def __post_init__(self) -> None:
        """Inherit the quantiser layout: the latent space adds no bits."""
        self.layout = self.quantiser.layout

    def latent(self, points: np.ndarray) -> np.ndarray:
        """Run the trained encoder MLP forward in numpy."""
        activations = np.atleast_2d(points).astype(np.float64)
        for depth, (matrix, bias) in enumerate(self.weights):
            activations = activations @ matrix + bias
            if depth < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)
        return activations

    def quantised_error(self, points: np.ndarray) -> float:
        """Quantisation error as a share of latent variance.

        Relative, not absolute: an untrained encoder emits near-zero latents
        that any codebook reconstructs cheaply, which would flatter it.
        """
        latents = self.latent(points)
        residual = ((latents - self.quantiser.reconstruct(latents)) ** 2).mean()
        return float(residual / latents.var())

    def encode(self, points: np.ndarray) -> np.ndarray:
        """Map (n, d) embeddings to (n,) packed latent residual codes."""
        return self.quantiser.encode(self.latent(points))

    def probe(
        self,
        query: np.ndarray,
        depth: int,
        n_probes: int,
    ) -> list[tuple[int, ...]]:
        """Beam search the latent codebooks for the cells nearest the query."""
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
    joint: bool = True
    """Train codebooks with the autoencoder. False fits them after freezing it,
    which is the weaker `ae+rvq` model, kept as the variant to measure against."""

    @property
    def family(self) -> str:
        """Name the model this spec actually describes."""
        return "rqvae" if self.joint else "ae+rvq"


def fit_rqvae(points: np.ndarray, spec: RqVaeSpec) -> RqVaeEncoder:
    """Train encoder, decoder and residual codebooks jointly."""
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TorchMissingError from exc

    torch.manual_seed(spec.seed)
    data = torch.tensor(np.asarray(points, dtype=np.float32))
    encoder = nn.Sequential(
        nn.Linear(data.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, spec.latent_dims),
    )
    decoder = nn.Sequential(
        nn.Linear(spec.latent_dims, 256),
        nn.ReLU(),
        nn.Linear(256, data.shape[1]),
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=256,
        shuffle=True,
    )
    warmup = int(spec.epochs * WARMUP_SHARE) if spec.joint else spec.epochs
    reconstruction = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()],
        lr=1e-3,
    )
    for _ in range(warmup):
        for (batch,) in loader:
            reconstruction.zero_grad()
            nn.functional.mse_loss(decoder(encoder(batch)), batch).backward()
            reconstruction.step()

    codebooks = _seed_codebooks(torch, nn, encoder, data, spec)
    optimiser = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters(), *codebooks.parameters()],
        lr=1e-3,
    )
    for _ in range(spec.epochs - warmup):
        for (batch,) in loader:
            optimiser.zero_grad()
            latent = encoder(batch)
            quantised, commitment = _quantise(torch, latent, codebooks)
            # Straight-through: the decoder reads the code, the encoder gets the
            # gradient, and the lookup in between stays differentiable.
            passthrough = latent + (quantised - latent).detach()
            loss = (
                nn.functional.mse_loss(decoder(passthrough), batch)
                + nn.functional.mse_loss(quantised, latent.detach())
                + COMMITMENT_WEIGHT * commitment
            )
            loss.backward()
            optimiser.step()
    return _harvest(torch, encoder, codebooks, spec)


def _seed_codebooks(
    torch_module: object,
    nn: object,
    encoder: object,
    data: object,
    spec: RqVaeSpec,
) -> object:
    """Seed each stage's codebook by k-means on the warmed-up latent residuals."""
    torch = torch_module
    with torch.no_grad():
        latents = encoder(data).numpy().astype(np.float64)
    seeded = fit_residual_kmeans(
        latents,
        levels=spec.levels,
        branching=spec.branching,
        seed=spec.seed,
    )
    return nn.ParameterList(
        [
            nn.Parameter(torch.tensor(book, dtype=torch.float32))
            for book in seeded.codebooks
        ],
    )


def _quantise(torch_module: object, latent: object, codebooks: object) -> tuple:
    """Quantise a latent batch stage by stage, returning it with its commitment."""
    torch = torch_module
    residual = latent
    quantised = torch.zeros_like(latent)
    commitment = torch.zeros((), dtype=latent.dtype)
    for codebook in codebooks:
        distances = torch.cdist(residual, codebook)
        chosen = codebook[distances.argmin(dim=1)]
        commitment = commitment + ((residual - chosen.detach()) ** 2).mean()
        quantised = quantised + chosen
        residual = residual - chosen.detach()
    return quantised, commitment


def _harvest(
    torch_module: object,
    encoder: object,
    codebooks: object,
    spec: RqVaeSpec,
) -> RqVaeEncoder:
    """Lift the trained weights and codebooks out of torch and into numpy."""
    torch = torch_module
    from torch import nn  # noqa: PLC0415

    with torch.no_grad():
        weights = [
            (
                layer.weight.numpy().T.astype(np.float64),
                layer.bias.numpy().astype(np.float64),
            )
            for layer in encoder
            if isinstance(layer, nn.Linear)
        ]
        learned = [book.numpy().astype(np.float64) for book in codebooks]
    quantiser = ResidualKMeansEncoder(
        name=f"{spec.family}-codebooks-L{spec.levels}xK{spec.branching}",
        codebooks=learned,
        branching=spec.branching,
    )
    return RqVaeEncoder(
        name=f"{spec.family}-L{spec.levels}xK{spec.branching}",
        weights=weights,
        quantiser=quantiser,
    )
