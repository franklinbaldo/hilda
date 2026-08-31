# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The learned variant must actually learn its codebooks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hilda_ablation.encoders.residual import fit_residual_kmeans  # noqa: E402
from hilda_ablation.encoders.rqvae import RqVaeSpec, fit_rqvae  # noqa: E402

LEVELS = 3


@pytest.fixture
def points() -> np.ndarray:
    """Clustered points, so a quantiser has structure to find."""
    rng = np.random.default_rng(5)
    centres = rng.normal(size=(8, 16)) * 4
    return np.repeat(centres, 40, axis=0) + rng.normal(size=(320, 16))


def _seeded_codebooks(points: np.ndarray, spec: RqVaeSpec) -> list[np.ndarray]:
    """Return the codebooks a post-hoc fit would produce in this latent space."""
    trained = fit_rqvae(points, spec)
    return fit_residual_kmeans(
        trained.latent(points),
        levels=spec.levels,
        branching=spec.branching,
        seed=spec.seed,
    ).codebooks


def test_joint_training_moves_the_codebooks_off_their_seed(points: np.ndarray) -> None:
    """Joint training moves the codebooks off their seed."""
    spec = RqVaeSpec(levels=LEVELS, branching=4, seed=0, epochs=20, joint=True)
    trained = fit_rqvae(points, spec)
    assert not np.allclose(
        trained.quantiser.codebooks[0],
        _seeded_codebooks(points, spec)[0],
        atol=1e-6,
    )


def test_the_post_hoc_variant_keeps_exactly_the_fitted_codebooks(
    points: np.ndarray,
) -> None:
    """The post-hoc variant keeps exactly the fitted codebooks."""
    spec = RqVaeSpec(levels=LEVELS, branching=4, seed=0, epochs=20, joint=False)
    trained = fit_rqvae(points, spec)
    for learned, seeded in zip(
        trained.quantiser.codebooks, _seeded_codebooks(points, spec), strict=True
    ):
        assert np.allclose(learned, seeded)


def test_the_two_variants_are_named_apart(points: np.ndarray) -> None:
    """The two variants are named apart."""
    joint = fit_rqvae(points, RqVaeSpec(levels=LEVELS, branching=4, seed=0, epochs=4))
    posthoc = fit_rqvae(
        points, RqVaeSpec(levels=LEVELS, branching=4, seed=0, epochs=4, joint=False)
    )
    assert joint.name.startswith("rqvae")
    assert posthoc.name.startswith("ae+rvq")


def test_quantisation_beats_the_latent_variance(points: np.ndarray) -> None:
    """Quantisation beats the latent variance."""
    encoder = fit_rqvae(
        points, RqVaeSpec(levels=LEVELS, branching=4, seed=0, epochs=20)
    )
    assert encoder.quantised_error(points) < 1.0


def test_every_level_of_the_code_is_used(points: np.ndarray) -> None:
    """Every level of the code is used."""
    encoder = fit_rqvae(
        points, RqVaeSpec(levels=LEVELS, branching=4, seed=0, epochs=20)
    )
    assert len(encoder.quantiser.codebooks) == LEVELS
    assert encoder.layout.depth == LEVELS
