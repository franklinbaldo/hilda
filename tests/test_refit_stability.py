# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Regression tests for the preregistered refit-stability decision boundary."""

from __future__ import annotations

import numpy as np
import pytest

from hilda_ablation.refit_stability import (
    DecisionConfig,
    ReplicaEvidence,
    evaluate_refit_stability,
)

N_REPLICAS = 8
N_OBJECTS = 40
RECALL = 0.80
CANDIDATE_FRACTION = 0.25
MINIMUM_DELTA = DecisionConfig().minimum_delta
ARMS = ("plain", "quasar", "shuffled", "random")


def _labels(arm: str, replica: int, *, stable_quasar: bool) -> tuple[int, ...]:
    """Build deterministic partitions.

    The quasar arm can be stable while the controls are deliberately unstable.
    """
    base = np.repeat(np.arange(4), N_OBJECTS // 4)
    if arm == "quasar" and stable_quasar:
        return tuple(int(value) for value in base)
    if arm == "quasar":
        shift = replica
    elif arm == "plain":
        shift = replica * 2
    elif arm == "shuffled":
        shift = replica * 3
    else:
        shift = replica * 4
    return tuple(int(value) for value in np.roll(base, shift))


def _evidence(*, stable_quasar: bool = True) -> list[ReplicaEvidence]:
    """Build all four preregistered arms at matched retrieval utility."""
    return [
        ReplicaEvidence(
            arm=arm,
            replica=replica,
            seed=1000 + replica,
            labels=_labels(arm, replica, stable_quasar=stable_quasar),
            calibration_digest=f"{arm}-calibration-{replica}",
            hierarchy_digest=f"{arm}-hierarchy-{replica}",
            recall=RECALL,
            candidate_fraction=CANDIDATE_FRACTION,
        )
        for arm in ARMS
        for replica in range(N_REPLICAS)
    ]


def test_stable_quasar_partition_continues() -> None:
    """A stable quasar arm beats independently unstable controls and continues."""
    result = evaluate_refit_stability(
        _evidence(),
        DecisionConfig(bootstrap_samples=500),
    )

    assert result.decision == "continue"
    assert result.delta_ari >= MINIMUM_DELTA
    assert result.delta_ari_ci95[0] > 0
    assert result.retrieval_matched
    assert result.beats_shuffled
    assert result.beats_random


def test_no_quasar_advantage_kills() -> None:
    """A quasar arm without the preregistered ARI gain cannot be rescued."""
    evidence = _evidence(stable_quasar=False)
    plain = {item.replica: item.labels for item in evidence if item.arm == "plain"}
    evidence = [
        ReplicaEvidence(
            arm=item.arm,
            replica=item.replica,
            seed=item.seed,
            labels=plain[item.replica] if item.arm == "quasar" else item.labels,
            calibration_digest=item.calibration_digest,
            hierarchy_digest=item.hierarchy_digest,
            recall=item.recall,
            candidate_fraction=item.candidate_fraction,
        )
        for item in evidence
    ]

    result = evaluate_refit_stability(
        evidence,
        DecisionConfig(bootstrap_samples=200),
    )

    assert result.decision == "kill"
    assert result.delta_ari == pytest.approx(0.0)


def test_unmatched_retrieval_is_inconclusive() -> None:
    """Stability cannot be claimed when the retrieval operating points differ."""
    evidence = [
        ReplicaEvidence(
            arm=item.arm,
            replica=item.replica,
            seed=item.seed,
            labels=item.labels,
            calibration_digest=item.calibration_digest,
            hierarchy_digest=item.hierarchy_digest,
            recall=0.50 if item.arm == "plain" else item.recall,
            candidate_fraction=item.candidate_fraction,
        )
        for item in _evidence()
    ]

    result = evaluate_refit_stability(
        evidence,
        DecisionConfig(bootstrap_samples=200),
    )

    assert result.decision == "inconclusive"
    assert not result.retrieval_matched


def test_quasar_calibration_must_be_independent() -> None:
    """Reusing one fixed quasar calibration across replicas fails closed."""
    evidence = [
        ReplicaEvidence(
            arm=item.arm,
            replica=item.replica,
            seed=item.seed,
            labels=item.labels,
            calibration_digest=(
                "fixed-calibration" if item.arm == "quasar" else item.calibration_digest
            ),
            hierarchy_digest=item.hierarchy_digest,
            recall=item.recall,
            candidate_fraction=item.candidate_fraction,
        )
        for item in _evidence()
    ]

    with pytest.raises(ValueError, match="re-estimate calibration independently"):
        evaluate_refit_stability(evidence, DecisionConfig(bootstrap_samples=50))
