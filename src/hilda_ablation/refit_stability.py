# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Decision executor for the preregistered HILDA refit-stability experiment.

The fitting/calibration layer may change as Experiment A is implemented, but the
scientific decision boundary is already frozen in ``REFIT_STABILITY_PROTOCOL.md``.
This module makes that boundary executable over replica evidence without treating
exchangeable cluster labels as stable identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Final, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REQUIRED_ARMS: Final = ("plain", "quasar", "shuffled", "random")


@dataclass(frozen=True)
class ReplicaEvidence:
    """Held-out evidence from one independently estimated replica."""

    arm: str
    replica: int
    seed: int
    labels: tuple[int, ...]
    calibration_digest: str
    hierarchy_digest: str
    recall: float
    candidate_fraction: float


@dataclass(frozen=True)
class DecisionConfig:
    """Frozen thresholds used to turn Experiment A evidence into a decision."""

    minimum_replicas: int = 8
    minimum_delta: float = 0.10
    recall_tolerance: float = 0.02
    candidate_fraction_tolerance: float = 0.01
    bootstrap_samples: int = 4000
    bootstrap_seed: int = 20260906


@dataclass(frozen=True)
class PairwiseAgreement:
    """Permutation-invariant agreement for one pair of independent replicas."""

    left: int
    right: int
    ari: float
    nmi: float


@dataclass(frozen=True)
class ExperimentDecision:
    """Machine-readable decision produced from preregistered Experiment A evidence."""

    decision: str
    delta_ari: float
    delta_ari_ci95: tuple[float, float]
    retrieval_matched: bool
    beats_shuffled: bool
    beats_random: bool
    replica_count: int
    pairwise: dict[str, tuple[PairwiseAgreement, ...]]
    evidence: tuple[ReplicaEvidence, ...]
    config: DecisionConfig

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable artifact payload."""
        return {
            "decision": self.decision,
            "delta_ari": self.delta_ari,
            "delta_ari_ci95": list(self.delta_ari_ci95),
            "retrieval_matched": self.retrieval_matched,
            "beats_shuffled": self.beats_shuffled,
            "beats_random": self.beats_random,
            "replica_count": self.replica_count,
            "pairwise": {
                arm: [asdict(item) for item in values]
                for arm, values in self.pairwise.items()
            },
            "evidence": [asdict(item) for item in self.evidence],
            "config": asdict(self.config),
        }


def evaluate_refit_stability(
    evidence: Sequence[ReplicaEvidence],
    config: DecisionConfig | None = None,
) -> ExperimentDecision:
    """Apply the frozen Experiment A rule to independent replica evidence.

    Labels are compared only through ARI/NMI, so learned k-means integer labels
    may permute freely between refits. The executor also fails closed when the
    arms are not measured at matched retrieval utility or when quasar replicas
    reuse one calibration digest.
    """
    active = config or DecisionConfig()
    grouped = _validate_and_group(evidence)
    pairwise = {arm: _pairwise(values) for arm, values in grouped.items()}

    quasar = np.asarray([item.ari for item in pairwise["quasar"]], dtype=float)
    plain = np.asarray([item.ari for item in pairwise["plain"]], dtype=float)
    shuffled = np.asarray([item.ari for item in pairwise["shuffled"]], dtype=float)
    random = np.asarray([item.ari for item in pairwise["random"]], dtype=float)

    delta = quasar - plain
    delta_ari = float(delta.mean())
    ci95 = _replica_bootstrap_ci(pairwise, active)
    retrieval_matched = _retrieval_is_matched(grouped, active)
    beats_shuffled = float(quasar.mean()) > float(shuffled.mean())
    beats_random = float(quasar.mean()) > float(random.mean())
    replica_count = len(grouped["quasar"])

    decision = _decide(
        replica_count=replica_count,
        delta_ari=delta_ari,
        ci95=ci95,
        retrieval_matched=retrieval_matched,
        beats_shuffled=beats_shuffled,
        beats_random=beats_random,
        config=active,
    )
    return ExperimentDecision(
        decision=decision,
        delta_ari=delta_ari,
        delta_ari_ci95=ci95,
        retrieval_matched=retrieval_matched,
        beats_shuffled=beats_shuffled,
        beats_random=beats_random,
        replica_count=replica_count,
        pairwise=pairwise,
        evidence=tuple(evidence),
        config=active,
    )


def _validate_and_group(
    evidence: Sequence[ReplicaEvidence],
) -> dict[str, tuple[ReplicaEvidence, ...]]:
    """Validate arm symmetry, held-out cardinality and independent calibrations."""
    grouped: dict[str, list[ReplicaEvidence]] = {arm: [] for arm in REQUIRED_ARMS}
    for item in evidence:
        if item.arm not in grouped:
            msg = f"unknown arm {item.arm!r}; expected one of {REQUIRED_ARMS}"
            raise ValueError(msg)
        grouped[item.arm].append(item)

    replica_sets = {arm: {item.replica for item in values} for arm, values in grouped.items()}
    if not replica_sets["quasar"]:
        msg = "at least one replica is required for every arm"
        raise ValueError(msg)
    expected = replica_sets["quasar"]
    if any(replicas != expected for replicas in replica_sets.values()):
        msg = f"all arms must contain identical replica ids, got {replica_sets}"
        raise ValueError(msg)

    ordered = {
        arm: tuple(sorted(values, key=lambda item: item.replica))
        for arm, values in grouped.items()
    }
    label_lengths = {len(item.labels) for values in ordered.values() for item in values}
    if len(label_lengths) != 1:
        msg = f"all replicas must label the same held-out objects, got {label_lengths}"
        raise ValueError(msg)

    quasar_digests = [item.calibration_digest for item in ordered["quasar"]]
    if len(set(quasar_digests)) != len(quasar_digests):
        msg = "quasar replicas must re-estimate calibration independently"
        raise ValueError(msg)
    return ordered


def _pairwise(values: Sequence[ReplicaEvidence]) -> tuple[PairwiseAgreement, ...]:
    """Measure every replica pair using label-permutation-invariant statistics."""
    rows: list[PairwiseAgreement] = []
    for left, right in combinations(values, 2):
        rows.append(
            PairwiseAgreement(
                left=left.replica,
                right=right.replica,
                ari=float(adjusted_rand_score(left.labels, right.labels)),
                nmi=float(normalized_mutual_info_score(left.labels, right.labels)),
            )
        )
    if not rows:
        msg = "at least two replicas are required to estimate stability"
        raise ValueError(msg)
    return tuple(rows)


def _retrieval_is_matched(
    grouped: dict[str, tuple[ReplicaEvidence, ...]],
    config: DecisionConfig,
) -> bool:
    """Require every control to operate near the quasar arm's retrieval utility."""
    reference_recall = float(np.mean([item.recall for item in grouped["quasar"]]))
    reference_fraction = float(
        np.mean([item.candidate_fraction for item in grouped["quasar"]])
    )
    for arm in ("plain", "shuffled", "random"):
        recall = float(np.mean([item.recall for item in grouped[arm]]))
        fraction = float(np.mean([item.candidate_fraction for item in grouped[arm]]))
        if abs(recall - reference_recall) > config.recall_tolerance:
            return False
        if abs(fraction - reference_fraction) > config.candidate_fraction_tolerance:
            return False
    return True


def _replica_bootstrap_ci(
    pairwise: dict[str, tuple[PairwiseAgreement, ...]],
    config: DecisionConfig,
) -> tuple[float, float]:
    """Cluster-bootstrap the ARI delta by resampling replica identities.

    Duplicate draws of one replica are not treated as a new independent pair;
    they are skipped. This preserves the protocol's replica-level uncertainty
    boundary instead of pretending held-out objects are independent experiments.
    """
    lookup: dict[str, dict[tuple[int, int], float]] = {}
    replicas = sorted({item.left for item in pairwise["quasar"]} | {item.right for item in pairwise["quasar"]})
    for arm, rows in pairwise.items():
        lookup[arm] = {(item.left, item.right): item.ari for item in rows}

    rng = np.random.default_rng(config.bootstrap_seed)
    draws: list[float] = []
    for _ in range(config.bootstrap_samples):
        sampled = rng.choice(replicas, size=len(replicas), replace=True)
        deltas: list[float] = []
        for left_pos, right_pos in combinations(range(len(sampled)), 2):
            left = int(sampled[left_pos])
            right = int(sampled[right_pos])
            if left == right:
                continue
            key = (min(left, right), max(left, right))
            deltas.append(lookup["quasar"][key] - lookup["plain"][key])
        if deltas:
            draws.append(float(np.mean(deltas)))
    if not draws:
        msg = "bootstrap produced no independent replica pairs"
        raise ValueError(msg)
    low, high = np.quantile(np.asarray(draws), [0.025, 0.975])
    return float(low), float(high)


def _decide(
    *,
    replica_count: int,
    delta_ari: float,
    ci95: tuple[float, float],
    retrieval_matched: bool,
    beats_shuffled: bool,
    beats_random: bool,
    config: DecisionConfig,
) -> str:
    """Apply the preregistered continue/kill/inconclusive rule literally."""
    if replica_count < config.minimum_replicas or not retrieval_matched:
        return "inconclusive"
    if delta_ari < config.minimum_delta or not beats_shuffled or not beats_random:
        return "kill"
    if ci95[0] <= 0.0:
        return "inconclusive"
    return "continue"
