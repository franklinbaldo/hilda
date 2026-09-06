# ruff: noqa: INP001
# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Apply the frozen Experiment A decision rule to replica evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hilda_ablation.refit_stability import (
    DecisionConfig,
    ReplicaEvidence,
    evaluate_refit_stability,
)


def _load(path: Path) -> tuple[list[ReplicaEvidence], DecisionConfig]:
    """Load replica evidence and optional frozen configuration from JSON."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    evidence = [
        ReplicaEvidence(
            arm=item["arm"],
            replica=int(item["replica"]),
            seed=int(item["seed"]),
            labels=tuple(int(value) for value in item["labels"]),
            calibration_digest=item["calibration_digest"],
            hierarchy_digest=item["hierarchy_digest"],
            recall=float(item["recall"]),
            candidate_fraction=float(item["candidate_fraction"]),
        )
        for item in payload["evidence"]
    ]
    config = DecisionConfig(**payload.get("config", {}))
    return evidence, config


def main() -> None:
    """Read replica evidence, evaluate it, and write the machine-readable result."""
    parser = argparse.ArgumentParser(
        description="Evaluate HILDA refit-stability Experiment A evidence."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON file containing replica evidence",
    )
    parser.add_argument("output", type=Path, help="Destination JSON artifact")
    args = parser.parse_args()

    evidence, config = _load(args.input)
    result = evaluate_refit_stability(evidence, config)
    args.output.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
