# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Command line entry point for the ablation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hilda_ablation.corpus import load_corpus
from hilda_ablation.runner import RosterConfig, SweepGrid, run_ablation

logger = logging.getLogger("hilda_ablation")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ablation's command line."""
    parser = argparse.ArgumentParser(
        description="Run the HILDA representation ablation",
    )
    parser.add_argument("--corpus-size", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/ablation.csv"))
    parser.add_argument("--no-rqvae", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ablation from the command line."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = load_corpus(
        size=args.corpus_size,
        n_queries=args.queries,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    result = run_ablation(
        corpus,
        roster=RosterConfig(seed=args.seed, include_rqvae=not args.no_rqvae),
        grid=SweepGrid(k=args.k),
    )
    result.write_csv(args.out)
    result.write_notes(args.out.with_suffix(".json"))
    for name, value in result.notes.items():
        logger.info("%s = %.4f", name, value)
    logger.info("wrote %d operating points to %s", len(result.points), args.out)
    return 0
