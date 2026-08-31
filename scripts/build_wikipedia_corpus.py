# Copyright (c) 2026 Franklin Baldo. See LICENSE.
# /// script
# requires-python = ">=3.11"
# dependencies = ["hilda-ablation", "sentence-transformers", "torch", "datasets"]
#
# [tool.uv.sources]
# hilda-ablation = { path = "..", editable = true }
# ///
"""Embed a Wikipedia corpus large enough for a scale ladder.

20 newsgroups tops out near 18,000 usable documents, which is too small to ask
how the candidate budget grows. This streams English Wikipedia, keeps the lead
paragraphs, and embeds them once so every rung of the ladder is a nested
subsample of one distribution.

Usage:
    uv run scripts/build_wikipedia_corpus.py --size 300000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("wikipedia_corpus")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CHARS, MAX_CHARS = 300, 1200
BATCH = 25_000
"""Documents embedded and flushed at a time, so a long run keeps its progress."""


def _texts(size: int) -> list[str]:
    """Stream lead paragraphs until `size` of them are long enough to embed."""
    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    kept: list[str] = []
    for row in stream:
        lead = row["text"].strip().split("\n\n")[0]
        if len(lead) < MIN_CHARS:
            continue
        kept.append(lead[:MAX_CHARS])
        if len(kept) >= size:
            break
        if len(kept) % 50_000 == 0:
            logger.info("collected %d texts", len(kept))
    return kept


def main(argv: list[str] | None = None) -> int:
    """Build and cache the embedded corpus."""
    parser = argparse.ArgumentParser(description="Embed Wikipedia for the scale ladder")
    parser.add_argument("--size", type=int, default=300_000)
    parser.add_argument("--out", type=Path, default=Path("data/wikipedia.npy"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("streaming %d documents", args.size)
    texts = _texts(args.size)
    logger.info("embedding %d documents with %s", len(texts), MODEL)
    model = SentenceTransformer(MODEL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH):
        chunk = model.encode(
            texts[start : start + BATCH], batch_size=256, show_progress_bar=False
        )
        chunks.append(np.asarray(chunk, dtype=np.float32))
        done = sum(len(c) for c in chunks)
        np.save(args.out, np.concatenate(chunks))
        logger.info("embedded %d/%d", done, len(texts))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
