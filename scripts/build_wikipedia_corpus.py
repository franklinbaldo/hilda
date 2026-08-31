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
how the candidate budget grows. This streams English Wikipedia, keeps only one
text batch in memory, and writes embeddings directly into a preallocated NPY
memmap. Every rung of the ladder can therefore be a nested prefix of one fixed
distribution without the corpus builder itself becoming an out-of-memory test.

Reserve held-out queries by asking for ``max_corpus + n_queries`` rows. The
scaling runner uses the final rows as queries, outside every corpus prefix.

Usage:
    uv run scripts/build_wikipedia_corpus.py --size 300200
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("wikipedia_corpus")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CHARS, MAX_CHARS = 300, 1200
BATCH = 5_000
"""Maximum number of raw texts and fresh embeddings resident at once."""


def _text_batches(size: int, batch_size: int) -> Iterator[list[str]]:
    """Yield eligible lead paragraphs in bounded batches until ``size`` rows."""
    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    batch: list[str] = []
    kept = 0
    for row in stream:
        lead = row["text"].strip().split("\n\n")[0]
        if len(lead) < MIN_CHARS:
            continue
        batch.append(lead[:MAX_CHARS])
        kept += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
        if kept >= size:
            break
    if batch:
        yield batch


def _dimension(model: SentenceTransformer) -> int:
    """Return the model's embedding width, rejecting an unknown dimension."""
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        message = f"{MODEL} did not report an embedding dimension"
        raise RuntimeError(message)
    return int(dimension)


def main(argv: list[str] | None = None) -> int:
    """Stream, embed, and cache the corpus with bounded working memory."""
    parser = argparse.ArgumentParser(description="Embed Wikipedia for the scale ladder")
    parser.add_argument("--size", type=int, default=300_200)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--out", type=Path, default=Path("data/wikipedia.npy"))
    args = parser.parse_args(argv)
    if args.size <= 0 or args.batch_size <= 0:
        parser.error("--size and --batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("streaming and embedding %d documents with %s", args.size, MODEL)
    model = SentenceTransformer(MODEL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        args.out,
        mode="w+",
        dtype=np.float32,
        shape=(args.size, _dimension(model)),
    )

    done = 0
    for texts in _text_batches(args.size, args.batch_size):
        chunk = np.asarray(
            model.encode(texts, batch_size=256, show_progress_bar=False),
            dtype=np.float32,
        )
        stop = done + len(chunk)
        matrix[done:stop] = chunk
        matrix.flush()
        done = stop
        logger.info("embedded %d/%d", done, args.size)

    if done != args.size:
        message = f"stream ended after {done} eligible documents; expected {args.size}"
        raise RuntimeError(message)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
