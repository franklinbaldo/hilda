# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Corpus loading and embedding, cached so a run is reproducible and cheap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_NAME = "20newsgroups"
MIN_DOCUMENT_CHARS = 200
"""Below this a newsgroup post is a signature or a one-liner, not a topic."""


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be built at all."""

    def __init__(self, name: str, reason: str) -> None:
        """Record the structured detail the caller needs."""
        message = f"cannot build corpus {name!r}: {reason}"
        super().__init__(message)
        self.name = name
        self.reason = reason


@dataclass(frozen=True)
class Corpus:
    """Embeddings split into an indexed corpus and held-out queries."""

    name: str
    documents: np.ndarray
    queries: np.ndarray

    @property
    def dims(self) -> int:
        """Return the width of the projected space."""
        return self.documents.shape[1]


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    """Embed a batch of documents with the pinned sentence encoder."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        reason = "sentence-transformers is not installed"
        raise CorpusError(CORPUS_NAME, reason) from exc
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, batch_size=128, show_progress_bar=False))


def _load_texts(size: int, seed: int) -> list[str]:
    """Sample documents long enough to carry a topic."""
    from sklearn.datasets import fetch_20newsgroups  # noqa: PLC0415

    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=seed,
    )
    texts = [t.strip() for t in data.data if len(t.strip()) > MIN_DOCUMENT_CHARS]
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(texts))[:size]
    return [texts[i][:2000] for i in chosen]


def load_corpus(
    size: int,
    n_queries: int,
    seed: int,
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
) -> Corpus:
    """Embed 20 newsgroups once, then reuse the cache across runs."""
    cache = cache_dir / f"20news-{size}-{n_queries}-{seed}.npz"
    if cache.exists():
        stored = np.load(cache)
        return Corpus(CORPUS_NAME, stored["documents"], stored["queries"])
    total = size + n_queries
    texts = _load_texts(total, seed)
    if len(texts) < total:
        reason = f"only {len(texts)} usable documents"
        raise CorpusError(CORPUS_NAME, reason)
    logger.info("embedding %d documents with %s", total, model_name)
    embeddings = _embed(texts, model_name).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    documents, queries = embeddings[:size], embeddings[size:]
    np.savez_compressed(cache, documents=documents, queries=queries)
    return Corpus(CORPUS_NAME, documents, queries)
