#!/usr/bin/env python3
"""
Build a 4,096-word HILDA vocabulary from wordfreq + Google embeddings.

- Streams top-N English words from wordfreq (already frequency-ordered).
- Filters by length/Zipf & alphabetic forms.
- Embeds with Google "text-embedding-004" via google-genai.
- Greedy max–min (cosine) selection to 4,096 words.
- Optional near-duplicate suppression by edit distance.

Outputs: hilda_codebook_4096.csv
"""

import os, math, csv, sys
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

# --- Config ------------------------------------------------------------------

TARGET_K = 4096
TOP_N = 120_000          # how many candidates to consider from wordfreq (tune 60k–200k)
MIN_ZIPF = 3.0           # keep not-too-rare (≈ ≥1 per million)
MAX_ZIPF = 6.5           # drop function words at extreme frequency (optional)
LENGTH_RANGE = (3, 10)   # length filter for speakability
BATCH = 512              # embedding batch size
EMBED_DIM_FALLBACK = 768 # used to pre-alloc if you want, not required
USE_RAPIDFUZZ = True     # requires pip install rapidfuzz

# --- 1) Candidates from wordfreq --------------------------------------------

from wordfreq import top_n_list, zipf_frequency  # docs: iter/top_n & Zipf scale
# See: wordfreq Zipf docs. Zipf=log10(freq per billion). 6 ≈ 1/1k, 3 ≈ 1/1e6.

def stream_candidates(n=TOP_N) -> List[Tuple[str, float]]:
    words = top_n_list("en", n)  # already in descending frequency order.
    kept = []
    lo, hi = LENGTH_RANGE
    for w in words:
        if not w.isalpha() or not w.islower():
            continue
        z = zipf_frequency(w, "en")
        if z < MIN_ZIPF or z > MAX_ZIPF:
            continue
        if not (lo <= len(w) <= hi):
            continue
        kept.append((w, z))
    return kept

# --- 2) Google embeddings (Gemini / Vertex via google-genai) -----------------

# Docs: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
# and model reference (text-embedding-004).
# Gemini API docs: https://ai.google.dev/gemini-api/docs/embeddings

import google.genai as genai

def embed_google_texts(texts: List[str]) -> np.ndarray:
    """
    Returns float32 ndarray [len(texts), D].
    Uses model="text-embedding-004".
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY env var.")
    genai.configure(api_key=api_key)
    model = "models/text-embedding-004"

    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding (Google)"):
        batch = texts[i:i+BATCH]
        resp = genai.embed_content(
            model=model,
            content=batch,
            task_type="RETRIEVAL_DOCUMENT"
        )
        # google-genai returns list[Embedding] under resp.embeddings
        vecs = [np.array(e, dtype=np.float32) for e in resp['embedding']]
        out.append(np.vstack(vecs))
    X = np.vstack(out)
    # L2 normalize for cosine distance
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X

# --- 3) (Optional) near-duplicate suppression --------------------------------

if USE_RAPIDFUZZ:
    from rapidfuzz.distance import Levenshtein  # fast C-ext
else:
    Levenshtein = None

def too_similar(a: str, b: str) -> bool:
    if not Levenshtein:
        return False
    # quick block: edit distance < 3 is often a near-miss (s/t typo, plural)
    return Levenshtein.distance(a, b) < 3

# --- 4) Greedy max–min (Gonzalez) -------------------------------------------

def farthest_point_sampling(words: List[str], X: np.ndarray, K: int = TARGET_K) -> List[int]:
    """
    Greedy max–min on cosine distance (1 - dot) using normalized X.
    O(K*N) with vectorized updates; good up to ~200k with float32 & batches.
    """
    N = X.shape[0]
    # Seed: word most aligned with the centroid (central word)
    centroid = X.mean(axis=0, keepdims=True)
    centroid /= np.linalg.norm(centroid) + 1e-9
    seed = int(np.argmax(X @ centroid.T))

    selected = [seed]
    in_pool = np.ones(N, dtype=bool)
    in_pool[seed] = False

    dmin = 1.0 - (X @ X[seed])  # cosine dist to seed

    # Convenience: track for dup suppression
    for _ in tqdm(range(1, K), desc="Greedy max–min"):
        # ignore already picked
        dmin[~in_pool] = -1.0
        nxt = int(np.argmax(dmin))

        # near-duplicate suppression (optional)
        if Levenshtein:
            # If chosen too close to any selected word by edit distance, zero out and re-pick.
            skip = any(too_similar(words[nxt], words[j]) for j in selected)
            if skip:
                in_pool[nxt] = False
                continue

        selected.append(nxt)
        in_pool[nxt] = False

        # update dmin: dist = 1 - dot
        add = 1.0 - (X @ X[nxt])
        dmin = np.minimum(dmin, add)

        if len(selected) % 256 == 0:
            # preventive: free any -1 padding and avoid NaN drift
            dmin = np.maximum(dmin, -1.0)

        if len(selected) >= K:
            break

    return selected

# --- 5) Main -----------------------------------------------------------------

def main():
    print("Collecting candidates from wordfreq…")
    cand = stream_candidates(TOP_N)
    words = [w for w, z in cand]
    zipfs = np.array([z for w, z in cand], dtype=np.float32)
    print(f"Candidates kept: {len(words):,}")

    print("Embedding with Google text-embedding-004…")
    X = embed_google_texts(words)  # normalized

    print("Selecting 4,096 diverse words (greedy max–min)…")
    idx = farthest_point_sampling(words, X, TARGET_K)
    idx = idx[:TARGET_K]
    picked = [(i, words[i], float(zipfs[i])) for i in idx]

    out = "hilda_codebook_4096.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "word", "zipf", "rank_in_candidates"])
        for rank, (i, word, z) in enumerate(picked):
            w.writerow([rank, word, f"{z:.2f}", i])
    print(f"Wrote {out} with {len(picked)} rows.")

if __name__ == "__main__":
    main()
