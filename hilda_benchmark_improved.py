
import numpy as np
from time import perf_counter
from collections import defaultdict
from typing import Tuple, Dict, List

# -----------------------------
# Hilbert curve (2D), symmetric i<->(x,y) without gray-code mismatch
# -----------------------------

def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int,int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def d2xy(n: int, d: int) -> Tuple[int,int]:
    """Hilbert distance d -> (x,y), where n = 2**b is grid size per axis."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def xy2d(n: int, x: int, y: int) -> int:
    """(x,y) -> Hilbert distance d."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s //= 2
    return d

# For convenience
def xy2i(n: int, x: int, y: int) -> int:
    return xy2d(n, x, y)

def i2xy(n: int, i: int) -> Tuple[int,int]:
    return d2xy(n, i)

# -----------------------------
# Data generation (clustered unit vectors)
# -----------------------------

def generate_clustered_embeddings(N=2000, dim=100, num_clusters=20, seed=42):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(num_clusters, dim))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    sizes = [N // num_clusters] * num_clusters
    sizes[-1] += (N % num_clusters)
    Xs = []
    for c, size in enumerate(sizes):
        noise = 0.02 * rng.normal(size=(size, dim))
        Xc = means[c] + noise
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9
        Xs.append(Xc)
    X = np.vstack(Xs)
    # shuffle
    idx = rng.permutation(N)
    return X[idx]

# -----------------------------
# PCA2 anchored on codebook
# -----------------------------

def fit_pca2(X: np.ndarray):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T  # D x 2
    Y = Xc @ W    # N x 2
    mins = Y.min(axis=0)
    maxs = Y.max(axis=0)
    return mu.astype(np.float32), W.astype(np.float32), mins.astype(np.float32), maxs.astype(np.float32)

def project01(v: np.ndarray, mu: np.ndarray, W: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    y = (v - mu) @ W  # 1x2
    z = (y - mins) / (np.maximum(maxs - mins, 1e-9))
    return np.clip(z, 0.0, 1.0)

# -----------------------------
# HILDA packing (only need Hilbert index for prefixing)
# -----------------------------

def quantize_hilbert(z01: np.ndarray, bits_per_axis: int):
    """Return (x,y,h) for 2D z in [0,1]^2 at given bits per axis."""
    n = 1 << bits_per_axis
    maxcoord = n - 1
    x = int(np.round(z01[0,0] * maxcoord))
    y = int(np.round(z01[0,1] * maxcoord))
    x = min(max(x, 0), maxcoord)
    y = min(max(y, 0), maxcoord)
    h = xy2i(n, x, y)  # 2*b bits implicit range
    return x, y, h

# -----------------------------
# Benchmark
# -----------------------------

def benchmark(N=2000, k=10, dim=100, num_clusters=20, codebook_size=4096,
              bits_per_axis=10, prefix_bits=12, seed=7, widen_targets=(64, 256)):
    """
    Compare:
      - Baseline: cosine with all N items
      - HILDA: range scan by prefix then cosine only over candidates (with optional prefix widening)
    """
    rng = np.random.default_rng(seed)

    # Fit PCA2 on a codebook (could be separate; here reuse clustered)
    codebook = generate_clustered_embeddings(codebook_size, dim, num_clusters=min(num_clusters*2, max(2, codebook_size//50)), seed=seed)
    mu, W, mins, maxs = fit_pca2(codebook)

    # Corpus
    X = generate_clustered_embeddings(N, dim, num_clusters=num_clusters, seed=seed+1).astype(np.float32)

    # Precompute normalized vectors for cosine
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # Compute Hilbert indices and bucket by prefix
    n = 1 << bits_per_axis
    total_h_bits = 2 * bits_per_axis
    shift = total_h_bits - prefix_bits
    hs = np.empty(N, dtype=np.int64)
    buckets = defaultdict(list)

    t0 = perf_counter()
    for i in range(N):
        z = project01(Xn[i:i+1], mu, W, mins, maxs)
        _, _, h = quantize_hilbert(z, bits_per_axis)
        hs[i] = h
        buckets[h >> shift].append(i)
    t_bucket = perf_counter() - t0

    # Queries (take first Q examples)
    Q = min(100, N)  # number of queries
    total_recall = 0.0
    total_ms_hilda = 0.0
    total_ms_full = 0.0
    total_candidates = 0

    for q in range(Q):
        qv = Xn[q:q+1]
        # Baseline full cosine
        t1 = perf_counter()
        sims = (Xn @ qv.T).reshape(-1)  # cosine since normalized
        sims[q] = -1e9
        topk_full = np.argpartition(-sims, k)[:k]
        topk_full = topk_full[np.argsort(-sims[topk_full])]
        t_full = (perf_counter() - t1) * 1000.0

        # HILDA prefilter
        t2 = perf_counter()
        prefix = hs[q] >> shift
        candidates = buckets.get(prefix, [])
        # Widen if too few candidates
        if len(candidates) < widen_targets[0]:
            alt = []
            # linear probe nearby prefixes; wrap with mask
            mask = (1 << prefix_bits) - 1
            step = 1
            while len(candidates) + len(alt) < widen_targets[0] and step < (1 << (prefix_bits-1)):
                alt.extend(buckets.get((prefix + step) & mask, []))
                alt.extend(buckets.get((prefix - step) & mask, []))
                step += 1
            candidates = candidates + alt
        total_candidates += len(candidates)

        if len(candidates) == 0:
            recall = 0.0
        else:
            cand = np.array(candidates, dtype=np.int64)
            cs = (Xn[cand] @ qv.T).reshape(-1)
            # pick top-k within candidates
            take = min(k, len(cs))
            topk_c_idx = np.argpartition(-cs, take-1)[:take]
            topk_c = cand[topk_c_idx]
            # exact rerank among picked
            rerank = (Xn[topk_c] @ qv.T).reshape(-1)
            topk_c = topk_c[np.argsort(-rerank)]
            recall = len(set(topk_c[:k]) & set(topk_full)) / float(k)

        t_hilda = (perf_counter() - t2) * 1000.0

        total_recall += recall
        total_ms_hilda += t_hilda
        total_ms_full += t_full

    avg_recall = total_recall / Q
    avg_hilda = total_ms_hilda / Q
    avg_full = total_ms_full / Q
    avg_cands = total_candidates / Q

    # Locality correlation (quick rank-corr proxy)
    q0 = 0
    sims0 = (Xn @ Xn[q0:q0+1].T).reshape(-1)
    cosdist0 = 1.0 - sims0
    deltah0 = np.abs(hs - hs[q0])
    # Spearman via numpy rank-corr proxy
    r1 = (-cosdist0).argsort().argsort().astype(np.float32)  # higher sim => higher rank
    r2 = (-deltah0).argsort().argsort().astype(np.float32)
    r1 -= r1.mean(); r2 -= r2.mean()
    spearman = float((r1 @ r2) / (np.linalg.norm(r1)*np.linalg.norm(r2) + 1e-9))

    return {
        "N": N, "k": k, "bits_per_axis": bits_per_axis, "prefix_bits": prefix_bits,
        "avg_recall@k": round(avg_recall, 3),
        "avg_candidates": int(round(avg_cands)),
        "avg_ms_full": round(avg_full, 3),
        "avg_ms_hilda": round(avg_hilda, 3),
        "speedup_x": round((avg_full / max(avg_hilda,1e-6)), 2),
        "spearman_locality": round(spearman, 3),
        "bucket_build_ms": round(t_bucket*1000.0, 3),
        "queries": Q
    }

if __name__ == "__main__":
    res = benchmark(N=2000, k=10, bits_per_axis=10, prefix_bits=12)
    print(res)
