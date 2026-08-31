# HILDA representation ablation

HILDA mints a 128-bit identifier whose numeric order carries meaning, so that
a range scan on an ordinary B-tree can stand in for a vector index. The
architecture in `PLAN.md` gets there with PCA(2) plus a Hilbert curve. This
ablation asks whether that projection is the right mechanism, comparing
families at **matched prefix bits** and swapping only the representation.

The question it answers:

> Can a prefix-sortable semantic ID provide high-recall candidate generation
> using only ordinary ordered indexes, and which representation gets there?

## Results

Corpus: 8,000 documents from 20 newsgroups, embedded with
`all-MiniLM-L6-v2`; 200 held-out queries; ground truth is exact cosine top-10.
Every encoder is fitted, encoded and probed on the unit sphere, which is the
geometry the ground truth scores in. See [results/REPORT.md](results/REPORT.md)
for the generated table and [results/ablation.csv](results/ablation.csv) for
every operating point.

recall@10 while the mean per-query scan stays inside a budget, with the mean
number of separate B-tree ranges and the 95th-percentile per-query scan:

| encoder | mean scan ≤1% | mean scan ≤5% |
|---|---|---|
| `hkmeans-L4xK16` | 0.614 (7r · p95 0.8%) | 0.947 (49r · p95 5.3%) |
| `hkmeans-L4xK8` | 0.619 (7r · p95 0.8%) | 0.938 (47r · p95 5.6%) |
| `hkmeans-L6xK4` | 0.596 (6r · p95 1.1%) | 0.851 (23r · p95 3.8%) |
| `rvq-L4xK16` | 0.476 (11r · p95 1.5%) | 0.749 (5r · p95 5.7%) |
| `rqvae-L4xK8` | 0.321 (6r · p95 1.8%) | 0.714 (35r · p95 6.4%) |
| `rvq-L4xK8` | 0.453 (11r · p95 1.7%) | 0.709 (33r · p95 3.9%) |
| `ae+rvq-L4xK8` | 0.378 (12r · p95 1.3%) | 0.628 (36r · p95 3.1%) |
| `hilbert-pca4-b15` | 0.165 (8r · p95 1.9%) | 0.469 (4r · p95 7.0%) |
| `hilbert-pca2-b30` | 0.055 (9r · p95 1.2%) | 0.298 (6r · p95 7.2%) |
| `hilbert-pca2-b30-minmax` | 0.059 (3r · p95 1.6%) | 0.193 (6r · p95 5.9%) |
| `hilbert-rp2-b30` | 0.022±0.005 (3r · p95 1.0%) | 0.079±0.013 (2r · p95 3.6%) |

Six findings:

1. **Hierarchical k-means wins by a wide margin.** At a 1% mean scan it reaches
   11x the recall of the quantile-scaled PCA(2)+Hilbert baseline, and at 5%
   it reaches 0.947 against 0.298. Against `-minmax`, the scaling the
   repository's own benchmark uses, the gap at 5% is 0.947 against 0.193.
2. **Its scan cost is also better behaved per query.** At a 1% mean budget the
   tree's p95 is 0.8%, tighter than every SFC variant's (1.2% to 1.9%). The
   worry that unbalanced cells would flatter the tree runs the other way here.
3. **More projection dimensions help inside the SFC family, and never enough
   to matter.** PCA(4) beats PCA(3) beats PCA(2) at every budget, so
   "maximise the PCA" is directionally right and strategically irrelevant:
   PCA(4)+Hilbert still loses to every quantiser.
4. **PCA carries real signal.** Random projection at the same dimension
   reaches a third of PCA(2)'s recall, averaged over five seeds.
5. **The curve barely affects recall, and does affect seek count.** Hilbert and
   Morton tie to three decimal places at every budget, while Hilbert
   consolidates the same candidates into fewer ranges. Locality preservation
   buys range consolidation, not recall.
6. **Joint training recovers what freezing loses, and no more.** At the same
   shape, the RQ-VAE (0.714 at 5%) beats the post-hoc `ae+rvq` (0.628) and
   lands level with plain residual k-means (0.709). Training the codebooks
   with the autoencoder is worth doing and does not close the gap to the tree.

PCA(2) explains 7.2% of the embedding variance on this corpus, PCA(3) 9.6%,
PCA(4) 11.5%.

## What this does not show

- **No ANN reference row.** The comparison is against exact cosine, not
  against HNSW or IVF at matched cost. Recall here is candidate-generation
  recall, so it bounds what a re-rank stage can recover, and says nothing
  about end-to-end latency against a real vector index.
- **This is a small-prefix ablation, not a 60-bit comparison.** The sweep
  compares families at 4 to 20 matched prefix bits. The SFC encoders can
  address the full 60-bit field; the quantisers here spend 12 to 20 bits, and
  the tree stops splitting once a node holds fewer points than its branching
  factor, so at 8,000 documents `L5xK16` and `L4xK16` produce identical codes.
  The 5x12-bit shape that aligns with the 4,096-word codebook needs a corpus
  several orders of magnitude larger before it is a real configuration.
- **The scan budget binds on the mean.** Each cell reports the p95 alongside,
  but the operating point is still selected by mean scan fraction. A
  comparison at a matched per-query candidate budget would be stricter.
- **One corpus, one embedder, one k.** 20 newsgroups is topically coarse,
  which plausibly favours the tree.
- **Range counts are modelled, not measured.** `n_ranges` counts merged
  intervals over a sorted code column. It stands in for seeks; it is not a
  Postgres plan, and no latency was measured.
- **Normalisation changed nothing here.** `all-MiniLM-L6-v2` ends its pipeline
  with a normalisation module, so the cached embeddings already had unit norm
  to within 4.4e-08. Fitting on the unit sphere is still what the code does,
  rather than depending on a property of one model's pipeline.

## Running it

```bash
uv run scripts/run_ablation.py --corpus-size 8000 --queries 200
uv run scripts/report.py results/ablation.csv > results/REPORT.md
```

The first run downloads 20 newsgroups and the sentence encoder, then caches
the embeddings under `data/`. Later runs read the cache.

## How it is put together

Every encoder emits the same shape of code: a tuple of fixed-width digits,
packed most-significant level first, so a depth-r prefix is one contiguous
integer range. That is the only property an ordered index needs, and it is
what makes the families comparable.

| module | role |
|---|---|
| `codes.py` | digit layouts, prefix ranges, range merging |
| `curves.py` | Hilbert (Skilling's transpose, any dimension) and Morton |
| `geometry.py` | the unit sphere every stage agrees on |
| `projections.py` | PCA and random projection |
| `encoders/sfc.py` | quantile or min-max grid, plus a space-filling curve |
| `encoders/tree.py` | hierarchical k-means |
| `encoders/residual.py` | residual vector quantisation |
| `encoders/rqvae.py` | RQ-VAE, and the post-hoc `ae+rvq` to measure it against |
| `evaluation.py` | ground truth, range scans, the three costs |
| `runner.py` | the roster, swept by prefix bits |

The learned variant trains encoder, decoder and codebooks together, with a
straight-through estimator carrying the gradient through the lookup, after a
warm-up that seeds the codebooks by k-means. Freezing the autoencoder and
fitting codebooks afterwards is a different and weaker model; it ships as
`ae+rvq` so the difference is measured rather than assumed.

Tests cover the properties the whole scheme rests on: that a curve index is a
bijection, that successive Hilbert indices are grid neighbours, that a code
prefix addresses the enclosing coarse cell, that each encoder's first probe is
the query's own cell, and that joint training moves the codebooks off their
seed while the post-hoc variant keeps them exactly.

```bash
uv run pytest tests/
uv run ruff check src tests scripts
```
