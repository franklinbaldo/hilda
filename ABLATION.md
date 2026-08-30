# HILDA representation ablation

HILDA mints a 128-bit identifier whose numeric order carries meaning, so that
a range scan on an ordinary B-tree can stand in for a vector index. The
architecture in `PLAN.md` gets there with PCA(2) plus a Hilbert curve. This
ablation asks whether that projection is the right mechanism, by holding the
60-bit semantic field fixed and swapping only the representation.

The question it answers:

> Can a prefix-sortable semantic ID provide high-recall candidate generation
> using only ordinary ordered indexes, and which representation gets there?

## Results

Corpus: 8,000 documents from 20 newsgroups, embedded with
`all-MiniLM-L6-v2`; 200 held-out queries; ground truth is exact cosine top-10.
Every encoder spends the same 60 semantic bits. See
[results/REPORT.md](results/REPORT.md) for the generated table and
[results/ablation.csv](results/ablation.csv) for every operating point.

recall@10 within a scan budget, with the mean number of separate B-tree
ranges that scan takes:

| encoder | ≤1% scanned | ≤5% scanned |
|---|---|---|
| `hkmeans-L4xK8` | 0.603 (7r) | 0.931 (46r) |
| `hkmeans-L6xK4` | 0.589 (6r) | 0.850 (10r) |
| `rvq-L4xK8` | 0.453 (11r) | 0.709 (33r) |
| `rqvae-L4xK8` | 0.393 (11r) | 0.632 (35r) |
| `hilbert-pca4-b15` | 0.165 (8r) | 0.469 (4r) |
| `hilbert-pca2-b30` | 0.055 (9r) | 0.298 (6r) |
| `hilbert-rp2-b30` | 0.022±0.005 (3r) | 0.079±0.013 (2r) |

Five findings:

1. **Hierarchical k-means wins by a wide margin.** At a 1% scan budget it
   reaches 11x the recall of the current PCA(2)+Hilbert architecture, and at
   5% it reaches 0.931 against 0.298.
2. **More projection dimensions help inside the SFC family, and never enough
   to matter.** PCA(4) beats PCA(3) beats PCA(2) at every budget, so
   "maximise the PCA" is directionally right and strategically irrelevant:
   PCA(4)+Hilbert still loses to every quantiser.
3. **PCA carries real signal.** Random projection at the same dimension
   reaches a third of PCA(2)'s recall, averaged over five seeds. The
   projection is doing work; the ceiling is the mechanism around it.
4. **The curve barely affects recall, and does affect seek count.** Hilbert
   and Morton tie to three decimal places at every budget, while Hilbert
   consolidates the same candidates into fewer ranges. Locality preservation
   buys range consolidation, not recall.
5. **Learning the latent space did not pay here.** The RQ-VAE tracks plain
   residual k-means and both trail the tree. On this corpus the win comes
   from hierarchy, not from a learned encoder.

PCA(2) explains 7.2% of the embedding variance on this corpus, PCA(3) 9.6%,
PCA(4) 11.5%.

## What this does not show

- **No ANN reference row.** The comparison is against exact cosine, not
  against HNSW or IVF at matched cost. Recall here is candidate-generation
  recall, so it bounds what a re-rank stage can recover, and says nothing
  about end-to-end latency against a real vector index.
- **One corpus, one embedder, one k.** 20 newsgroups is topically coarse,
  which plausibly favours the tree.
- **Range counts are modelled, not measured.** `n_ranges` counts merged
  intervals over a sorted code column. It stands in for seeks; it is not a
  Postgres plan, and no latency was measured.
- **The quantiser shapes are small.** `L4xK8` and `L6xK4` spend 12 bits, not
  the full 60. The 5x12-bit shape that aligns with the 4,096-word codebook
  needs a corpus large enough to fit it.

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
| `projections.py` | PCA and random projection |
| `encoders/sfc.py` | quantile grid plus space-filling curve |
| `encoders/tree.py` | hierarchical k-means |
| `encoders/residual.py` | residual vector quantisation |
| `encoders/rqvae.py` | learned latent space with residual codebooks |
| `evaluation.py` | ground truth, range scans, the three costs |
| `runner.py` | the roster, swept by prefix bits |

Tests cover the properties the whole scheme rests on: that a curve index is a
bijection, that successive Hilbert indices are grid neighbours, that a code
prefix addresses the enclosing coarse cell, and that each encoder's first
probe is the query's own cell.

```bash
uv run pytest tests/
uv run ruff check src tests scripts
```
