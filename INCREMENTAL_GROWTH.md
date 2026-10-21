# Incremental growth without global reindexing

HILDA's strongest operational claim may not be query latency. A frozen semantic codebook defines regions of embedding space before documents occupy them. New documents can therefore be encoded into an already-defined scalar address and inserted into an ordinary B-tree without retraining the codebook or rebuilding the existing index.

HNSW is also incrementally insertable, so the claim under test is **not** that HNSW requires a rebuild after every append. pgvector maintains the graph during inserts. The narrower question is whether HILDA's marginal maintenance cost is materially lower, and whether that advantage survives semantic drift when the codebook is not retrained.

## Experiment

Fit one hkmeans L4 x K16 codebook on the initial 50,000 documents. Freeze it for the rest of the run.

Grow the database in 50,000-document waves to 300,000 rows. The same embeddings and append batches feed two secondary-index strategies:

- HILDA: encode each incoming vector with the frozen codebook and insert `(code, embedding)` while maintaining only the scalar B-tree;
- HNSW: insert the same already-computed embedding while maintaining pgvector's graph index.

A no-secondary-index table provides the PostgreSQL append baseline.

At every wave record:

- code-assignment wall time, separately from PostgreSQL maintenance;
- append wall time and rows/s;
- append WAL bytes and WAL bytes/row;
- physical secondary-index bytes and bytes/row;
- HILDA recall@10 using the *same frozen codebook*;
- HNSW recall@10 at a fixed `ef_search` operating point;
- exact-search recall=1 baseline.

## Decision rule

The incremental-maintenance thesis is supported only if HILDA's append/storage cost stays materially below HNSW while held-out recall does not deteriorate systematically as the corpus moves away from the 50k training snapshot.

If HILDA stays cheap but recall falls with each wave, the result is not "no reindexing required". It is a measurable rebuild interval: the largest corpus growth the frozen codebook tolerates before quality falls below the deployment target.

That rebuild interval is itself an operational quantity and should be reported in rows added and growth multiple relative to the training snapshot.
