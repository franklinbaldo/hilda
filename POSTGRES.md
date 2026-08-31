# HILDA against pgvector, on a real Postgres

The ablation counts ranges over a sorted array in memory. That is a model of a
range scan, not a range scan. This asks the planner instead: the same codes in
a B-tree, the same embeddings under HNSW and IVFFlat, one query set, three
plans.

The question it answers:

> Does a B-tree range scan plus a cosine re-rank actually beat a vector index,
> or only in the proxy?

## Setup

Postgres 16 with pgvector 0.6.0, `shared_buffers = 256MB`, parallel workers
off. 8,000 documents from 20 newsgroups embedded with `all-MiniLM-L6-v2`, 200
held-out test queries, recall@10 against exact cosine. The HILDA plan uses
`hkmeans-L4xK16` at the operating point the ablation selected on its validation
queries (depth 3, 64 probes). Latency is measured client-side around
`cursor.execute`, so it includes the round trip on a local socket.

Regenerate with [scripts/run_postgres_benchmark.py](scripts/run_postgres_benchmark.py);
the raw numbers are in [results/postgres.json](results/postgres.json).

## Query cost

| plan | recall@10 | p50 | p95 | shared hits |
|---|---|---|---|---|
| `postgres-seqscan-exact` | 1.000 | 5.90 ms | 7.85 ms | 1600 |
| `hilda-btree+rerank` | 0.958 | 2.09 ms | 2.81 ms | 429 |
| `pgvector-hnsw-ef40` | 0.992 | 1.03 ms | 1.68 ms | 539 |
| `pgvector-hnsw-ef100` | 0.998 | 1.80 ms | 2.56 ms | 992 |
| `pgvector-hnsw-ef200` | 1.000 | 2.76 ms | 3.76 ms | 1623 |
| `pgvector-ivfflat-probes1` | 0.581 | 0.47 ms | 0.66 ms | 50 |
| `pgvector-ivfflat-probes10` | 0.963 | 1.01 ms | 1.40 ms | 201 |
| `pgvector-ivfflat-probes30` | 0.995 | 1.73 ms | 2.65 ms | 527 |

The answer depends on what the range scan is being compared against, and the
two comparisons point opposite ways.

**Against a vector index, the range scan loses.** IVFFlat reaches the same
recall (0.963 against 0.958) at half the p50 latency and half the buffer
traffic. HNSW at `ef_search = 40` beats it on both axes at once: higher recall
*and* lower latency. Where pgvector is available and its index affordable,
there is no budget on this corpus where the range scan is the better query.

**Against no vector index, the range scan wins clearly.** The exact sequential
scan is the plan a deployment without pgvector actually runs, and it costs
5.90 ms and 1,600 buffer hits. The range scan answers in 2.09 ms reading 429,
giving up 4 points of recall: 2.8x faster on a quarter of the traffic. That
gap should widen with corpus size, since the sequential scan grows with the
table while the range scan grows with the candidate set.

So the cheap index is not merely cheap to keep; it is what buys the faster
query in the regime where a vector index is not on the table.

## Index cost

| index | build | size |
|---|---|---|
| `code_idx` (B-tree) | 0.012 s | 0.15 MB |
| `hnsw_idx` | 2.335 s | 16.35 MB |
| `ivf_idx` | 0.886 s | 13.58 MB |

**This is where the approach wins, and by two orders of magnitude.** The B-tree
is 109x smaller than the HNSW index — smaller than the table's own 13 MB — and
builds roughly 200x faster. A vector index here costs more to store than the
data it indexes.

Read together with the query table, this is the shape of the result: the
semantic ID buys a fast-enough query at almost no index cost. That is a weaker
claim than `PLAN.md`'s — it does not beat a vector index — and a more useful
one than it sounds, because the regimes where 16 MB of index per shard is
unaffordable are common: many small tenants, write-heavy tables that must be
re-indexed, storage-bound deployments, or any Postgres without pgvector
installed at all. In those, the alternative is the 5.90 ms sequential scan,
not the 1.03 ms HNSW.

## What this does not show

- **Everything is cached.** `shared_read` is zero on every plan: the table, both
  vector indexes and the B-tree all fit in 256 MB. HNSW traversal is random
  access and degrades when the graph does not fit in memory, which is exactly
  the regime where a sequential range scan could win. At 8,000 rows that regime
  is not reached, and this benchmark cannot speak to it.
- **One machine, no concurrency.** Single client, warm cache, no competing
  load, latency including a local round trip.
- **The regimes are not priced.** "A vector index is unaffordable here" is an
  assertion about someone's deployment, not something this benchmark measured.
  What it measures is what each plan costs once that choice is made.
- **The recall ceiling is the encoder's.** The HILDA plan cannot exceed the
  0.958 its operating point reaches; the vector indexes approach 1.0. Raising
  it means scanning more candidates, which widens the latency gap.
- **pgvector 0.6.0.** Later versions changed HNSW build and search
  substantially.

## Running it

```bash
initdb -D "$PGDATA" -A trust -U postgres
pg_ctl -D "$PGDATA" -o '-p 5433 -c shared_buffers=256MB' start
psql -p 5433 -U postgres -c 'CREATE EXTENSION vector'
uv run scripts/run_postgres_benchmark.py --dsn postgresql://postgres@127.0.0.1:5433/postgres
```
