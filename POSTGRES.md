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
| `hilda-btree+rerank` | 0.958 | 2.52 ms | 3.75 ms | 429 |
| `pgvector-hnsw-ef40` | 0.992 | 1.14 ms | 1.88 ms | 516 |
| `pgvector-hnsw-ef100` | 0.998 | 1.74 ms | 2.48 ms | 976 |
| `pgvector-hnsw-ef200` | 1.000 | 2.89 ms | 4.46 ms | 1615 |
| `pgvector-ivfflat-probes1` | 0.574 | 0.50 ms | 0.78 ms | 47 |
| `pgvector-ivfflat-probes10` | 0.963 | 0.94 ms | 1.18 ms | 198 |
| `pgvector-ivfflat-probes30` | 0.996 | 1.93 ms | 2.74 ms | 520 |

**The operational thesis does not hold at this scale.** IVFFlat reaches the
same recall as the HILDA plan (0.963 against 0.958) at 2.7x lower p50 latency
and less than half the buffer traffic. HNSW at `ef_search = 40` beats it on
both axes at once: higher recall (0.992) *and* lower latency (1.14 ms). There
is no budget on this corpus where the range scan is the better query.

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

So the trade is real but inverted from the paper's claim: the semantic ID does
not buy a faster query, it buys an almost free index. Whether that matters
depends on a workload where index build time and size dominate — frequent
re-indexing, many small partitions, storage-bound deployments — not on query
latency, which is what `PLAN.md` sets out to improve.

## What this does not show

- **Everything is cached.** `shared_read` is zero on every plan: the table, both
  vector indexes and the B-tree all fit in 256 MB. HNSW traversal is random
  access and degrades when the graph does not fit in memory, which is exactly
  the regime where a sequential range scan could win. At 8,000 rows that regime
  is not reached, and this benchmark cannot speak to it.
- **One machine, no concurrency.** Single client, warm cache, no competing
  load, latency including a local round trip.
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
