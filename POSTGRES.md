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

`Δ recall` is measured against the exact scan, which returns the true top-10
by construction. The plans are not semantically equivalent, and no ratio of
latencies should be read without it.

| plan | recall@10 | Δ recall | p50 | p95 | shared hits |
|---|---|---|---|---|---|
| `postgres-seqscan-exact` | 1.000 | — | 5.90 ms | 7.85 ms | 1600 |
| `hilda-btree+rerank` | 0.958 | −0.042 | 2.09 ms | 2.81 ms | 429 |
| `pgvector-hnsw-ef40` | 0.992 | −0.008 | 1.03 ms | 1.68 ms | 539 |
| `pgvector-hnsw-ef100` | 0.998 | −0.002 | 1.80 ms | 2.56 ms | 992 |
| `pgvector-hnsw-ef200` | 1.000 | −0.000 | 2.76 ms | 3.76 ms | 1623 |
| `pgvector-ivfflat-probes1` | 0.581 | −0.419 | 0.47 ms | 0.66 ms | 50 |
| `pgvector-ivfflat-probes10` | 0.963 | −0.037 | 1.01 ms | 1.40 ms | 201 |
| `pgvector-ivfflat-probes30` | 0.995 | −0.005 | 1.73 ms | 2.65 ms | 527 |

The answer depends on what the range scan is being compared against, and the
two comparisons point opposite ways.

**Against a vector index, the range scan loses.** IVFFlat reaches the same
recall (0.963 against 0.958) at half the p50 latency and half the buffer
traffic. HNSW at `ef_search = 40` beats it on both axes at once: higher recall
*and* lower latency. Where pgvector is available and its index affordable,
there is no budget on this corpus where the range scan is the better query.

**Against no vector index, the range scan offers a much better cost per
query, at a cost in recall.** The exact sequential scan is the plan a
deployment without pgvector actually runs: 5.90 ms and 1,600 buffer hits, for
the true top-10. The range scan answers in 2.09 ms reading 429 of them, 2.8x
faster on 27% of the traffic, and returns 0.958 of those neighbours. It does
not dominate the exact scan — nothing that returns 0.958 dominates something
that returns 1.000 — it trades 0.042 of recall for most of the cost.

Whether that advantage grows with the corpus is a hypothesis, not a result.
The exact scan necessarily grows with the whole table. The range scan's cost
tracks the candidates it retrieves, so the advantage widens **only if the
candidate budget needed to hold recall grows sublinearly with the corpus**. It
may not: more points per cell, or worse locality at the same prefix depth,
could force a wider window to keep 0.958. Measuring that curve is the next
experiment, and this benchmark does not settle it.

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
claim than `PLAN.md`'s, and a more useful one than it sounds.

## Four regimes, and what each one rests on

The regimes differ in what evidence they need, which is why they are worth
separating rather than lumping into "where a vector index is impractical".

| regime | status |
|---|---|
| 1. A vector index exists and is affordable | **Measured.** HILDA loses on both recall and latency. |
| 2. No vector index is available — pgvector not installed, managed service without it | **Measured.** The alternative is the exact scan, and HILDA is 2.8x cheaper per query at −0.042 recall. |
| 3. A vector index could exist, but its storage or rebuild cost makes it undesirable | **Not measured.** The index costs above (0.15 MB against 16.35 MB, 0.014 s against 1.851 s) are inputs to that argument, not the argument. Whether they dominate some real deployment's budget is a claim about that deployment. |
| 4. The working set exceeds memory | **Not measured.** See below. |

Regime 2 is a database capability; regime 3 is an economic decision. They may
end up sharing a query strategy, but they need different evidence, and only
one of them has any here.

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
