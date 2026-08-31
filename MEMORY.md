# HILDA against HNSW when the working set does not fit

Every other Postgres number in this repository was measured with the whole
working set resident, and [POSTGRES.md](POSTGRES.md) called that out as the
fourth regime it could not speak to: *the vector working set exceeds available
memory*. This is that experiment.

The question, stated so it can come out either way:

> When the working set stops fitting in memory, is there a point where a B-tree
> range scan plus a cosine re-rank offers a better latency/recall trade-off
> than HNSW?

**The answer is no.** Memory pressure does degrade HNSW badly, and it degrades
it harder than the range scan in relative terms. It never degrades it far
enough to cross. At every matched recall, under every cap measured, HNSW is
faster while returning at least as many true neighbours.

## Protocol

`shared_buffers` is not a memory limit. PostgreSQL also reads through the
operating system's page cache, so an index far larger than `shared_buffers` can
be entirely resident and never touch a device — which is why POSTGRES.md
insisted this regime needs a cap on *total* memory. The server therefore starts
inside a memory cgroup, joined by a root shell that drops to the `postgres` user
before `exec`, so the postmaster and every backend it forks are charged against
one budget.

Caps are ratios of the working set **measured after loading** — table plus
`hnsw_idx` plus `code_idx` — not of a size assumed in advance. Where a corpus is
too small for a requested ratio to pressure rather than starve the server, the
cap clamps to a 192 MiB floor and the row reports the ratio achieved beside the
one asked for.

**Cold** means a full stop, a global `drop_caches`, and a fresh start. Dropping
the page cache alone leaves `shared_buffers` populated, which is a shared-buffer
miss experiment, not a memory-pressure one. **Warm** means the plan ran the same
queries once and the reported pass measures a cache it established itself.

`shared_buffers` (32 MB), parallelism (off), JIT (off), the corpus and the query
set are identical across every condition. Latency, buffer counts and I/O time
come from one `EXPLAIN (ANALYZE, BUFFERS)` per query with `track_io_timing = on`;
collecting them from a second execution would destroy the cold state the
restart established.

### `shared_read` is not evidence of disk

With `shared_buffers` at 32 MiB, almost every block a query touches counts as a
shared *read* whether or not it reaches a device — the page cache inside the cap
serves most of them. The run demonstrates this directly: the exact scan reports
the same ~56,000 shared reads warm and cold, at 26 ms and 47 ms of I/O time
respectively. **`read_ms`, under the cgroup and cold protocol, is the pressure
evidence.** `shared_read` is a buffer-pool statistic and is reported only for
completeness.

### The plan under test has to be the plan named

The smoke run caught a substitution that would have invalidated everything.
HILDA and HNSW reported identical buffer counts — 619.9 hit, 130.6 read, to the
decimal — because with an HNSW index on the table the planner satisfied
`ORDER BY embedding <=> q` from the graph and demoted the code predicate to a
filter. Earlier benchmarks escaped this only because the range scan was measured
before `hnsw_idx` existed. The re-rank now sits behind an `OFFSET 0`
optimisation fence, and `EXPLAIN` confirms each plan:

```
postgres-seqscan-exact     Limit <- Sort <- Seq Scan
hilda-btree+rerank         Limit <- Sort <- Subquery Scan <- Nested Loop <- Values Scan <- Index Scan[code_idx]
pgvector-hnsw-ef80         Limit <- Index Scan[hnsw_idx]
```

The fence changes nothing on a table with no vector index, where the sort has no
index to come from either way, so it does not invalidate POSTGRES.md.

## The sweep: one operating point per method

300,000 Wikipedia lead paragraphs under `all-MiniLM-L6-v2`, 200 held-out queries
split into 100 that choose each method's operating point and 100 that report it,
target recall 0.95. HILDA's depth and probe width and HNSW's `ef_search` are
both chosen on validation and frozen. Raw numbers in
[results/memory_pressure.json](results/memory_pressure.json) and
[results/memory_pressure_tight.json](results/memory_pressure_tight.json).

Warm p50 at 299,800 rows. Working set 1,057 MiB: table 469 MiB, `hnsw_idx`
585 MiB, `code_idx` 3.4 MiB.

| cap | of working set | HILDA (0.960) | HNSW (0.941–0.944) | exact (1.000) | HILDA / HNSW |
|---|---|---|---|---|---|
| 2114 MiB | 2.00 | 33.39 ms | 8.48 ms | 251.24 ms | 3.94x |
| 792 MiB | 0.75 | 35.12 ms | 8.71 ms | 258.54 ms | 4.03x |
| 370 MiB | 0.35 | 168.45 ms | 63.92 ms | 276.85 ms | 2.64x |
| 264 MiB | 0.25 | 217.02 ms | 74.92 ms | 295.87 ms | 2.90x |
| 192 MiB | 0.18 | 259.98 ms | 87.78 ms | 280.87 ms | 2.96x |

**Pressure has a threshold, not a slope.** Nothing happens between 2.00 and
0.75; everything happens between 0.75 and 0.35. The reason is that a query's
footprint is far smaller than the structures it reads from — HNSW at `ef80`
touches ~1,470 blocks, about 11 MiB, of a 585 MiB index — so a cap set as a fraction of total
size does not bind until it drops below what the query stream actually revisits.

**The gap narrows and then re-widens.** It reaches 2.64x at 0.35 and returns to
about 2.96x under deeper pressure. Read from the first three rows alone this
looks like the beginning of a crossing; it is not.

**The range scan's real loss is against the exact scan.** Resident, it is 7.5x
cheaper than scanning the table. At 0.18 it is 1.08x cheaper — 259.98 ms against
280.87 ms — and its I/O time (243 ms) is twice the sequential scan's (116 ms).
About 210 scattered intervals fetch more from a device than one streaming read
once the page cache can no longer amortise them. **The range scan's advantage
over the exact scan is itself a memory-resident phenomenon**, which is the
finding that most directly limits regime 2.

## The frontier: several operating points per method

One point per method at different held-out recalls answers "did the
pre-selected settings cross". It cannot say whose frontier dominates: a latency
ratio between plans returning different answers is not a trade-off. So both
methods are walked across settings, on **one HNSW graph built once**, and every
point reports the recall it actually reaches. Nothing here selects, so held-out
queries are the honest set to score on — a frontier is the curve, not a choice
made from it. Raw numbers in [results/frontier.json](results/frontier.json).

Each HILDA setting is matched to the cheapest HNSW setting returning **at least
as much** recall, which is the comparison most favourable to HILDA.

### Resident (2114 MiB cap, 2.00 of the working set)

Exact scan 262.82 ms at recall 1.000.

| recall | HILDA probes | HILDA p50 | HNSW ef | HNSW p50 | ratio |
|---|---|---|---|---|---|
| 0.867 vs 0.910 | 128 | 8.14 ms | 40 | 5.02 ms | 1.62x |
| 0.927 vs 0.948 | 256 | 15.30 ms | 80 | 8.82 ms | 1.73x |
| 0.960 vs 0.973 | 512 | 34.74 ms | 160 | 15.66 ms | 2.22x |
| 0.983 vs 0.988 | 1024 | 65.89 ms | 320 | 26.05 ms | 2.53x |

### Under pressure (192 MiB cap, 0.18 of the working set)

Exact scan 308.81 ms at recall 1.000.

| recall | HILDA probes | HILDA p50 | HNSW ef | HNSW p50 | ratio |
|---|---|---|---|---|---|
| 0.867 vs 0.910 | 128 | 71.06 ms | 40 | 53.08 ms | 1.34x |
| 0.927 vs 0.948 | 256 | 124.24 ms | 80 | 83.12 ms | 1.49x |
| 0.960 vs 0.973 | 512 | 264.87 ms | 160 | 155.50 ms | 1.70x |
| 0.983 vs 0.988 | 1024 | 526.73 ms | 320 | 273.32 ms | 1.93x |

**Pressure narrows the matched-recall gap by about a quarter and never closes
it.** 1.62 → 1.34, 1.73 → 1.49, 2.22 → 1.70, 2.53 → 1.93. The ordering is the
same at every point on both curves.

HILDA's widest setting, 2048 probes at 0.994, has no HNSW match: HNSW saturates
at 0.992 with `ef640`. Compared at that near-tie, HILDA costs 131.42 ms against
50.39 ms resident and 1,077.17 ms against 494.11 ms under pressure — 2.6x and
2.2x, the same story.

**Under pressure the range scan's useful range collapses.** At 1024 probes it
costs 526.73 ms for 0.983 recall, while the exact sequential scan returns
**everything** in 308.81 ms. Above roughly 0.96 recall, a pressured range scan
is dominated by simply reading the table. HNSW's range is squeezed too — `ef640`
at 494.11 ms is also worse than the exact scan — but it holds a useful frontier
up to 0.988.

## What this changes

POSTGRES.md's four regimes, revised by these measurements:

| regime | status |
|---|---|
| 1. A vector index exists and is affordable | **Measured, twice.** HILDA loses at one operating point and across the whole frontier. |
| 2. No vector index is available | **Measured, and narrowed.** HILDA is 7.5x cheaper than the exact scan resident, 1.08x under pressure, and dominated by it above ~0.96 recall under pressure. The advantage is real where memory is not scarce. |
| 3. A vector index is affordable but undesirable | **Still not measured.** The index costs are inputs to that argument, not the argument. |
| 4. The working set exceeds available memory | **Measured. No crossover.** Pressure degrades HNSW harder in relative terms and never enough to change the ordering. |

The memory-pressure hypothesis was worth testing and it failed. HILDA's case
rests on regimes 2 and 3 — a query fast enough at an index cost two orders of
magnitude smaller — and not on any regime where it beats a vector index at
search.

## What this does not show

- **One machine, one client, no concurrency.** Contention could plausibly favour
  the plan with the smaller resident footprint; nothing here measures that.
- **IVFFlat is excluded.** A second vector index in the same buffer pool would
  make the working set the caps are ratios of ambiguous. IVFFlat was competitive
  with HILDA at 8,000 rows and is untested here.
- **One corpus, one embedder, one `k`, `pgvector` 0.6.0.** Later versions changed
  HNSW search substantially.
- **The cap is on the server, not the machine.** `drop_caches` is global because
  no per-cgroup eviction interface exists.
- **HNSW's build is randomised.** The tighter caps and the frontier come from
  rebuilt graphs whose held-out recalls differ slightly (0.941 against 0.944 at
  `ef80`). Every table reports the recall its own points reached.

## Running it

```bash
uv run scripts/run_memory_pressure_benchmark.py \
  --sizes 100000,299800 --ratios 2.0,0.75,0.35
uv run scripts/run_memory_pressure_benchmark.py \
  --sizes 299800 --ratios 0.25,0.18 --out results/memory_pressure_tight.json
uv run scripts/run_frontier_benchmark.py --rows 299800 --ratios 2.0 0.18
uv run scripts/report_memory.py \
  results/memory_pressure.json results/memory_pressure_tight.json \
  --frontier results/frontier.json
```

Both benchmarks need root: they create a memory cgroup and write
`/proc/sys/vm/drop_caches`.
