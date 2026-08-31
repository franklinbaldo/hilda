# Memory-pressure crossover benchmark

PR #3 established two things separately: HILDA's candidate budget grows sublinearly at roughly fixed recall on the measured ladder, and HILDA loses to pgvector's vector indexes while everything is memory-resident. This experiment asks the missing fourth-regime question without conflating `shared_buffers` with total memory.

## Question

> When the vector working set no longer fits in the process/container memory budget, does B-tree range scan + exact rerank ever become a better latency/recall trade-off than HNSW?

A negative result is useful: it confines HILDA's operational advantage to deployments without a vector index or where index cost is unacceptable. A positive result identifies a real memory-pressure crossover.

## Experimental contract

The independent variable is **total memory available to the database workload**, enforced by a cgroup/container memory limit. `shared_buffers` is recorded but is not the memory boundary: PostgreSQL and pgvector also benefit from the operating-system page cache.

For each corpus size, run the same data and held-out queries under at least three memory regimes:

1. **resident** — relation + active index comfortably fit inside the total memory limit;
2. **near-boundary** — the measured working set is close to the total memory limit;
3. **pressure** — relation + active index exceed the total memory limit enough to force cache eviction during randomized query access.

Do not infer a memory regime from `index_size > shared_buffers` alone.

## Plans

Measure these independently so inactive indexes do not distort the working set:

- exact sequential scan, no vector index;
- HILDA B-tree + exact cosine rerank;
- HNSW at validation-selected `ef_search` for the target recall.

IVFFlat may be retained as a secondary reference, but the crossover claim is about HNSW because its graph traversal is the random-access case most exposed to memory pressure.

## Recall matching

Use a pre-registered target recall, default 0.95. Select HILDA's depth/budget and HNSW's `ef_search` on validation queries, then freeze them before timing test queries. Report both validation and test recall.

Do not compare latency at materially different recall. If a plan cannot reach the target on validation, record that as an unreachable operating point rather than silently lowering the target.

## Warm and cold phases

Each `(corpus size, memory limit, plan)` cell has two phases.

### Warm

Run one unreported warm-up pass, then time the reported pass without restarting PostgreSQL. This measures steady-state behavior under the imposed memory ceiling.

### Cold / eviction-sensitive

Restart the database workload inside the same memory-limited cgroup/container before the reported pass. A restart clears PostgreSQL's own shared-buffer state but does **not** prove the host page cache is cold.

A run may be labelled `cold` only when one of these is true:

- the benchmark environment explicitly drops the relevant page cache before the run; or
- the randomized working set demonstrably exceeds the cgroup memory ceiling, and cgroup/page-fault counters show eviction/refault activity during the run.

Otherwise label it `restart-only`, not `cold`.

## Required evidence per cell

Record:

- corpus rows and embedding width;
- target, validation, and test recall;
- p50 and p95 client latency;
- candidate count and range count for HILDA;
- `shared_hit` and `shared_read` from `EXPLAIN (ANALYZE, BUFFERS)` samples;
- heap/table size and active index size via `pg_relation_size`;
- `shared_buffers`;
- cgroup memory limit and current/peak memory where available;
- cgroup major-fault/refault or equivalent I/O-pressure counters where available;
- whether the phase is `warm`, `restart-only`, or genuinely `cold`.

`shared_read > 0` is evidence that PostgreSQL missed shared buffers. It is not by itself proof of physical disk I/O because the operating system may still satisfy the read from page cache.

## Decision rule

The experiment supports a memory-pressure crossover only if, on held-out test queries at comparable recall, HILDA becomes preferable to HNSW on latency in one or more pressure cells and the change coincides with independently observed memory/I/O pressure.

A one-off faster sample without pressure evidence is not a crossover.

If HNSW remains better through the pressure cells, state that plainly. The scale-ladder result remains valid against exact scan, but the project should not claim an HNSW out-of-memory advantage.

## Suggested matrix

Start with a corpus large enough that HNSW and the table can exceed a practical memory ceiling. For each selected corpus size, choose memory limits from measured relation sizes rather than fixed labels. A useful first pass is approximately:

- 2.0× measured active working-set size;
- 1.0× measured active working-set size;
- 0.5× measured active working-set size.

The ratios are starting points, not claims. Persist the actual byte limits and relation sizes in the result JSON.

## Separation from the scale ladder

The scale ladder answers `candidate_budget(N)` while embeddings are memory-resident. This benchmark answers whether database-plan behavior changes when the working set cannot stay resident. Neither result substitutes for the other.
