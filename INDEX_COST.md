# Index ownership cost frontier

The latency experiments answer whether HILDA beats a vector index at query time. This experiment asks a different question: **how much infrastructure must a deployment own to make semantic retrieval available at all?**

HILDA does not eliminate indexing. Its measured query plan depends on a conventional B-tree over a scalar semantic code. The claim under test is narrower: that this scalar index is materially cheaper to store, build, rebuild, and maintain than a dedicated ANN index.

## Pre-registered comparison

The same normalized MiniLM embeddings are loaded into three otherwise-identical PostgreSQL tables:

1. no secondary index, used only as the append baseline;
2. `code bigint` with a B-tree;
3. `embedding vector(384)` with pgvector HNSW.

The benchmark reports physical index bytes and bytes per row, index-build wall time and WAL, append throughput for the same held-out batch, append WAL, and append cost relative to the no-secondary-index table.

Semantic-code assignment for the appended rows is timed separately. HNSW receives already-computed embeddings, so folding HILDA's model-side encoding into PostgreSQL's index-maintenance time would answer a different question.

## Existing storage signal

The 100,000-row memory-pressure run already establishes the order of magnitude: the scalar `code_idx` was 1,564,672 bytes while `hnsw_idx` was 204,800,000 bytes, about 131x larger. The new benchmark tests whether that storage advantage is accompanied by lower build and write-maintenance cost.

## Incremental growth

`INCREMENTAL_GROWTH.md` defines the more demanding version of the thesis: train the semantic codebook once, freeze it, and append documents in waves without recoding existing rows. That run must report both marginal maintenance cost and recall drift. A cheap append path that steadily loses retrieval quality is not evidence that rebuilds are unnecessary; it defines a rebuild interval.
