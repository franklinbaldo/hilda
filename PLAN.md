# HILDA — Implementation Plan

*Hilbert-Indexed Locality for Document Addressing*

## 0) Goals & non-goals

**Goals**

* Mint **meaning-sortable 128-bit IDs** (HILDA) for any text/chunk.
* Provide a **4096-word codebook** + **semantic codec** (11-word mnemonic; semantic snapping).
* Expose **sCIDR prefixes** (multi-resolution semantic ranges) and **range-scan APIs**.
* Let LLMs use **semantic pointers**: *emit → expand* tools that replace long chains of thought with compact handles.
* Ship a pilot that proves **latency ↓**, **token usage ↓**, and **recall@k ≈ baseline**.

**Non-goals (v1)**

* Cryptographic identity (use content hashes alongside HILDA).
* Perfect neighbor preservation (we’ll re-rank with true cosine).
* Cross-model stability (we pin model+projection per version).

---

## 1) System architecture (high level)

**Artifacts (versioned)**

* `v`: 4-bit semantic space version.
* **Embedder** (e.g., `all-MiniLM-L6-v2`) pinned by commit/hash.
* **Codebook**: 4096 max–min words + embeddings.
* **Projection**: PCA(2) fit on codebook; min/max; Hilbert parameters.

**Services / components**

1. **Codebook Service** → builds/serves codebook; semantic codec (11-word).
2. **HILDA Service** → `mint`, `explain`, `neighbors`, `range-scan`, `prefix`.
3. **Indexer** → ingests content; stores `{content_hash, suid_hi, suid_lo, embed_vec, meta}`.
4. **LLM Tools** → `emit_hilda(text, precision_bits)`, `expand_hilda(suid_or_prefix)`.
5. **Store**

   * Postgres (B-tree on `(suid_hi, suid_lo)`), optional vector index (pgvector/FAISS) for rerank.
   * Object store for artifacts (codebook, PCA npz).

**Data model (Postgres)**

```sql
-- 128-bit SUID stored as two bigints for lexicographic sort/range scan
CREATE TABLE items (
  content_hash BYTEA PRIMARY KEY,
  suid_hi BIGINT NOT NULL,       -- high 64 bits
  suid_lo BIGINT NOT NULL,       -- low 64 bits
  embed VECTOR(384),             -- optional, for final rerank
  v SMALLINT NOT NULL,           -- HILDA version
  meta JSONB
);
CREATE INDEX ON items (suid_hi, suid_lo);
-- Optional: prefix helpers
-- e.g., store a 60-bit hilbert in suid_hi: bits 60..?
```

---

## 2) Phased delivery (12 weeks, 4 sprints)

### Sprint 1 (Weeks 1–3): Foundations

**Tasks**

* Pin **embedder** (public Sentence-Transformers) and infra image.
* Implement **max–min codebook** builder (we already have a script; wrap as a job).
* Produce **v1 artifacts**: `codebook.csv`, `codebook_emb.npy`, `pca_params.npz`.
* Implement **semantic codec**:

  * UUID↔11 words (base-4096 + CRC-4), **semantic snapping** decoder.
* Implement **HILDA mint/explain** (PCA→Hilbert→pack 128b).
* Set up **artifact registry** with provenance (model hash, time, dataset).

**Deliverables**

* Containerized jobs/services:

  * `codebook:build`, `codec:encode/decode`, `hilda:mint/explain`.
* Postgres schema + migration.
* Small CLI/SDK (Python + TypeScript).

**Success**

* Deterministic IDs for the same input under `v=1`.
* End-to-end: text → HILDA → explain returns nearest codebook words.

---

### Sprint 2 (Weeks 4–6): sCIDR + storage + range scans

**Tasks**

* Add **prefix (precision) API**: `precision_bits ∈ [24..60]`.
* Implement **range-scan** over `(suid_hi, suid_lo)` for a given prefix.
* Add **rerank** stage: fetch N by range, compute cosine over stored embeddings, return top-k.
* Build **ingestion indexer**: compute HILDA + store embeddings & metadata.

**APIs**

* `POST /hilda/mint {text, precision_bits}` → `{suid_hex, prefix}`
* `GET /hilda/range?prefix=...&limit=N` → items (approx neighbors)
* `POST /neighbors {text|suid, k}` → hybrid (range + cosine rerank)

**Success**

* Range-scan latency ~ **<10ms** for B-tree fetch (small N).
* Recall@k vs HNSW baseline within **−3%** on an eval set.

---

### Sprint 3 (Weeks 7–9): LLM semantic pointers (Emit→Expand)

**Tasks**

* Define **tool spec** (OpenAI Tools / function calling schema):

  * `emit_hilda(text, precision_bits=40)` → `{suid, prefix, nearest_words}`
  * `expand_hilda({suid|prefix}, k=8)` → `{snippets, citations, neighbors}`
* Add **pointer markup**: `⟦HILDA:...⟧` recognized by middleware.
* Few-shot **prompting / light finetune**:

  * Teach the model to **emit** HILDAs instead of verbose CoT for recurring sub-steps.
  * Teach **expand** calls on demand.
* Build **cache** keyed by prefix to serve hot neighborhoods.

**Success**

* Token usage **↓ 20–40%** on target tasks (internal eval).
* Quality parity vs. standard RAG (within confidence bounds).

---

### Sprint 4 (Weeks 10–12): Advanced & pilot

**Tasks**

* Add **multi-HILDA triangulation** (emit 2–3 IDs; intersect ranges).
* Optional **PQ-words handle** (8×12-bit product code emitted as 8 words).
* **Pilot integration** (choose a corpus):

  * Index documents/precedents; wire HILDA search & pointer expansions.
* **Monitoring/Drift**

  * Collision rate (items per cell), neighbor drift, version gatekeeping.

**Success**

* Pilot users obtain **faster retrieval** and **consistent pointer reuse**.
* Metrics dashboards live; alarms for drift/collisions.

---

## 3) Technical details & checklists

### 3.1 Versioning / reproducibility

* `v` = 4 bits in SUID. Freeze:

  * model name + hash
  * codebook CSV + embeddings
  * `pca_params.npz` (mu, W, mins, maxs)
* Record **build manifest**; reject cross-version mixing at runtime.

### 3.2 HILDA packing

* 128 bits (big-endian):

  * `ver(4) | hilbert60(60) | unix_ms(48) | rand16(16)`
* Store as `(hi64, lo64)`; implement pure-SQL `prefix_mask(ℓ)` helpers for sCIDR.

### 3.3 Range-scan → rerank

* Choose N (e.g., **N=512**) by prefix window, then cosine rerank to **k=20**.
* Auto-widen prefix (coarser) if too sparse; narrow if too dense.

### 3.4 Codebook & codec

* Max–min selection with:

  * L2-normed embeddings
  * near-duplicate suppression (edit distance ≤2, sim ≥0.95)
  * optional POS/frequency penalties
* **Decoder** supports **free-phrase snapping** per slot; CRC-4 integrity; ECC optional.

### 3.5 SDKs

* Python & TS:

  * `mint(text, precision_bits=60) -> HILDA`
  * `prefix(hilda, bits) -> prefix`
  * `range(prefix, limit)`
  * `neighbors(text|hilda, k)`
  * `encode_uuid`, `semantic_decode(words|phrases)`

### 3.6 Observability

* **Counters**: mints, range-scans, expands, cache hit rate.
* **Quality**: recall@k vs baseline, triangle-inequality violations (%), cell occupancy hist.
* **Drift**: KL divergence over prefix histograms (weekly).

### 3.7 Security & governance

* HILDA is **not secret**; do **authZ at expansion**.
* Log **expansion provenance** (docs, timestamps).
* Data retention policies for embeddings and expansions.

---

## 4) Pilot plan (domain-ready)

1. **Scope corpus** (e.g., briefs/precedents/notes).
2. **Index** with HILDA + embeddings.
3. **Define expansions**: for frequent theses, curate stable templates keyed by HILDA.
4. **Evaluate**:

   * Retrieval: latency, recall@k
   * Reasoning: tokens, accuracy, judge scores
5. **Adopt**: ship pointer-first workflows in your app; keep full RAG as fallback.

---

## 5) Risks & mitigations

* **Model drift** → strict versioning; automatic re-mint only behind `v+1`.
* **Hot cells/collisions** → auto split (finer prefix), multi-HILDA triangulation.
* **Bias/domain mismatch** → domain-specific `v=2-<domain>` with its own PCA.
* **User confusion** → always **explain**: show nearest codebook words for any HILDA.

---

## 6) Acceptance criteria (v1)

* Deterministic mint/explain under `v=1`.
* Range-scan + rerank achieves **≥97%** of HNSW recall@k on eval.
* Average token reduction **≥25%** on targeted flows using pointers.
* P99 range-scan **<15 ms**; P99 neighbors end-to-end **<150 ms** (warm cache).
* Full audit trail: artifact manifest, version checks, access control at expansion.

---

## 7) Immediate next steps (this week)

* Turn the existing scripts into **Dockerized services**.
* Generate **v1 artifacts** from a seed corpus and pin them.
* Stand up **Postgres** + schema; ingest a small dataset.
* Wire **/mint**, **/range**, **/neighbors**; expose **LLM tools**.
* Ship a **sandbox demo** (UI): type text → see HILDA, nearest words, range neighbors.

If you want, I can output a **project board** (issues & owners) or a **Terraform/Docker Compose** starter to stand up the stack in one go.
