# HILDA/SUID Implementation Project Plan

**12-Week Development & Deployment Roadmap**

**Version:** 1.0  
**Last Updated:** October 15, 2025  
**Status:** Planning Phase

---

## Executive Summary

### Vision
Build a production-ready semantic UUID system (HILDA/SUID) that bridges human readability, database efficiency, and semantic meaning through:
- A 4096-word diversity-first codebook for human I/O
- 128-bit meaning-sortable identifiers for B-tree storage
- Semantic pointer protocol for LLM token efficiency

### Key Objectives
1. **Deterministic & versioned** semantic identifiers
2. **Range-scannable** retrieval achieving ≥97% of vector index recall
3. **Token reduction** of ≥25% for LLM reasoning tasks
4. **Sub-15ms** p99 latency for range scans
5. **Production pilot** with monitoring & drift detection

### Timeline Overview
- **Sprint 1** (Weeks 1-3): Foundations & Core Implementation
- **Sprint 2** (Weeks 4-6): Storage, Indexing & Range Scans
- **Sprint 3** (Weeks 7-9): LLM Integration & Pointer Protocol
- **Sprint 4** (Weeks 10-12): Advanced Features & Production Pilot

### Resource Requirements
- **Engineering:** 2-3 full-time engineers
- **Infrastructure:** Postgres DB, embedder service, artifact storage
- **Compute:** GPU for embedder (can be shared/batched)
- **Budget:** ~$2-5K/month for cloud resources during development

---

## Sprint 1: Foundations (Weeks 1-3)

**Goal:** Establish core artifacts, implement HILDA encoding/decoding, and validate deterministic ID generation.

### Week 1: Infrastructure & Embedder Setup

#### Task 1.1: Pin Embedder Model
**Owner:** DevOps + ML Engineer  
**Effort:** 2 days  
**Dependencies:** None

**Details:**
- Select embedder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) or `all-mpnet-base-v2` (768-dim)
- Pin exact version and model hash
- Containerize embedder service (FastAPI + transformers)
- Implement batch inference endpoint: `POST /embed` → embeddings
- Add health checks and model warmup

**Acceptance Criteria:**
- [ ] Embedder container builds and runs
- [ ] Consistent outputs for same input (temperature=0)
- [ ] Documented model hash in artifact manifest
- [ ] API responds within 50ms for single text (p95)

**Dockerfile Example:**
```dockerfile
FROM python:3.10-slim
RUN pip install torch sentence-transformers fastapi uvicorn
COPY embedder_service.py /app/
CMD ["uvicorn", "app.embedder_service:app", "--host", "0.0.0.0"]
```

---

#### Task 1.2: Docker Base Images & CI/CD
**Owner:** DevOps  
**Effort:** 3 days  
**Dependencies:** Task 1.1

**Details:**
- Set up multi-stage Docker builds for services
- GitHub Actions / GitLab CI for automated builds
- Container registry (Docker Hub, ECR, GCR)
- Environment configs: dev, staging, prod

**Acceptance Criteria:**
- [ ] CI/CD pipeline builds on every commit
- [ ] Automated tests run in containers
- [ ] Tagged images pushed to registry
- [ ] Environment-specific configs validated

---

#### Task 1.3: Artifact Registry Design
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** None

**Details:**
- Design S3/GCS bucket structure:
  ```
  artifacts/
    v1/
      manifest.json
      codebook.csv
      codebook_embeddings.npy
      pca_params.npz
      build_log.txt
  ```
- Implement artifact versioning: content-addressed hashes
- Create manifest schema:
  ```json
  {
    "version": "v1",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "model_hash": "sha256:...",
    "codebook_size": 4096,
    "embedding_dim": 384,
    "pca_components": 2,
    "created_at": "2025-10-15T10:00:00Z",
    "corpus_source": "wikipedia_en_sample_100k"
  }
  ```

**Acceptance Criteria:**
- [ ] Upload/download scripts tested
- [ ] Manifest validation implemented
- [ ] Version-locked artifact retrieval works

---

### Week 2: Codebook Generation

#### Task 2.1: Candidate Lexicon Curation
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Task 1.1

**Details:**
- Source candidate words (50k-100k):
  - Common nouns/verbs from word frequency lists
  - Filter: 3-12 characters, ASCII-friendly
  - Remove offensive/ambiguous terms
- POS tagging (spaCy): prefer nouns/verbs, limit adjectives
- Embed all candidates using pinned embedder
- Save `candidates.csv` with columns: word, pos, frequency, embedding_idx

**Acceptance Criteria:**
- [ ] 50k+ candidate words collected
- [ ] POS distribution: >60% nouns, >20% verbs
- [ ] All embeddings computed and saved
- [ ] Documented filtering criteria

---

#### Task 2.2: Max-Min Selection Implementation
**Owner:** ML Engineer  
**Effort:** 2 days  
**Dependencies:** Task 2.1

**Details:**
- Implement greedy max-min (Gonzalez) algorithm:
  ```python
  def select_codebook(embeddings, k=4096, seed='central'):
      # L2 normalize
      X = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
      
      # Initialize with seed (e.g., closest to mean)
      selected = [seed_idx]
      d_min = 1 - X @ X[seed_idx]  # cosine distances
      
      for _ in range(k - 1):
          # Select farthest point
          j = np.argmax(d_min)
          selected.append(j)
          
          # Update distances
          d_j = 1 - X @ X[j]
          d_min = np.minimum(d_min, d_j)
          
          # Near-duplicate suppression
          # (edit distance ≤2 or cosine >0.95)
      
      return selected
  ```
- Add penalties:
  - Frequency penalty: `penalty = λ * log(freq + 1)`
  - POS penalty: discourage rare POS classes
- Near-duplicate suppression: edit distance ≤2 or cosine similarity >0.95

**Acceptance Criteria:**
- [ ] 4096 words selected
- [ ] Minimum pairwise cosine distance >0.3
- [ ] Documented selection parameters (λ, μ, seed)
- [ ] Visual verification: t-SNE plot shows spread

---

#### Task 2.3: Generate v1 Codebook Artifacts
**Owner:** ML Engineer  
**Effort:** 1 day  
**Dependencies:** Task 2.2

**Details:**
- Run codebook selection on curated candidates
- Fit PCA(2) on codebook embeddings:
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(codebook_embeddings)
  
  # Save artifacts
  np.savez('pca_params.npz',
           mu=mu, W=pca.components_.T,
           mins=pca_space.min(axis=0),
           maxs=pca_space.max(axis=0))
  ```
- Assign integer IDs [0, 4095] → words (sorted by selection order)
- Upload artifacts to registry with manifest

**Acceptance Criteria:**
- [ ] `codebook.csv`: word, id, embedding_idx
- [ ] `codebook_embeddings.npy`: shape (4096, 384)
- [ ] `pca_params.npz`: mu, W, mins, maxs
- [ ] Manifest uploaded with SHA-256 hashes

---

### Week 3: Core HILDA Implementation

#### Task 3.1: Hilbert Curve Encoding/Decoding
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** None

**Details:**
- Implement symmetric 2D Hilbert curve (no gray-code mismatch)
- Functions:
  ```python
  def xy2d(n: int, x: int, y: int) -> int:
      """Convert 2D coordinates to Hilbert distance."""
      # (implementation from hilda_benchmark_improved.py)
  
  def d2xy(n: int, d: int) -> Tuple[int, int]:
      """Convert Hilbert distance to 2D coordinates."""
      # (implementation from hilda_benchmark_improved.py)
  ```
- Unit tests: round-trip consistency for all (x,y) pairs at various bit depths

**Acceptance Criteria:**
- [ ] Round-trip tests pass: `d2xy(xy2d(x,y)) == (x,y)` for all coords
- [ ] Supports bit depths: 10, 15, 20, 30 bits per axis
- [ ] Performance: <1μs per encode/decode

---

#### Task 3.2: PCA Fitting & Projection
**Owner:** ML Engineer  
**Effort:** 2 days  
**Dependencies:** Task 2.3

**Details:**
- Load PCA params from artifacts
- Implement projection functions:
  ```python
  def project_to_2d(v: np.ndarray, mu, W, mins, maxs):
      """Project D-dim vector to [0,1]^2."""
      y = (v - mu) @ W  # 2D PCA space
      z = (y - mins) / (maxs - mins + 1e-9)
      return np.clip(z, 0, 1)
  ```
- Handle edge cases: out-of-distribution vectors (clip gracefully)

**Acceptance Criteria:**
- [ ] All codebook vectors project to diverse (x,y) coords
- [ ] Projection is deterministic (same input → same output)
- [ ] Performance: <0.1ms per projection

---

#### Task 3.3: HILDA Mint & Explain Functions
**Owner:** Backend Engineer  
**Effort:** 3 days  
**Dependencies:** Task 3.1, 3.2

**Details:**
- Implement HILDA minting:
  ```python
  def mint_hilda(text: str, precision_bits: int = 60,
                 embedder, pca_params, version: int = 1):
      """Generate HILDA from text."""
      # 1. Embed and normalize
      v = embedder.encode(text)
      v = v / np.linalg.norm(v)
      
      # 2. Project to [0,1]^2
      z = project_to_2d(v, **pca_params)
      
      # 3. Quantize to (x, y) at precision_bits per axis
      n = 1 << precision_bits
      x = int(round(z[0] * (n - 1)))
      y = int(round(z[1] * (n - 1)))
      
      # 4. Hilbert encode
      h = xy2d(n, x, y)
      
      # 5. Pack into 128 bits
      # Layout: ver(4) | h(60) | unix_ms(48) | rand(16)
      timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
      rand16 = random.randint(0, (1 << 16) - 1)
      
      hilda_int = (version << 124) | (h << 64) | (timestamp_ms << 16) | rand16
      return hilda_int.to_bytes(16, 'big').hex()
  ```

- Implement explain function:
  ```python
  def explain_hilda(hilda_hex: str, pca_params, codebook):
      """Decode HILDA to nearest codebook words."""
      hilda_bytes = bytes.fromhex(hilda_hex)
      hilda_int = int.from_bytes(hilda_bytes, 'big')
      
      # Extract fields
      version = (hilda_int >> 124) & 0xF
      h = (hilda_int >> 64) & ((1 << 60) - 1)
      timestamp_ms = (hilda_int >> 16) & ((1 << 48) - 1)
      rand16 = hilda_int & 0xFFFF
      
      # Decode Hilbert to (x, y)
      n = 1 << 30  # assuming 30 bits per axis
      x, y = d2xy(n, h)
      
      # Map back to [0, 1]^2
      z = np.array([x / (n - 1), y / (n - 1)])
      
      # Find nearest codebook words
      # (project codebook embeddings, compute distances)
      nearest_words = find_nearest_k_words(z, codebook, k=5)
      
      return {
          'version': version,
          'hilbert_index': h,
          'timestamp': timestamp_ms,
          'nearest_words': nearest_words
      }
  ```

**Acceptance Criteria:**
- [ ] Deterministic minting (same text+precision → same prefix)
- [ ] Explain returns sensible nearest words
- [ ] Round-trip: mint → explain → similar semantics
- [ ] Unit tests cover edge cases

---

#### Task 3.4: Semantic Codec (UUID ↔ Words)
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 2.3

**Details:**
- Implement base-4096 encoding:
  ```python
  def encode_uuid_to_words(uuid_bytes: bytes, codebook: List[str]):
      """Encode 128-bit UUID as 11 words with CRC-4."""
      uuid_int = int.from_bytes(uuid_bytes, 'big')
      
      # Add CRC-4 nibble
      crc = compute_crc4(uuid_bytes)
      payload = (uuid_int << 4) | crc  # 132 bits
      
      # Convert to base-4096 (11 digits, 12 bits each)
      digits = []
      for _ in range(11):
          digits.append(payload & 0xFFF)
          payload >>= 12
      
      words = [codebook[d] for d in reversed(digits)]
      return words
  ```

- Implement semantic decoding (phrase snapping):
  ```python
  def decode_words_to_uuid(phrases: List[str], codebook, embedder):
      """Decode words/phrases to UUID via semantic snapping."""
      digits = []
      for phrase in phrases:
          # Embed phrase
          phrase_emb = embedder.encode(phrase)
          phrase_emb /= np.linalg.norm(phrase_emb)
          
          # Find nearest codebook word
          similarities = codebook_embeddings @ phrase_emb
          best_idx = np.argmax(similarities)
          digits.append(best_idx)
      
      # Reconstruct payload
      payload = 0
      for d in digits:
          payload = (payload << 12) | d
      
      # Verify CRC
      crc_received = payload & 0xF
      uuid_int = payload >> 4
      uuid_bytes = uuid_int.to_bytes(16, 'big')
      crc_computed = compute_crc4(uuid_bytes)
      
      if crc_received != crc_computed:
          raise ValueError("CRC mismatch")
      
      return uuid_bytes.hex()
  ```

**Acceptance Criteria:**
- [ ] Encode/decode round-trip works for random UUIDs
- [ ] Semantic decoding handles paraphrases (>90% accuracy)
- [ ] CRC detects single-word errors
- [ ] Performance: <5ms per encode/decode

---

### Sprint 1 Deliverables

**Code:**
- [ ] Dockerized embedder service
- [ ] Codebook builder job
- [ ] HILDA mint/explain library (Python)
- [ ] Semantic codec (UUID ↔ words)

**Artifacts:**
- [ ] v1 codebook (4096 words)
- [ ] v1 PCA parameters
- [ ] Artifact manifest with hashes

**Documentation:**
- [ ] API specs for embedder service
- [ ] Codebook generation runbook
- [ ] HILDA encoding spec

**Tests:**
- [ ] Unit tests: Hilbert, PCA, codec
- [ ] Integration tests: end-to-end mint → explain

### Sprint 1 Success Criteria
✅ **Determinism:** Same input → same HILDA (fixed precision, version)  
✅ **Explain:** Returns top-5 nearest codebook words within 50ms  
✅ **Codec:** Semantic decoding handles paraphrases with >90% accuracy  
✅ **Artifacts:** All v1 artifacts uploaded with verified hashes

### Sprint 1 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Codebook lacks diversity | High | Low | Monitor min cosine distance; adjust penalties |
| PCA loses too much info | Medium | Medium | Evaluate reconstruction error; consider 3D |
| Hilbert implementation bug | High | Low | Extensive unit tests; visual validation |
| Embedder drift (model updates) | High | Low | Pin exact version; never auto-update |

---

## Sprint 2: Storage & Range Scans (Weeks 4-6)

**Goal:** Implement database schema, range-scan APIs, and hybrid retrieval (range + rerank).

### Week 4: Database Schema & Indexing

#### Task 4.1: Postgres Schema Design
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Sprint 1 complete

**Details:**
- Create schema:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  
  CREATE TABLE items (
      content_hash BYTEA PRIMARY KEY,
      suid_hi BIGINT NOT NULL,        -- high 64 bits of HILDA
      suid_lo BIGINT NOT NULL,        -- low 64 bits of HILDA
      embed vector(384),              -- full embedding for rerank
      version SMALLINT NOT NULL,      -- HILDA version
      text_content TEXT,
      metadata JSONB,
      created_at TIMESTAMPTZ DEFAULT NOW()
  );
  
  -- Index for range scans (lexicographic order)
  CREATE INDEX idx_suid ON items (suid_hi, suid_lo);
  
  -- Optional: GIN index on metadata
  CREATE INDEX idx_metadata ON items USING GIN (metadata);
  ```

- Helper functions for prefix queries:
  ```sql
  CREATE FUNCTION suid_prefix_range(
      prefix_hi BIGINT,
      prefix_lo BIGINT,
      prefix_bits INT
  ) RETURNS TABLE(min_hi BIGINT, min_lo BIGINT, 
                   max_hi BIGINT, max_lo BIGINT) AS $$
  BEGIN
      -- Compute range for prefix (mask lower bits)
      -- Implementation details...
  END;
  $$ LANGUAGE plpgsql IMMUTABLE;
  ```

**Acceptance Criteria:**
- [ ] Schema created and migrated
- [ ] Indexes tested with EXPLAIN ANALYZE
- [ ] Helper functions validated
- [ ] Connection pooling configured (pgBouncer)

---

#### Task 4.2: Ingestion Pipeline
**Owner:** Backend Engineer  
**Effort:** 3 days  
**Dependencies:** Task 4.1

**Details:**
- Implement batch indexer:
  ```python
  def ingest_documents(docs: List[Dict], embedder, hilda_minter, db_conn):
      """Batch ingest documents into items table."""
      for doc in docs:
          # Compute content hash
          content = doc['text']
          content_hash = hashlib.sha256(content.encode()).digest()
          
          # Embed
          embedding = embedder.encode(content)
          
          # Mint HILDA
          hilda_hex = hilda_minter.mint(content, precision_bits=60)
          hilda_bytes = bytes.fromhex(hilda_hex)
          suid_hi = int.from_bytes(hilda_bytes[:8], 'big', signed=True)
          suid_lo = int.from_bytes(hilda_bytes[8:], 'big', signed=True)
          
          # Insert
          db_conn.execute("""
              INSERT INTO items (content_hash, suid_hi, suid_lo, 
                                 embed, version, text_content, metadata)
              VALUES (%s, %s, %s, %s, %s, %s, %s)
              ON CONFLICT (content_hash) DO NOTHING
          """, (content_hash, suid_hi, suid_lo, embedding, 1, 
                content, doc.get('metadata', {})))
  ```

- CLI tool: `hilda index --input docs.jsonl --batch-size 100`

**Acceptance Criteria:**
- [ ] Batch ingestion of 10K documents in <5 minutes
- [ ] Handles duplicates (content_hash conflict)
- [ ] Progress bar and error logging
- [ ] Dry-run mode for validation

---

#### Task 4.3: Basic Query API
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 4.2

**Details:**
- FastAPI service:
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel
  
  app = FastAPI()
  
  class MintRequest(BaseModel):
      text: str
      precision_bits: int = 60
  
  @app.post("/hilda/mint")
  def mint_endpoint(req: MintRequest):
      hilda_hex = hilda_minter.mint(req.text, req.precision_bits)
      explain = explainer.explain(hilda_hex)
      return {
          "hilda": hilda_hex,
          "version": explain['version'],
          "nearest_words": explain['nearest_words']
      }
  
  @app.get("/hilda/explain/{hilda_hex}")
  def explain_endpoint(hilda_hex: str):
      return explainer.explain(hilda_hex)
  ```

**Acceptance Criteria:**
- [ ] API runs in Docker container
- [ ] OpenAPI docs auto-generated
- [ ] Rate limiting configured (100 req/min)
- [ ] Healthcheck endpoint

---

### Week 5: Range-Scan Implementation

#### Task 5.1: Prefix Query Logic
**Owner:** Backend Engineer  
**Effort:** 3 days  
**Dependencies:** Task 4.1

**Details:**
- Implement sCIDR-style prefix matching:
  ```python
  def range_scan(prefix_hex: str, prefix_bits: int, limit: int = 100):
      """Query items within a semantic prefix."""
      # Parse prefix
      prefix_bytes = bytes.fromhex(prefix_hex)
      prefix_hi = int.from_bytes(prefix_bytes[:8], 'big', signed=True)
      prefix_lo = int.from_bytes(prefix_bytes[8:], 'big', signed=True)
      
      # Compute range boundaries
      # (mask lower bits based on prefix_bits)
      mask_bits = 128 - prefix_bits
      if mask_bits <= 64:
          # Prefix entirely in hi
          mask = ~((1 << mask_bits) - 1)
          min_hi = prefix_hi & mask
          max_hi = min_hi | ((1 << mask_bits) - 1)
          min_lo = -(1 << 63)  # Full range
          max_lo = (1 << 63) - 1
      else:
          # Prefix spans hi and lo
          # (more complex logic)
      
      # Query
      rows = db.execute("""
          SELECT content_hash, suid_hi, suid_lo, text_content, metadata
          FROM items
          WHERE (suid_hi BETWEEN %s AND %s)
            AND (suid_lo BETWEEN %s AND %s)
          ORDER BY suid_hi, suid_lo
          LIMIT %s
      """, (min_hi, max_hi, min_lo, max_lo, limit))
      
      return rows
  ```

- API endpoint:
  ```python
  @app.get("/hilda/range")
  def range_endpoint(prefix: str, prefix_bits: int = 40, limit: int = 100):
      results = range_scan(prefix, prefix_bits, limit)
      return {"count": len(results), "items": results}
  ```

**Acceptance Criteria:**
- [ ] Correct prefix masking for all bit lengths
- [ ] Query plan uses index (no seq scan)
- [ ] p95 latency <10ms for prefixes returning <1000 items
- [ ] Unit tests for boundary conditions

---

#### Task 5.2: Adaptive Prefix Widening
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 5.1

**Details:**
- Auto-adjust prefix if too sparse or dense:
  ```python
  def adaptive_range_scan(prefix_hex: str, target_count: int = 500,
                          min_bits: int = 24, max_bits: int = 60):
      """Adaptively widen/narrow prefix to hit target count."""
      current_bits = max_bits
      while current_bits >= min_bits:
          results = range_scan(prefix_hex, current_bits, limit=target_count*2)
          if len(results) >= target_count * 0.5:
              return results[:target_count]
          current_bits -= 4  # Coarsen by 4 bits
      
      # Fallback: return what we have
      return results
  ```

**Acceptance Criteria:**
- [ ] Dynamically adjusts to hit target count (±25%)
- [ ] Logs adjustments for monitoring
- [ ] Graceful fallback if no results at coarsest level

---

### Week 6: Hybrid Retrieval (Range + Rerank)

#### Task 6.1: Cosine Rerank Implementation
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Task 5.1

**Details:**
- Fetch embeddings from range results and rerank:
  ```python
  def neighbors(query_text: str, k: int = 20, 
                rerank_budget: int = 500, precision_bits: int = 40):
      """Hybrid retrieval: range-scan + cosine rerank."""
      # 1. Mint HILDA for query
      query_emb = embedder.encode(query_text)
      query_emb /= np.linalg.norm(query_emb)
      hilda_hex = hilda_minter.mint(query_text, precision_bits)
      
      # 2. Range scan
      prefix_hex = hilda_hex[:precision_bits//4]  # Hex chars = bits/4
      candidates = range_scan(prefix_hex, precision_bits, 
                               limit=rerank_budget)
      
      # 3. Rerank by cosine similarity
      candidate_embs = np.array([row['embed'] for row in candidates])
      similarities = candidate_embs @ query_emb
      top_k_indices = np.argsort(similarities)[-k:][::-1]
      
      return [candidates[i] for i in top_k_indices]
  ```

- API endpoint:
  ```python
  @app.post("/hilda/neighbors")
  def neighbors_endpoint(query: str, k: int = 20):
      results = neighbors(query, k)
      return {"query": query, "count": len(results), "neighbors": results}
  ```

**Acceptance Criteria:**
- [ ] Returns top-k most similar items
- [ ] Recall@k measured against exhaustive search
- [ ] p95 latency <150ms (warm cache)
- [ ] Logs: range_count, rerank_count for analysis

---

#### Task 6.2: Caching Layer
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 6.1

**Details:**
- Redis cache for hot prefixes:
  ```python
  def neighbors_cached(query_text: str, k: int = 20):
      # Cache key: hash(query_text, k, precision_bits)
      cache_key = f"neighbors:{hash(query_text)}:{k}"
      
      # Try cache
      cached = redis.get(cache_key)
      if cached:
          return json.loads(cached)
      
      # Compute
      results = neighbors(query_text, k)
      
      # Cache for 1 hour
      redis.setex(cache_key, 3600, json.dumps(results))
      return results
  ```

**Acceptance Criteria:**
- [ ] Cache hit rate >60% on repeated queries
- [ ] TTL configurable per endpoint
- [ ] Cache invalidation on index updates
- [ ] Metrics: hit/miss rate, latency improvement

---

### Sprint 2 Deliverables

**Code:**
- [ ] Postgres schema & migrations
- [ ] Ingestion pipeline & CLI
- [ ] Range-scan API with adaptive widening
- [ ] Neighbors API with rerank
- [ ] Redis caching layer

**Infrastructure:**
- [ ] Postgres instance (Cloud SQL, RDS, or self-hosted)
- [ ] Redis instance
- [ ] Load balancer for API services

**Documentation:**
- [ ] API reference (OpenAPI)
- [ ] Query optimization guide
- [ ] Ingestion runbook

**Tests:**
- [ ] Integration tests: ingest → query
- [ ] Load tests: 100 concurrent users
- [ ] Correctness: recall@k validation

### Sprint 2 Success Criteria
✅ **Latency:** p99 range-scan <15ms, p99 neighbors <150ms  
✅ **Recall:** Recall@10 ≥97% of HNSW baseline on eval set  
✅ **Throughput:** 100 queries/sec sustained  
✅ **Storage:** Successfully indexed 100K+ documents

### Sprint 2 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Index not used (seq scan) | High | Medium | EXPLAIN ANALYZE all queries; tune indexes |
| Poor recall at coarse prefixes | Medium | Medium | Multi-HILDA triangulation (Sprint 4) |
| Hot cells (collision) | Medium | Low | Monitor occupancy; auto-split cells |
| Cache stampede | Low | Medium | Use lock-based cache warming |

---

## Sprint 3: LLM Integration (Weeks 7-9)

**Goal:** Enable LLMs to emit and expand semantic pointers, reducing token usage while maintaining accuracy.

### Week 7: Tool Definitions & Infrastructure

#### Task 7.1: Define LLM Tools (OpenAI Function Schema)
**Owner:** ML Engineer  
**Effort:** 2 days  
**Dependencies:** Sprint 2 complete

**Details:**
- Tool spec for `emit_hilda`:
  ```json
  {
    "name": "emit_hilda",
    "description": "Generate a semantic pointer (HILDA) for a piece of text. Use this to create compact handles for concepts, arguments, or evidence that you may reference later.",
    "parameters": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "The text to encode as a HILDA pointer"
        },
        "precision_bits": {
          "type": "integer",
          "description": "Semantic resolution: 24 (coarse) to 60 (fine)",
          "default": 40
        }
      },
      "required": ["text"]
    }
  }
  ```

- Tool spec for `expand_hilda`:
  ```json
  {
    "name": "expand_hilda",
    "description": "Expand a HILDA pointer to retrieve its content, nearest neighbors, and citations.",
    "parameters": {
      "type": "object",
      "properties": {
        "hilda": {
          "type": "string",
          "description": "The HILDA pointer (hex string)"
        },
        "k": {
          "type": "integer",
          "description": "Number of neighbors to return",
          "default": 5
        }
      },
      "required": ["hilda"]
    }
  }
  ```

**Acceptance Criteria:**
- [ ] Tool schemas validated with OpenAI API
- [ ] Compatible with Claude function calling
- [ ] Examples documented in tool descriptions

---

#### Task 7.2: Tool Implementation Endpoints
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 7.1

**Details:**
- Implement tool handlers:
  ```python
  @app.post("/tools/emit_hilda")
  def tool_emit_hilda(text: str, precision_bits: int = 40):
      hilda_hex = hilda_minter.mint(text, precision_bits)
      explain = explainer.explain(hilda_hex)
      return {
          "hilda": hilda_hex,
          "nearest_words": explain['nearest_words'][:3],
          "message": f"Pointer created: {' '.join(explain['nearest_words'][:3])}"
      }
  
  @app.post("/tools/expand_hilda")
  def tool_expand_hilda(hilda: str, k: int = 5):
      # Get neighbors
      # (inverse mint to recover approx text, then use neighbors API)
      neighbors_list = neighbors_from_hilda(hilda, k)
      
      return {
          "hilda": hilda,
          "neighbors": neighbors_list,
          "summary": f"Retrieved {len(neighbors_list)} related items"
      }
  ```

**Acceptance Criteria:**
- [ ] Tools callable via POST requests
- [ ] Return format matches LLM expectations
- [ ] Error handling for invalid HILDAs
- [ ] Logged for observability

---

#### Task 7.3: Pointer Markup & Middleware
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 7.2

**Details:**
- Define pointer syntax: `⟦HILDA:0123abcd...⟧`
- Middleware to auto-expand pointers in LLM outputs:
  ```python
  def expand_pointers_in_text(text: str):
      """Find and expand HILDA pointers in text."""
      pattern = r'⟦HILDA:([0-9a-f]+)⟧'
      matches = re.findall(pattern, text)
      
      expansions = {}
      for hilda_hex in matches:
          expansion = tool_expand_hilda(hilda_hex, k=3)
          expansions[hilda_hex] = expansion
      
      # Replace pointers with inline expansions
      for hilda_hex, exp in expansions.items():
          summary = f"[{exp['neighbors'][0]['text'][:50]}...]"
          text = text.replace(f"⟦HILDA:{hilda_hex}⟧", summary)
      
      return text, expansions
  ```

**Acceptance Criteria:**
- [ ] Pointers auto-expanded in responses
- [ ] Graceful handling of invalid pointers
- [ ] Expansion details logged for audit

---

### Week 8: Prompt Engineering & Training Data

#### Task 8.1: Few-Shot Prompt Templates
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Task 7.3

**Details:**
- Create prompt templates teaching emit/expand pattern:
  ```
  System: You have access to semantic pointers (HILDA) to compress your reasoning. 
  When you encounter a concept you may reuse, call emit_hilda(text) to create a 
  compact handle. Later, call expand_hilda(hilda) to retrieve details.
  
  Example:
  User: Summarize the argument for climate action in the Paris Agreement.
  
  Assistant: Let me create pointers for the key arguments:
  - [calls emit_hilda("economic benefits of renewable energy transition")]
    → ⟦HILDA:a1b2c3d4⟧ (pointer: economic, renewable, transition)
  - [calls emit_hilda("scientific consensus on anthropogenic warming")]
    → ⟦HILDA:e5f6g7h8⟧ (pointer: scientific, consensus, warming)
  
  The Paris Agreement leverages ⟦HILDA:a1b2c3d4⟧ and ⟦HILDA:e5f6g7h8⟧ to 
  justify urgent policy action...
  
  [When user asks for details, expand pointers]
  ```

- Multi-turn examples:
  - Emit pointers in reasoning
  - Expand pointers when needed
  - Reuse pointers across turns

**Acceptance Criteria:**
- [ ] 20+ diverse few-shot examples
- [ ] Covers: summarization, QA, legal reasoning
- [ ] Validates with GPT-4 / Claude (pointer usage rate >50%)

---

#### Task 8.2: Synthetic Training Data Generation
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Task 8.1

**Details:**
- Generate training traces:
  ```python
  def generate_toolformer_traces(documents: List[str]):
      """Create (input, tool_calls, output) training triples."""
      traces = []
      for doc in documents:
          # Extract key concepts (NER, keyword extraction)
          concepts = extract_concepts(doc)
          
          # Simulate emit calls
          tool_calls = [
              {"tool": "emit_hilda", "args": {"text": concept}}
              for concept in concepts[:3]
          ]
          
          # Simulate reasoning with pointers
          reasoning = create_reasoning_with_pointers(doc, tool_calls)
          
          traces.append({
              "input": doc,
              "tool_calls": tool_calls,
              "output": reasoning
          })
      
      return traces
  ```

- Target: 10K training examples
- Validation: manual review of 100 samples

**Acceptance Criteria:**
- [ ] 10K+ training traces generated
- [ ] Quality score >4/5 on manual review
- [ ] Balanced across domains (general, legal, technical)
- [ ] Uploaded to training data repo

---

#### Task 8.3: Light Fine-Tuning (Optional)
**Owner:** ML Engineer  
**Effort:** 4 days (optional)  
**Dependencies:** Task 8.2

**Details:**
- Fine-tune small model (Llama 3.1 8B, Mistral 7B) on traces
- LoRA/QLoRA for efficiency
- Evaluate:
  - Pointer emission rate
  - Reasoning quality (human eval)
  - Token reduction

**Acceptance Criteria:**
- [ ] Model emits pointers >60% of the time (when beneficial)
- [ ] Task accuracy within 2% of baseline
- [ ] Token reduction ≥25%

**Note:** If fine-tuning is skipped, rely on few-shot prompting + retrieval.

---

### Week 9: Performance Optimization & Caching

#### Task 9.1: Hot Pointer Cache
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 7.2

**Details:**
- Cache expansions for frequently-used HILDAs:
  ```python
  # Pre-warm cache with top-1000 most common pointers
  def warm_cache(top_hildas: List[str]):
      for hilda in top_hildas:
          expansion = tool_expand_hilda(hilda, k=10)
          redis.setex(f"expansion:{hilda}", 3600, json.dumps(expansion))
  ```

- Monitor pointer usage; auto-warm high-traffic HILDAs

**Acceptance Criteria:**
- [ ] Cache hit rate >80% for expansions
- [ ] Latency improvement: 100ms → 10ms (cached)
- [ ] Daily cache warming job

---

#### Task 9.2: Batch Tool Execution
**Owner:** Backend Engineer  
**Effort:** 2 days  
**Dependencies:** Task 7.2

**Details:**
- Support batched emit/expand calls:
  ```python
  @app.post("/tools/batch_emit")
  def batch_emit_hilda(texts: List[str]):
      # Batch embed
      embeddings = embedder.encode_batch(texts)
      
      # Batch mint
      hildas = [hilda_minter.mint_from_embedding(emb) for emb in embeddings]
      
      return [{"text": t, "hilda": h} for t, h in zip(texts, hildas)]
  ```

**Acceptance Criteria:**
- [ ] 10x throughput improvement for batch calls
- [ ] Used in LLM middleware for multi-pointer outputs

---

#### Task 9.3: Semantic Template Registry
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Task 9.1

**Details:**
- Curate high-value pointers with vetted expansions:
  ```json
  {
    "hilda": "a1b2c3d4e5f6...",
    "template": {
      "title": "Climate Action Economic Argument",
      "summary": "The economic case for climate action based on...",
      "citations": ["IPCC 2023", "Stern Review"],
      "key_points": [...]
    }
  }
  ```

- Store in registry; serve on expand with higher priority

**Acceptance Criteria:**
- [ ] 50+ curated templates for common concepts
- [ ] Templates served with <5ms latency
- [ ] Versioned and auditable

---

### Sprint 3 Deliverables

**Code:**
- [ ] LLM tool definitions & handlers
- [ ] Pointer markup middleware
- [ ] Few-shot prompt library
- [ ] Synthetic training data (10K examples)
- [ ] Hot pointer cache
- [ ] Template registry

**Documentation:**
- [ ] LLM integration guide
- [ ] Prompt engineering best practices
- [ ] Tool usage examples

**Tests:**
- [ ] Tool correctness tests
- [ ] End-to-end: emit → expand flow
- [ ] Token reduction validation

### Sprint 3 Success Criteria
✅ **Token Reduction:** ≥25% on target tasks (vs. standard CoT)  
✅ **Quality:** Task accuracy within 2% of baseline (human eval)  
✅ **Latency:** Tool calls add <100ms overhead (p95)  
✅ **Adoption:** LLM emits pointers >50% when beneficial

### Sprint 3 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LLM ignores tools (low uptake) | High | Medium | Stronger few-shot examples; fine-tuning |
| Pointers too coarse (poor expansions) | Medium | Medium | Adaptive precision; multi-HILDA triangulation |
| Cache staleness | Low | Medium | TTL + manual invalidation on updates |
| Training data quality | Medium | Low | Manual review; active learning |

---

## Sprint 4: Advanced Features & Pilot (Weeks 10-12)

**Goal:** Implement multi-HILDA triangulation, launch production pilot, and establish monitoring.

### Week 10: Advanced Retrieval Techniques

#### Task 10.1: Multi-HILDA Triangulation
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Sprint 3 complete

**Details:**
- Emit 2-4 HILDAs per query; intersect neighborhoods:
  ```python
  def triangulated_neighbors(query_text: str, k: int = 20, 
                              num_hildas: int = 3):
      """Emit multiple HILDAs and intersect results."""
      # Generate slight perturbations of query
      queries = [query_text] + perturb_query(query_text, num_hildas-1)
      
      # Mint HILDAs
      hildas = [hilda_minter.mint(q, precision_bits=40) for q in queries]
      
      # Get neighbors for each
      neighbor_sets = [
          set([n['content_hash'] for n in neighbors(q, k=k*2)])
          for q in queries
      ]
      
      # Intersect
      common = neighbor_sets[0].intersection(*neighbor_sets[1:])
      
      # If intersection too small, union instead
      if len(common) < k:
          common = neighbor_sets[0].union(*neighbor_sets[1:])
      
      # Fetch and rerank
      candidates = fetch_by_hashes(common)
      return rerank_by_cosine(query_text, candidates, k)
  ```

**Acceptance Criteria:**
- [ ] Recall improvement: +5% over single-HILDA
- [ ] Latency: <250ms for 3-HILDA triangulation
- [ ] Graceful degradation if intersections empty

---

#### Task 10.2: Product-Quantized (PQ) Handles
**Owner:** ML Engineer  
**Effort:** 3 days  
**Dependencies:** Task 10.1

**Details:**
- Split embeddings into M subspaces; quantize each:
  ```python
  def train_pq_codebooks(embeddings, M=8, k=4096):
      """Train M codebooks for product quantization."""
      D = embeddings.shape[1]
      subspace_dim = D // M
      
      codebooks = []
      for m in range(M):
          subspace = embeddings[:, m*subspace_dim:(m+1)*subspace_dim]
          kmeans = KMeans(n_clusters=k).fit(subspace)
          codebooks.append(kmeans.cluster_centers_)
      
      return codebooks
  
  def encode_pq(embedding, codebooks):
      """Encode embedding as M codes (12 bits each)."""
      codes = []
      M = len(codebooks)
      D = embedding.shape[0]
      subspace_dim = D // M
      
      for m in range(M):
          subspace = embedding[m*subspace_dim:(m+1)*subspace_dim]
          distances = np.linalg.norm(codebooks[m] - subspace, axis=1)
          codes.append(np.argmin(distances))
      
      return codes  # M integers in [0, 4095]
  ```

- Map codes to words: `codes → [word1, word2, ..., wordM]`
- Store PQ codes alongside HILDA for ultra-compact handles

**Acceptance Criteria:**
- [ ] PQ codes trained on codebook embeddings
- [ ] 8-code (96-bit) handles generated
- [ ] Approximate distance preserves ranking (Kendall τ >0.8)

---

### Week 11: Pilot Integration

#### Task 11.1: Choose Pilot Domain & Corpus
**Owner:** Product Manager + ML Engineer  
**Effort:** 2 days  
**Dependencies:** None

**Details:**
- Select domain: legal briefs, medical literature, technical docs
- Curate corpus: 10K-100K documents
- Define use cases:
  - Precedent search
  - Argument template retrieval
  - Deduplication
- Stakeholder interviews: collect requirements

**Acceptance Criteria:**
- [ ] Domain selected and documented
- [ ] Corpus cleaned and formatted
- [ ] Use cases prioritized (top 3)
- [ ] Success metrics defined

---

#### Task 11.2: Index Pilot Corpus
**Owner:** ML Engineer  
**Effort:** 2 days  
**Dependencies:** Task 11.1

**Details:**
- Run ingestion pipeline on pilot corpus
- Generate HILDAs + embeddings + metadata
- Validate data quality:
  - Check for null embeddings
  - Verify HILDA distribution (no hot cells)
  - Spot-check semantic neighborhoods

**Acceptance Criteria:**
- [ ] 100% of corpus indexed
- [ ] HILDA distribution approximately uniform
- [ ] Sample queries return relevant results

---

#### Task 11.3: Deploy Pilot Application
**Owner:** Full Stack Engineer  
**Effort:** 4 days  
**Dependencies:** Task 11.2

**Details:**
- Build simple UI:
  - Search bar: text → HILDA → neighbors
  - Display: results with HILDAs, nearest words
  - Feedback: thumbs up/down on results
- Integrate with existing workflow (if applicable)
- User onboarding: documentation + training

**Acceptance Criteria:**
- [ ] UI deployed and accessible
- [ ] 5+ pilot users onboarded
- [ ] Feedback mechanism active
- [ ] Usage analytics tracked

---

### Week 12: Monitoring, Metrics & Launch

#### Task 12.1: Observability Dashboard
**Owner:** DevOps + Backend Engineer  
**Effort:** 3 days  
**Dependencies:** Sprint 4 tasks

**Details:**
- Metrics to track:
  - **Request counters:** mints, range-scans, expansions
  - **Latency:** p50, p95, p99 per endpoint
  - **Recall:** measured on eval set (daily)
  - **Cache hit rate:** Redis performance
  - **HILDA distribution:** histogram of prefix occupancy
  - **Drift:** KL divergence of prefix distribution (weekly)
  - **User feedback:** thumbs up/down ratio

- Tools: Prometheus + Grafana or Datadog
- Alerts:
  - Latency >200ms (p95)
  - Recall drops >5%
  - Cache hit rate <50%
  - Hot cell detected (>10% of queries in one prefix)

**Acceptance Criteria:**
- [ ] Dashboard live with all metrics
- [ ] Alerts configured and tested
- [ ] Weekly drift report automated
- [ ] Runbook for incident response

---

#### Task 12.2: Security & Compliance Audit
**Owner:** Security Engineer  
**Effort:** 2 days  
**Dependencies:** Task 12.1

**Details:**
- Authorization: enforce at expansion time (not at mint)
- Audit logging: who expanded which HILDA, when
- Data retention: embeddings, expansions, user queries
- Compliance: GDPR, HIPAA (if applicable)
- Rate limiting: prevent abuse
- Input sanitization: prevent injection attacks

**Acceptance Criteria:**
- [ ] Authorization checks pass (penetration test)
- [ ] Audit logs complete and queryable
- [ ] Data retention policy documented
- [ ] Security sign-off obtained

---

#### Task 12.3: Go-Live & Handoff
**Owner:** Project Manager  
**Effort:** 1 day  
**Dependencies:** All Sprint 4 tasks

**Details:**
- Final stakeholder demo
- Handoff documentation:
  - Architecture diagram
  - API reference
  - Runbooks (deployment, scaling, troubleshooting)
  - Evaluation results
- Post-launch plan:
  - Weekly check-ins (first month)
  - Iterative improvements based on feedback
  - Roadmap for v2 features

**Acceptance Criteria:**
- [ ] Demo delivered to stakeholders
- [ ] Handoff docs complete
- [ ] On-call rotation established
- [ ] Post-launch plan approved

---

### Sprint 4 Deliverables

**Code:**
- [ ] Multi-HILDA triangulation
- [ ] PQ-code generation (optional)
- [ ] Pilot application UI
- [ ] Observability dashboard

**Infrastructure:**
- [ ] Production deployment (with failover)
- [ ] Monitoring & alerting
- [ ] Audit logging

**Documentation:**
- [ ] Architecture diagram
- [ ] API reference (complete)
- [ ] Runbooks
- [ ] User guide

**Pilot:**
- [ ] 10K+ documents indexed
- [ ] 5+ users onboarded
- [ ] Feedback collected

### Sprint 4 Success Criteria
✅ **Pilot Launch:** Live with real users, collecting feedback  
✅ **Performance:** All SLAs met (latency, recall, throughput)  
✅ **Monitoring:** Dashboard live, alerts functional  
✅ **Security:** Audit and authorization in place  
✅ **Handoff:** Documentation complete, team trained

### Sprint 4 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Pilot users dissatisfied | High | Low | Frequent check-ins; iterate quickly |
| Performance degradation at scale | Medium | Medium | Load testing; horizontal scaling plan |
| Security incident | High | Low | Security audit; bug bounty program |
| Insufficient monitoring | Medium | Low | Comprehensive metrics; on-call runbook |

---

## Infrastructure Requirements

### Compute
- **Embedder service:** 1 GPU instance (T4, V100) or CPU with batching
  - Estimated cost: $100-300/month
- **API services:** 2-4 CPU instances (2 vCPU, 4GB RAM each)
  - Estimated cost: $100-200/month
- **Batch jobs:** Spot instances for codebook generation
  - One-time cost: $50-100

### Storage
- **Postgres:** 100GB SSD, backups
  - Estimated cost: $50-100/month
- **Redis:** 8GB memory, persistence
  - Estimated cost: $30-50/month
- **Artifact storage:** S3/GCS, 10GB
  - Estimated cost: $5-10/month

### Networking
- **Load balancer:** Application LB
  - Estimated cost: $20-40/month
- **CDN (optional):** For static assets
  - Estimated cost: $10-20/month

### Total Estimated Cost
- **Development:** $300-600/month
- **Production:** $500-1000/month (with redundancy)

---

## Monitoring & Observability

### Key Metrics

**Request Metrics:**
- `hilda_mint_total`: Counter
- `hilda_mint_duration_seconds`: Histogram
- `hilda_range_total`: Counter
- `hilda_range_duration_seconds`: Histogram
- `hilda_neighbors_total`: Counter
- `hilda_neighbors_duration_seconds`: Histogram
- `hilda_expand_total`: Counter

**Quality Metrics:**
- `hilda_recall_at_k`: Gauge (daily eval)
- `hilda_cache_hit_rate`: Gauge
- `hilda_prefix_occupancy`: Histogram

**Drift Metrics:**
- `hilda_prefix_distribution_kl`: Gauge (weekly)
- `hilda_collision_rate`: Gauge

**User Feedback:**
- `hilda_thumbs_up_total`: Counter
- `hilda_thumbs_down_total`: Counter

### Alerting Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| `hilda_neighbors_duration_seconds{quantile=0.95}` | >200ms | Investigate cache misses; scale API |
| `hilda_recall_at_k{k=10}` | <0.92 | Review drift; consider re-indexing |
| `hilda_cache_hit_rate` | <50% | Tune cache TTL; pre-warm hot keys |
| `hilda_prefix_occupancy{bucket=top1%}` | >10% | Implement cell splitting |
| `hilda_collision_rate` | >5% | Increase precision_bits default |

### Dashboards

**Real-Time Dashboard:**
- Request rate (per endpoint)
- Latency distribution (p50, p95, p99)
- Error rate (4xx, 5xx)
- Cache hit rate

**Quality Dashboard:**
- Recall@k trend (daily)
- HILDA distribution heatmap
- Drift KL divergence trend
- User feedback ratio

**Infrastructure Dashboard:**
- CPU/memory utilization
- Database connections
- Redis memory usage
- Network throughput

---

## Decision Gates

### Gate 1: End of Sprint 1
**Criteria:**
- ✅ Deterministic HILDA generation validated
- ✅ Codebook diversity meets minimum threshold (cosine >0.3)
- ✅ All v1 artifacts uploaded and verified

**Decision:**
- **Go:** Proceed to Sprint 2
- **No-Go:** Re-tune codebook parameters; delay 1 week

---

### Gate 2: End of Sprint 2
**Criteria:**
- ✅ Recall@10 ≥95% of HNSW baseline
- ✅ p95 latency <20ms (range-scan), <200ms (neighbors)
- ✅ 10K+ documents indexed successfully

**Decision:**
- **Go:** Proceed to Sprint 3
- **No-Go:** Optimize queries; revisit PCA dimensionality; delay 1-2 weeks

---

### Gate 3: End of Sprint 3
**Criteria:**
- ✅ LLM emits pointers >40% when appropriate
- ✅ Token reduction ≥20% on target tasks
- ✅ Task accuracy within 5% of baseline

**Decision:**
- **Go:** Proceed to Sprint 4
- **No-Go:** Improve prompts; generate more training data; consider fine-tuning

---

### Gate 4: End of Sprint 4 (Launch)
**Criteria:**
- ✅ Pilot users rate system ≥4/5 (usability)
- ✅ All SLAs met for 1 week
- ✅ Security audit passed
- ✅ Monitoring dashboard operational

**Decision:**
- **Go:** Launch to broader user base
- **No-Go:** Address blockers; extend pilot by 2 weeks

---

## Appendix: Task Dependency Graph

```
Sprint 1:
1.1 (Embedder) → 1.2 (CI/CD) → 2.1 (Lexicon)
2.1 → 2.2 (Max-Min) → 2.3 (Artifacts)
2.3 → 3.2 (PCA)
3.1 (Hilbert) → 3.3 (Mint/Explain)
3.2, 3.3 → 3.4 (Codec)

Sprint 2:
Sprint 1 → 4.1 (Schema) → 4.2 (Ingestion) → 4.3 (API)
4.1 → 5.1 (Range-Scan) → 5.2 (Adaptive)
5.1 → 6.1 (Rerank) → 6.2 (Cache)

Sprint 3:
Sprint 2 → 7.1 (Tool Defs) → 7.2 (Tool Impl) → 7.3 (Middleware)
7.3 → 8.1 (Prompts) → 8.2 (Training Data) → 8.3 (Fine-Tune)
7.2 → 9.1 (Cache) → 9.2 (Batch)
9.1 → 9.3 (Templates)

Sprint 4:
Sprint 3 → 10.1 (Triangulation), 10.2 (PQ)
10.1 → 11.1 (Pilot Domain) → 11.2 (Index) → 11.3 (Deploy)
11.3 → 12.1 (Monitoring) → 12.2 (Security) → 12.3 (Go-Live)
```

---

## Team Roles & Responsibilities (Placeholder)

| Role | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 |
|------|----------|----------|----------|----------|
| **ML Engineer** | Codebook, PCA | Rerank, Eval | Prompts, Training | Triangulation, PQ |
| **Backend Engineer** | Hilbert, Codec | Schema, API | Tools, Cache | Monitoring |
| **DevOps** | Docker, CI/CD | Infra, Postgres | Redis, Scaling | Dashboard, Alerts |
| **Full Stack** | - | UI (basic) | LLM Integration | Pilot App |
| **Product Manager** | Requirements | Eval metrics | Use cases | Pilot, Launch |
| **Security** | - | - | AuthZ design | Audit, Compliance |

---

**End of Project Plan**

This plan provides a detailed roadmap for implementing HILDA/SUID over 12 weeks. Adjust timelines and priorities based on team capacity and organizational constraints. Good luck! 🚀
