# HILDA/SUID Evaluation Framework

**Comprehensive Testing & Validation Protocol**

**Version:** 1.0  
**Last Updated:** October 15, 2025  
**Status:** Planning Phase

---

## Overview

### Evaluation Philosophy

This evaluation framework validates the HILDA/SUID system across five dimensions:
1. **Codebook Quality:** Diversity, robustness, human usability
2. **Retrieval Performance:** Recall, latency, cost vs. vector indexes
3. **Clustering & Deduplication:** Semantic coherence via sCIDR
4. **LLM Integration:** Token efficiency, accuracy, consistency
5. **System Performance:** Scalability, stability, operational metrics

### Timeline Alignment

| Sprint | Week | Experiments |
|--------|------|-------------|
| Sprint 1 | 1-3 | Experiment 1: Codebook Quality |
| Sprint 2 | 4-6 | Experiments 2-3: Retrieval Performance, Deduplication |
| Sprint 3 | 7-9 | Experiment 4: LLM Token Efficiency |
| Sprint 4 | 10-12 | Experiments 5-6: System Benchmarks, Drift Analysis |

### Baseline Systems

All experiments compare HILDA against:
- **FAISS IVF+PQ:** Industry-standard approximate nearest neighbor (ANN)
- **HNSW (hnswlib):** Graph-based ANN with strong recall
- **Exhaustive Cosine:** Ground truth (when computationally feasible)
- **Standard RAG:** LLM with full-text retrieval (no pointers)

---

## Experiment 1: Codebook Quality Evaluation

**Timeline:** Sprint 1 (Weeks 1-3)  
**Owner:** ML Engineer  
**Status:** 🟡 Planned

### 1.1 Diversity Metrics

**Objective:** Verify that the 4096-word codebook provides good coverage of semantic space.

**Metrics:**

1. **Pairwise Cosine Distance:**
   - Minimum: `min_{i≠j} d(w_i, w_j)` where `d(w_i, w_j) = 1 - cos(w_i, w_j)`
   - Mean: `μ_d`
   - Median: `median_d`
   - **Target:** min >0.25, mean >0.50

2. **Coverage Score:**
   - Sample 10K random vectors from same distribution as candidates
   - For each, find nearest codebook word
   - Measure: `max_distance = max_i min_j d(v_i, w_j)`
   - **Target:** <0.40 (no vector too far from codebook)

3. **Entropy of Selection:**
   - Measure distribution of selected words across POS tags
   - **Target:** Shannon entropy >3.0 bits (diverse POS)

**Methodology:**

```python
def evaluate_codebook_diversity(codebook_embeddings):
    # Pairwise distances
    gram = codebook_embeddings @ codebook_embeddings.T
    distances = 1 - gram
    np.fill_diagonal(distances, np.inf)  # Exclude self
    
    min_dist = distances.min()
    mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()
    median_dist = np.median(distances[np.triu_indices_from(distances, k=1)])
    
    # Coverage
    random_samples = generate_random_embeddings(10000)
    nearest_dists = []
    for v in random_samples:
        dists = 1 - (codebook_embeddings @ v)
        nearest_dists.append(dists.min())
    max_distance = max(nearest_dists)
    
    return {
        'min_distance': min_dist,
        'mean_distance': mean_dist,
        'median_distance': median_dist,
        'coverage_max_distance': max_distance
    }
```

**Success Criteria:**
- ✅ Minimum pairwise distance ≥0.25
- ✅ Mean pairwise distance ≥0.50
- ✅ Coverage max distance <0.40

---

### 1.2 Robustness Testing

**Objective:** Validate codec resilience to speech/typo errors.

**Sub-Experiments:**

#### 1.2.1 Speech Simulation

**Setup:**
- Use phoneme confusion matrices (e.g., from CMU Sphinx)
- Simulate mishearings: "bear" → "bare", "night" → "knight"
- Test 1000 randomly generated 11-word sequences

**Metrics:**
- **Word Error Rate (WER):** % of words incorrectly snapped
- **Sequence Recovery Rate:** % of sequences correctly decoded (with CRC)
- **Target:** WER <5%, Recovery ≥90%

**Methodology:**

```python
def simulate_speech_errors(words, phoneme_confusion_prob=0.1):
    """Simulate speech recognition errors."""
    corrupted = []
    for word in words:
        if random.random() < phoneme_confusion_prob:
            # Apply phoneme confusion
            corrupted_word = apply_phoneme_confusion(word)
        else:
            corrupted_word = word
        corrupted.append(corrupted_word)
    return corrupted

def evaluate_speech_robustness(codec, codebook, n_trials=1000):
    errors = []
    recoveries = 0
    
    for _ in range(n_trials):
        # Generate random UUID
        uuid_bytes = os.urandom(16)
        words = codec.encode_to_words(uuid_bytes)
        
        # Simulate speech errors
        corrupted = simulate_speech_errors(words)
        
        # Decode with semantic snapping
        try:
            decoded_uuid = codec.decode_from_phrases(corrupted)
            if decoded_uuid == uuid_bytes.hex():
                recoveries += 1
            else:
                errors.append('CRC failed')
        except Exception as e:
            errors.append(str(e))
    
    wer = sum(1 for e in errors if 'mismatch' in e) / (n_trials * 11)
    recovery_rate = recoveries / n_trials
    
    return {'wer': wer, 'recovery_rate': recovery_rate}
```

**Success Criteria:**
- ✅ WER <5%
- ✅ Recovery rate ≥90% (with CRC error detection)

---

#### 1.2.2 Typo Simulation

**Setup:**
- Simulate keyboard typos: insertions, deletions, substitutions, transpositions
- Edit distance 1-3 from original word
- Test on 1000 sequences

**Metrics:**
- **Semantic Snapping Accuracy:** % of typos correctly snapped to intended word
- **Target:** >85% for edit distance ≤2

**Methodology:**

```python
def simulate_typos(word, edit_distance=1):
    """Generate typos at given edit distance."""
    typos = []
    
    # Deletions
    for i in range(len(word)):
        typos.append(word[:i] + word[i+1:])
    
    # Insertions
    for i in range(len(word) + 1):
        for c in 'abcdefghijklmnopqrstuvwxyz':
            typos.append(word[:i] + c + word[i:])
    
    # Substitutions
    for i in range(len(word)):
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c != word[i]:
                typos.append(word[:i] + c + word[i+1:])
    
    # Transpositions
    for i in range(len(word) - 1):
        typos.append(word[:i] + word[i+1] + word[i] + word[i+2:])
    
    return typos[:min(10, len(typos))]  # Sample

def evaluate_typo_robustness(embedder, codebook, n_trials=1000):
    correct_snaps = 0
    total_typos = 0
    
    for _ in range(n_trials):
        # Pick random codebook word
        word = random.choice(codebook.words)
        word_idx = codebook.word_to_idx[word]
        
        # Generate typos
        typos = simulate_typos(word, edit_distance=2)
        
        for typo in typos:
            # Embed and snap
            typo_emb = embedder.encode(typo)
            nearest_idx = codebook.snap_to_nearest(typo_emb)
            
            if nearest_idx == word_idx:
                correct_snaps += 1
            total_typos += 1
    
    accuracy = correct_snaps / total_typos
    return {'typo_snapping_accuracy': accuracy}
```

**Success Criteria:**
- ✅ Typo snapping accuracy ≥85% (edit distance ≤2)

---

### 1.3 Human Usability Study

**Objective:** Assess memorability and transmission accuracy in real-world conditions.

**Setup:**
- Recruit 20 participants (diverse backgrounds)
- Each memorizes 3 sequences of 11 words
- Tasks:
  1. **Recall:** After 5 minutes, 1 hour, 24 hours
  2. **Phone transmission:** Read sequence to partner over phone; partner writes down
  3. **Voice assistant:** Speak sequence to Siri/Google Assistant; measure transcription accuracy

**Metrics:**
- **Memorability score:** % sequences recalled correctly
- **Phone transmission accuracy:** % words correctly transmitted
- **Voice assistant WER:** Word error rate

**Methodology:**

```
Study Protocol:
1. Training phase (10 min): Participants learn 3 sequences
2. Immediate recall (5 min later)
3. Distractor task (1 hour)
4. Delayed recall (1 hour later)
5. Phone transmission (pairs)
6. Voice assistant test
7. Post-study survey: ease of memorization (1-5 Likert)

Analysis:
- Recall: % exact matches, % with ≤1 error
- Transmission: WER, sequence recovery rate
- Voice: WER, CRC validation rate
- Survey: mean ease score
```

**Success Criteria:**
- ✅ Immediate recall ≥80%
- ✅ Delayed recall (1h) ≥60%
- ✅ Phone transmission WER <10%
- ✅ Voice assistant WER <15%
- ✅ Ease score ≥3.5/5

---

### Experiment 1 Summary

| Metric | Target | Measurement |
|--------|--------|-------------|
| Min pairwise distance | ≥0.25 | Computed from codebook embeddings |
| Mean pairwise distance | ≥0.50 | Computed from codebook embeddings |
| Coverage max distance | <0.40 | 10K random samples |
| Speech WER | <5% | 1000 simulated sequences |
| Speech recovery rate | ≥90% | With CRC validation |
| Typo snapping accuracy | ≥85% | Edit distance ≤2 |
| Human immediate recall | ≥80% | n=20, 3 sequences each |
| Human delayed recall | ≥60% | After 1 hour |
| Phone transmission WER | <10% | Pairs over phone |
| Voice assistant WER | <15% | Siri/Google |

---

## Experiment 2: Retrieval Performance

**Timeline:** Sprint 2 (Weeks 4-6)  
**Owner:** ML Engineer + Backend Engineer  
**Status:** 🟡 Planned

### 2.1 Range-Scan vs. Vector Index Baselines

**Objective:** Compare HILDA range-scan + rerank against FAISS and HNSW.

#### Datasets

1. **MS MARCO Passages (dev set)**
   - 8.8M passages, 6,980 queries
   - Use 100K passage subset for initial testing
   - Metrics: MRR@10, Recall@100, Recall@1000

2. **BEIR Benchmark (subset)**
   - Select 3 diverse tasks: SciFact, NFCorpus, FiQA
   - ~10K-30K docs per task
   - Metrics: nDCG@10, Recall@100

3. **Custom Domain Corpus**
   - Legal briefs (10K docs) OR medical abstracts (20K docs)
   - 200 manually curated queries with relevance labels
   - Metrics: Precision@10, Recall@50, MAP

**Baseline Configurations:**

```python
# FAISS IVF+PQ
import faiss

d = 384  # embedding dimension
nlist = 100  # number of clusters
m = 8  # number of subquantizers
nbits = 8  # bits per subquantizer

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10  # search 10 clusters

# HNSW
import hnswlib

index = hnswlib.Index(space='cosine', dim=d)
index.init_index(max_elements=100000, ef_construction=200, M=16)
index.add_items(embeddings)
index.set_ef(50)  # search ef
```

**HILDA Configuration:**

```python
# HILDA range-scan + rerank
hilda_config = {
    'precision_bits': 40,  # sCIDR prefix length
    'rerank_budget': 500,  # candidates before rerank
    'adaptive': True  # auto-adjust prefix if needed
}
```

---

#### 2.1.1 Recall@k Evaluation

**Metrics:**
- Recall@k for k ∈ {1, 5, 10, 20, 50, 100}
- Recall@k = (# relevant docs in top-k) / (total # relevant docs)

**Methodology:**

```python
def evaluate_recall(queries, ground_truth, retrieval_fn, k_values):
    """Compute Recall@k for various k."""
    recalls = {k: [] for k in k_values}
    
    for query, relevant_docs in zip(queries, ground_truth):
        retrieved = retrieval_fn(query, k=max(k_values))
        retrieved_ids = [doc['id'] for doc in retrieved]
        
        for k in k_values:
            top_k = retrieved_ids[:k]
            hits = len(set(top_k) & set(relevant_docs))
            recall_at_k = hits / len(relevant_docs) if relevant_docs else 0
            recalls[k].append(recall_at_k)
    
    # Average across queries
    return {f'recall@{k}': np.mean(recalls[k]) for k in k_values}

# Run for each system
results = {}
for system_name, retrieval_fn in systems.items():
    results[system_name] = evaluate_recall(
        queries, ground_truth, retrieval_fn, 
        k_values=[1, 5, 10, 20, 50, 100]
    )
```

**Success Criteria:**
- ✅ HILDA Recall@10 ≥97% of HNSW baseline
- ✅ HILDA Recall@100 ≥95% of HNSW baseline

---

#### 2.1.2 Latency Benchmarking

**Metrics:**
- Query latency: p50, p95, p99 (milliseconds)
- Measure both cold and warm cache scenarios

**Methodology:**

```python
def benchmark_latency(retrieval_fn, queries, n_trials=3, warmup=10):
    """Measure query latency."""
    latencies = []
    
    # Warm-up
    for _ in range(warmup):
        retrieval_fn(random.choice(queries), k=20)
    
    # Benchmark
    for query in queries:
        for _ in range(n_trials):
            start = time.perf_counter()
            retrieval_fn(query, k=20)
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'mean': np.mean(latencies)
    }
```

**Success Criteria:**
- ✅ HILDA p99 latency <150ms (warm cache)
- ✅ HILDA p95 latency <100ms (warm cache)
- ✅ Cold cache latency <500ms

---

#### 2.1.3 Throughput Testing

**Metrics:**
- Queries per second (QPS) sustained
- Measure under concurrent load: 1, 10, 50, 100 users

**Methodology:**

```python
import asyncio
from locust import HttpUser, task, between

class SearchUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def search(self):
        query = random.choice(self.queries)
        self.client.post("/hilda/neighbors", json={"query": query, "k": 20})

# Run load test
# locust -f load_test.py --users 100 --spawn-rate 10 --host http://api.hilda.com
```

**Success Criteria:**
- ✅ Sustained 100 QPS at p95 latency <200ms
- ✅ No degradation after 1 hour continuous load

---

#### 2.1.4 Storage Overhead

**Metrics:**
- Storage per document: HILDA vs. FAISS vs. HNSW
- Index build time

**Methodology:**

```python
def measure_storage(system, n_docs=100000):
    """Measure storage requirements."""
    # Baseline: raw embeddings (float32)
    raw_size = n_docs * 384 * 4  # bytes
    
    # HILDA: 128-bit ID + 384-dim embedding (optional)
    hilda_size = n_docs * (16 + 384 * 4)  # with embeddings for rerank
    hilda_size_compact = n_docs * 16  # ID only
    
    # FAISS IVF+PQ: compressed vectors
    faiss_size = estimate_faiss_size(n_docs, m=8, nbits=8)
    
    # HNSW: graph + vectors
    hnsw_size = estimate_hnsw_size(n_docs, M=16, d=384)
    
    return {
        'raw': raw_size,
        'hilda': hilda_size,
        'hilda_compact': hilda_size_compact,
        'faiss': faiss_size,
        'hnsw': hnsw_size
    }
```

**Success Criteria:**
- ✅ HILDA storage competitive with FAISS (<2x overhead)
- ✅ Index build time <10 min for 100K docs

---

### 2.2 Ablation Studies

**Objective:** Understand impact of key hyperparameters.

#### 2.2.1 Prefix Length (sCIDR bits)

**Setup:**
- Vary `precision_bits` ∈ {24, 30, 40, 50, 60}
- Measure Recall@10 and latency
- Fixed rerank budget: N=500

**Expected Results:**
- Lower bits (24-30): faster but lower recall (coarse cells)
- Higher bits (50-60): better recall but slower (fine cells)
- Optimal: ~40 bits (trade-off)

**Success Criteria:**
- ✅ Identify optimal precision_bits for each dataset
- ✅ Document precision vs. recall curve

---

#### 2.2.2 Rerank Budget (N)

**Setup:**
- Vary rerank budget N ∈ {128, 256, 512, 1024, 2048}
- Fixed precision_bits = 40
- Measure Recall@10 and latency

**Expected Results:**
- Higher N: better recall, higher latency
- Diminishing returns after N>512

**Success Criteria:**
- ✅ Identify knee of the curve (optimal N)
- ✅ N=500 achieves ≥97% of exhaustive recall

---

#### 2.2.3 PCA Dimensions

**Setup:**
- Train PCA with 2, 3, 4, 5 dimensions
- Measure Recall@10 (all else equal)
- Note: 3D+ requires changes to Hilbert curve

**Expected Results:**
- 2D: fast but may lose some structure
- 3D: better, but more complex indexing
- 4D+: marginal gains

**Success Criteria:**
- ✅ 2D achieves >95% of 3D performance (if not, consider upgrading)

---

### 2.3 Cost Analysis

**Objective:** Compare $ per 1M queries across systems.

**Metrics:**

| System | Storage Cost | Query Cost (compute) | Total $ per 1M queries |
|--------|-------------|----------------------|------------------------|
| HILDA | Postgres + embeddings | API + rerank | $X |
| FAISS (cloud) | Compressed index | GPU inference | $Y |
| HNSW (cloud) | Graph index | CPU inference | $Z |
| Pinecone (SaaS) | Managed | Pay-per-query | $W |

**Methodology:**
- Use cloud pricing (AWS, GCP, Azure)
- Assume 100K docs, 1M queries/month
- Include index build + storage + query costs

**Success Criteria:**
- ✅ HILDA cost-competitive with FAISS (<1.5x)
- ✅ Cheaper than managed SaaS (Pinecone) by >30%

---

### Experiment 2 Summary

| Metric | Target | Baseline Comparison |
|--------|--------|---------------------|
| Recall@10 | ≥97% of HNSW | HNSW, FAISS, Exhaustive |
| Recall@100 | ≥95% of HNSW | HNSW, FAISS |
| p99 latency (warm) | <150ms | HNSW, FAISS |
| Throughput | 100 QPS | HNSW, FAISS |
| Storage overhead | <2x FAISS | FAISS, HNSW |
| Cost per 1M queries | <1.5x FAISS | FAISS, Pinecone |

---

## Experiment 3: Deduplication & Clustering

**Timeline:** Sprint 2-3 (Weeks 5-7)  
**Owner:** ML Engineer  
**Status:** 🟡 Planned

### 3.1 Near-Duplicate Detection

**Objective:** Use sCIDR bucketing to identify near-duplicates efficiently.

**Setup:**
- Dataset: 20K documents with 10% synthetically injected near-duplicates
- Near-duplicates: paraphrases, minor edits (80-95% similarity)
- Baseline: pairwise cosine with threshold (expensive)

**Metrics:**
- **Precision:** Of pairs in same sCIDR cell, % truly near-duplicates
- **Recall:** Of true near-duplicates, % captured in same cell
- **F1 score:** Harmonic mean of precision and recall

**Methodology:**

```python
def evaluate_deduplication(documents, ground_truth_pairs, precision_bits=40):
    """Evaluate sCIDR-based deduplication."""
    # Mint HILDAs
    hildas = [mint_hilda(doc, precision_bits) for doc in documents]
    
    # Group by sCIDR prefix
    cells = defaultdict(list)
    for i, hilda in enumerate(hildas):
        prefix = extract_prefix(hilda, precision_bits)
        cells[prefix].append(i)
    
    # Extract candidate pairs from cells
    candidate_pairs = set()
    for cell, doc_ids in cells.items():
        for i in range(len(doc_ids)):
            for j in range(i+1, len(doc_ids)):
                candidate_pairs.add((doc_ids[i], doc_ids[j]))
    
    # Evaluate
    true_pairs = set(ground_truth_pairs)
    tp = len(candidate_pairs & true_pairs)
    fp = len(candidate_pairs - true_pairs)
    fn = len(true_pairs - candidate_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

**Success Criteria:**
- ✅ Precision ≥70% (few false positives per cell)
- ✅ Recall ≥85% (captures most duplicates)
- ✅ F1 ≥0.75

---

### 3.2 Cluster Coherence

**Objective:** Measure semantic coherence within sCIDR cells.

**Setup:**
- Cluster documents by sCIDR prefix (40 bits)
- Compute intra-cluster and inter-cluster similarities

**Metrics:**
- **Silhouette Score:** `s = (b - a) / max(a, b)` where:
  - `a` = mean intra-cluster distance
  - `b` = mean nearest-cluster distance
- **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster variance

**Methodology:**

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_clustering(embeddings, labels):
    """Evaluate sCIDR clustering quality."""
    silhouette = silhouette_score(embeddings, labels, metric='cosine')
    ch_score = calinski_harabasz_score(embeddings, labels)
    
    return {'silhouette': silhouette, 'ch_score': ch_score}

# Label = sCIDR prefix for each document
labels = [extract_prefix(mint_hilda(doc, 40), 40) for doc in documents]
results = evaluate_clustering(embeddings, labels)
```

**Success Criteria:**
- ✅ Silhouette score ≥0.3 (reasonable clustering)
- ✅ CH score > baseline (random prefixes)

---

### Experiment 3 Summary

| Metric | Target | Comparison |
|--------|--------|------------|
| Dedup precision | ≥70% | Cosine threshold baseline |
| Dedup recall | ≥85% | Cosine threshold baseline |
| Dedup F1 | ≥0.75 | - |
| Silhouette score | ≥0.3 | Random prefix baseline |
| CH score | > baseline | Random prefix baseline |

---

## Experiment 4: LLM Token Efficiency

**Timeline:** Sprint 3 (Weeks 7-9)  
**Owner:** ML Engineer + Full Stack Engineer  
**Status:** 🟡 Planned

### 4.1 Pointer-Based Reasoning Experiments

**Objective:** Measure token reduction and quality when LLMs use semantic pointers.

#### Tasks

1. **Multi-Hop Question Answering**
   - Dataset: HotpotQA (1000 examples)
   - Baseline: Standard chain-of-thought (CoT)
   - Treatment: Emit HILDAs for sub-questions, expand on demand

2. **Legal Brief Summarization**
   - Dataset: 100 legal briefs (custom)
   - Baseline: Full RAG (retrieve + summarize)
   - Treatment: Pointer-based (emit for precedents, expand selectively)

3. **Argument Decomposition**
   - Dataset: 200 complex arguments
   - Baseline: Full text reasoning
   - Treatment: Emit HILDAs for sub-claims, reuse across turns

---

#### 4.1.1 Token Count Reduction

**Metrics:**
- **Input tokens:** Prompt + context
- **Output tokens:** LLM response
- **Total tokens:** Input + output
- **Reduction:** `(Baseline - Treatment) / Baseline * 100%`

**Methodology:**

```python
def measure_token_reduction(task_data, baseline_fn, treatment_fn):
    """Compare token usage."""
    results = []
    
    for example in task_data:
        # Baseline
        baseline_response = baseline_fn(example)
        baseline_tokens = count_tokens(baseline_response['prompt'] + 
                                       baseline_response['output'])
        
        # Treatment (with pointers)
        treatment_response = treatment_fn(example)
        treatment_tokens = count_tokens(treatment_response['prompt'] + 
                                        treatment_response['output'])
        
        reduction = (baseline_tokens - treatment_tokens) / baseline_tokens
        results.append({
            'example_id': example['id'],
            'baseline_tokens': baseline_tokens,
            'treatment_tokens': treatment_tokens,
            'reduction_pct': reduction * 100
        })
    
    return pd.DataFrame(results)
```

**Success Criteria:**
- ✅ Mean token reduction ≥25%
- ✅ Median token reduction ≥20%
- ✅ At least 80% of examples show reduction

---

#### 4.1.2 Task Accuracy

**Metrics:**
- **Multi-hop QA:** Exact match (EM), F1 score
- **Summarization:** ROUGE-L, human eval (fluency, completeness)
- **Argument decomposition:** Coverage of key claims, logical coherence (human eval)

**Methodology:**

```python
def evaluate_qa_accuracy(predictions, ground_truth):
    """Evaluate QA accuracy."""
    em_scores = []
    f1_scores = []
    
    for pred, true in zip(predictions, ground_truth):
        em = 1 if normalize(pred) == normalize(true) else 0
        f1 = compute_f1(pred, true)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores)
    }
```

**Success Criteria:**
- ✅ Accuracy within 2% of baseline (no quality degradation)
- ✅ For QA: EM ≥baseline - 0.02
- ✅ For summarization: ROUGE-L ≥baseline - 0.03

---

#### 4.1.3 Latency

**Metrics:**
- **End-to-end latency:** Including tool calls (emit, expand)
- **Overhead:** Latency added by tool calls

**Methodology:**

```python
def measure_e2e_latency(task_data, baseline_fn, treatment_fn):
    """Measure end-to-end latency."""
    baseline_latencies = []
    treatment_latencies = []
    
    for example in task_data:
        # Baseline
        start = time.perf_counter()
        baseline_fn(example)
        baseline_latencies.append(time.perf_counter() - start)
        
        # Treatment
        start = time.perf_counter()
        treatment_fn(example)
        treatment_latencies.append(time.perf_counter() - start)
    
    return {
        'baseline_p95': np.percentile(baseline_latencies, 95),
        'treatment_p95': np.percentile(treatment_latencies, 95),
        'overhead': np.mean(treatment_latencies) - np.mean(baseline_latencies)
    }
```

**Success Criteria:**
- ✅ Tool call overhead <100ms per pointer (p95)
- ✅ End-to-end latency competitive (within 10% of baseline)

---

#### 4.1.4 Consistency

**Objective:** Verify that same input produces same pointers (determinism).

**Metrics:**
- **Pointer stability:** Run same query 10 times; measure % identical pointers
- **Target:** >95% identical

**Methodology:**

```python
def measure_consistency(queries, emit_fn, n_trials=10):
    """Measure pointer consistency."""
    consistency_scores = []
    
    for query in queries:
        pointers = [emit_fn(query) for _ in range(n_trials)]
        
        # Count unique pointers
        unique_pointers = len(set(pointers))
        consistency = 1 - (unique_pointers - 1) / (n_trials - 1)
        consistency_scores.append(consistency)
    
    return np.mean(consistency_scores)
```

**Success Criteria:**
- ✅ Pointer stability ≥95%

---

### 4.2 Human Evaluation

**Objective:** Qualitative assessment of pointer-based reasoning.

**Setup:**
- Judge panel: 3 evaluators (domain experts)
- 100 examples per task (sampled)
- Blind comparison: Baseline vs. Treatment (random order)

**Evaluation Criteria (5-point Likert scale):**
1. **Coherence:** Is the reasoning logical and easy to follow?
2. **Correctness:** Does the output answer the question accurately?
3. **Efficiency:** Is the response concise without sacrificing quality?

**Methodology:**

```
Evaluation Form:
- Question ID: ___
- Response A: [baseline or treatment]
- Response B: [treatment or baseline]

Rate each response (1-5):
- Coherence: ___ / 5
- Correctness: ___ / 5
- Efficiency: ___ / 5

Which response do you prefer overall? A / B / Tie
```

**Success Criteria:**
- ✅ Mean scores within 0.3 points of baseline
- ✅ Preference: Treatment ≥40% (not worse than baseline)

---

### Experiment 4 Summary

| Metric | Target | Task |
|--------|--------|------|
| Token reduction | ≥25% | All tasks (mean) |
| Accuracy (QA EM) | ≥baseline - 2% | HotpotQA |
| Accuracy (ROUGE-L) | ≥baseline - 3% | Summarization |
| Latency overhead | <100ms | Per pointer (p95) |
| Pointer stability | ≥95% | Consistency test |
| Human eval (coherence) | ≥4.0/5 | 100 examples |
| Human eval (correctness) | ≥4.0/5 | 100 examples |

---

## Experiment 5: System Performance Benchmarks

**Timeline:** Sprint 4 (Weeks 10-12)  
**Owner:** DevOps + Backend Engineer  
**Status:** 🟡 Planned

### 5.1 End-to-End Load Testing

**Objective:** Validate system performance under realistic load.

**Setup:**
- Simulate concurrent users: 10, 100, 1000
- Request mix:
  - 40% `/hilda/mint`
  - 40% `/hilda/range`
  - 20% `/hilda/neighbors`
- Duration: 30 minutes per load level
- Ramp-up: 10 users/second

**Metrics:**
- **Throughput:** Requests per second (RPS)
- **Latency:** p50, p95, p99 per endpoint
- **Error rate:** % 4xx/5xx responses
- **Resource utilization:** CPU, memory, disk I/O

**Methodology:**

```python
# Use Locust for load testing
from locust import HttpUser, task, between

class HildaUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(4)
    def mint(self):
        self.client.post("/hilda/mint", json={
            "text": self.generate_random_text(),
            "precision_bits": 40
        })
    
    @task(4)
    def range_scan(self):
        hilda = self.get_random_hilda()
        self.client.get(f"/hilda/range?prefix={hilda[:10]}&limit=100")
    
    @task(2)
    def neighbors(self):
        self.client.post("/hilda/neighbors", json={
            "query": self.generate_random_text(),
            "k": 20
        })

# Run: locust -f load_test.py --users 1000 --spawn-rate 10
```

**Success Criteria:**
- ✅ At 100 concurrent users:
  - Throughput ≥100 RPS
  - p95 latency <200ms
  - Error rate <1%
- ✅ At 1000 concurrent users:
  - Throughput ≥500 RPS
  - p95 latency <500ms
  - Error rate <5%

---

### 5.2 Cost Analysis

**Objective:** Calculate operational cost per 1M queries.

**Components:**
1. **Compute:** API servers, embedder service
2. **Storage:** Postgres, Redis, S3/GCS
3. **Networking:** Data transfer, load balancer
4. **Overhead:** Monitoring, logging

**Methodology:**

```python
def calculate_monthly_cost(queries_per_month=1_000_000, docs=100_000):
    """Estimate monthly costs."""
    # API servers (2x 2vCPU, 4GB RAM)
    api_cost = 2 * 30 * 0.05  # $0.05/hour
    
    # Embedder (1x GPU or 4vCPU)
    embedder_cost = 1 * 30 * 0.20  # $0.20/hour (CPU) or $1.50 (GPU)
    
    # Postgres (100GB SSD)
    db_cost = 50  # managed DB
    
    # Redis (8GB)
    cache_cost = 30
    
    # Storage (artifacts + embeddings)
    storage_cost = (docs * 384 * 4 / 1e9) * 0.02  # $0.02/GB/month
    
    # Networking
    data_transfer = queries_per_month * 10e-6  # 10KB per query
    network_cost = data_transfer * 0.12  # $0.12/GB
    
    # Total
    total = api_cost + embedder_cost + db_cost + cache_cost + storage_cost + network_cost
    cost_per_1m_queries = total / (queries_per_month / 1e6)
    
    return {
        'monthly_total': total,
        'cost_per_1m_queries': cost_per_1m_queries,
        'breakdown': {
            'api': api_cost,
            'embedder': embedder_cost,
            'database': db_cost,
            'cache': cache_cost,
            'storage': storage_cost,
            'network': network_cost
        }
    }
```

**Success Criteria:**
- ✅ Total monthly cost (1M queries, 100K docs) <$500
- ✅ Cost per 1M queries <$50
- ✅ Cheaper than managed alternatives (Pinecone) by ≥30%

---

### 5.3 Stress Testing

**Objective:** Identify breaking points and failure modes.

**Tests:**
1. **Database Overload:** Max connections, query timeouts
2. **Cache Eviction:** Redis memory pressure
3. **Embedder Saturation:** Batch queue buildup
4. **Disk Space:** Index growth without cleanup

**Methodology:**
- Gradually increase load until failure
- Monitor: queue depths, connection pools, error logs
- Document: failure mode, recovery time

**Success Criteria:**
- ✅ Graceful degradation (no cascading failures)
- ✅ Recovery within 2 minutes after load reduction
- ✅ Clear error messages (not 500s)

---

### Experiment 5 Summary

| Metric | Target | Load Level |
|--------|--------|------------|
| Throughput (100 users) | ≥100 RPS | Sustained |
| p95 latency (100 users) | <200ms | All endpoints |
| Error rate | <1% | 100 users |
| Throughput (1000 users) | ≥500 RPS | Sustained |
| p95 latency (1000 users) | <500ms | All endpoints |
| Monthly cost (1M queries) | <$500 | 100K docs |
| Cost per 1M queries | <$50 | - |

---

## Experiment 6: Drift & Stability Analysis

**Timeline:** Sprint 4 + Ongoing (Weeks 10-12+)  
**Owner:** ML Engineer + DevOps  
**Status:** 🟡 Planned

### 6.1 Temporal Stability

**Objective:** Measure HILDA consistency over time with fixed embedder.

**Setup:**
- Mint HILDAs for 1000 documents at T0
- Re-mint same documents at T0 + 1 week, T0 + 1 month
- No model or corpus changes

**Metrics:**
- **ID stability:** % of documents with identical HILDA
- **Prefix stability:** % with identical prefix (40 bits)
- **Neighborhood stability:** Overlap in top-20 neighbors

**Methodology:**

```python
def measure_temporal_stability(documents, intervals=['1week', '1month']):
    """Measure HILDA stability over time."""
    t0_hildas = [mint_hilda(doc, precision_bits=60) for doc in documents]
    
    results = {'t0': t0_hildas}
    stability_metrics = {}
    
    for interval in intervals:
        # Wait, then re-mint
        time.sleep(interval_to_seconds(interval))
        tn_hildas = [mint_hilda(doc, precision_bits=60) for doc in documents]
        results[interval] = tn_hildas
        
        # Compare
        exact_match = sum(1 for h1, h2 in zip(t0_hildas, tn_hildas) if h1 == h2)
        prefix_match = sum(1 for h1, h2 in zip(t0_hildas, tn_hildas) 
                           if h1[:10] == h2[:10])  # 40-bit prefix
        
        stability_metrics[interval] = {
            'exact_match': exact_match / len(documents),
            'prefix_match': prefix_match / len(documents)
        }
    
    return stability_metrics
```

**Success Criteria:**
- ✅ Exact HILDA match ≥99.5% (1 week)
- ✅ Prefix match ≥99.9% (1 week)
- ✅ Exact HILDA match ≥98% (1 month)

---

### 6.2 Distribution Shift Detection

**Objective:** Monitor for corpus drift that may require re-indexing.

**Setup:**
- Collect prefix histogram weekly
- Compute KL divergence: `KL(P_week || P_baseline)`
- Alert if KL > threshold

**Metrics:**
- **KL divergence:** Measure distribution change
- **Collision rate:** % of documents in top-1% of cells

**Methodology:**

```python
def monitor_distribution_shift(current_hildas, baseline_hildas, precision_bits=40):
    """Detect distribution shift."""
    # Extract prefixes
    current_prefixes = [extract_prefix(h, precision_bits) for h in current_hildas]
    baseline_prefixes = [extract_prefix(h, precision_bits) for h in baseline_hildas]
    
    # Compute histograms
    current_hist, _ = np.histogram(current_prefixes, bins=2**precision_bits)
    baseline_hist, _ = np.histogram(baseline_prefixes, bins=2**precision_bits)
    
    # Normalize
    current_dist = current_hist / current_hist.sum()
    baseline_dist = baseline_hist / baseline_hist.sum()
    
    # KL divergence
    kl_div = entropy(current_dist, baseline_dist)
    
    # Collision rate (top-1% cells)
    top_1pct_threshold = np.percentile(current_hist, 99)
    collision_rate = (current_hist > top_1pct_threshold).mean()
    
    return {
        'kl_divergence': kl_div,
        'collision_rate': collision_rate
    }
```

**Alerting Rules:**
```yaml
alerts:
  - name: HighDrift
    condition: kl_divergence > 0.1
    action: notify_team
    message: "Significant distribution shift detected. Consider re-indexing."
  
  - name: HotCell
    condition: collision_rate > 0.05
    action: auto_split_cells
    message: "Hot cells detected. Applying finer precision."
```

**Success Criteria:**
- ✅ KL divergence <0.05 under normal operation
- ✅ Alerts trigger correctly on synthetic shift

---

### 6.3 Versioning & Migration Testing

**Objective:** Validate smooth transition when upgrading embedder or PCA.

**Setup:**
- Deploy v2 artifacts (new embedder or PCA)
- Dual-write: mint HILDAs for both v1 and v2
- Compare neighborhoods, measure disruption

**Metrics:**
- **Neighborhood overlap:** Jaccard similarity of top-20 neighbors (v1 vs v2)
- **User impact:** % queries requiring fallback to v1

**Methodology:**

```python
def test_version_migration(documents, v1_minter, v2_minter):
    """Test v1 → v2 migration."""
    v1_hildas = [v1_minter.mint(doc) for doc in documents]
    v2_hildas = [v2_minter.mint(doc) for doc in documents]
    
    overlaps = []
    for doc in documents[:100]:  # Sample
        v1_neighbors = get_neighbors_v1(doc, k=20)
        v2_neighbors = get_neighbors_v2(doc, k=20)
        
        overlap = len(set(v1_neighbors) & set(v2_neighbors)) / 20
        overlaps.append(overlap)
    
    return {
        'mean_overlap': np.mean(overlaps),
        'min_overlap': np.min(overlaps)
    }
```

**Success Criteria:**
- ✅ Mean neighborhood overlap ≥70% (acceptable disruption)
- ✅ Migration runbook tested and documented

---

### Experiment 6 Summary

| Metric | Target | Interval |
|--------|--------|----------|
| Exact HILDA match | ≥99.5% | 1 week |
| Prefix match | ≥99.9% | 1 week |
| Exact HILDA match | ≥98% | 1 month |
| KL divergence | <0.05 | Normal operation |
| Collision rate | <5% | Weekly check |
| Version migration overlap | ≥70% | v1 → v2 |

---

## Evaluation Timeline (Gantt Chart)

```
Week 1-3 (Sprint 1):
├── Exp 1: Codebook Quality
│   ├── 1.1 Diversity Metrics [Week 1]
│   ├── 1.2 Robustness Testing [Week 2]
│   └── 1.3 Human Usability Study [Week 3]

Week 4-6 (Sprint 2):
├── Exp 2: Retrieval Performance
│   ├── 2.1 Baseline Comparison [Week 4-5]
│   ├── 2.2 Ablation Studies [Week 5]
│   └── 2.3 Cost Analysis [Week 6]
├── Exp 3: Deduplication (start)
│   └── 3.1 Near-Duplicate Detection [Week 6]

Week 7-9 (Sprint 3):
├── Exp 3: Deduplication (finish)
│   └── 3.2 Cluster Coherence [Week 7]
├── Exp 4: LLM Token Efficiency
│   ├── 4.1 Token Reduction [Week 7-8]
│   └── 4.2 Human Evaluation [Week 9]

Week 10-12 (Sprint 4):
├── Exp 5: System Benchmarks
│   ├── 5.1 Load Testing [Week 10-11]
│   ├── 5.2 Cost Analysis [Week 11]
│   └── 5.3 Stress Testing [Week 12]
├── Exp 6: Drift Analysis (start)
│   ├── 6.1 Temporal Stability [Week 10+]
│   ├── 6.2 Distribution Shift [Week 11+]
│   └── 6.3 Migration Testing [Week 12]

Ongoing (Post-Sprint 4):
└── Exp 6: Drift Monitoring [Weekly]
```

---

## Reporting

### Weekly Metrics Dashboard

**Components:**
- Request counters (mint, range, neighbors)
- Latency distributions (p50, p95, p99)
- Cache hit rates
- Error rates
- HILDA distribution histogram
- User feedback (thumbs up/down)

**Tools:** Grafana + Prometheus or Datadog

---

### Sprint-End Evaluation Reports

**Format:**
- Executive summary (1 page)
- Key metrics: tables + charts
- Comparison to targets (✅/❌)
- Identified issues + mitigations
- Recommendations for next sprint

**Distribution:** Email to stakeholders + uploaded to project wiki

---

### Final Comprehensive Evaluation (Week 12)

**Contents:**
1. **System Overview:** Architecture, components, versions
2. **Experiment Results:** All 6 experiments, detailed findings
3. **Comparison Matrix:** HILDA vs. baselines (all metrics)
4. **Cost-Benefit Analysis:** $ savings, performance trade-offs
5. **Limitations & Future Work:** Known issues, roadmap
6. **Appendices:** Raw data, code, screenshots

**Format:** PDF report (20-30 pages) + slide deck (15-20 slides)

---

## Appendix A: Datasets

### MS MARCO Passages
- **Source:** https://microsoft.github.io/msmarco/
- **Size:** 8.8M passages, 6,980 dev queries
- **Subset:** 100K passages for testing
- **Preprocessing:** None (use as-is)

### BEIR Benchmark
- **Source:** https://github.com/beir-cellar/beir
- **Tasks:** SciFact, NFCorpus, FiQA
- **Size:** 10K-30K docs each
- **Preprocessing:** Standard BEIR tokenization

### HotpotQA
- **Source:** https://hotpotqa.github.io/
- **Size:** 1000 dev examples
- **Preprocessing:** Extract question + supporting facts

### Custom Legal Corpus
- **Source:** Public court opinions (e.g., CourtListener)
- **Size:** 10K briefs, 200 manual queries
- **Preprocessing:** OCR + chunking (500 tokens/chunk)

---

## Appendix B: Baseline Implementations

### FAISS (IVF+PQ)
```python
import faiss

d = 384
nlist = 100
m = 8
nbits = 8

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10
```

### HNSW (hnswlib)
```python
import hnswlib

index = hnswlib.Index(space='cosine', dim=384)
index.init_index(max_elements=100000, ef_construction=200, M=16)
index.add_items(embeddings, ids=range(len(embeddings)))
index.set_ef(50)
```

### Versions
- FAISS: 1.7.4
- hnswlib: 0.7.0
- sentence-transformers: 2.2.2

---

## Appendix C: Statistical Testing

### Significance Tests

**Paired t-test:** For comparing HILDA vs. baseline on same queries
```python
from scipy.stats import ttest_rel

# Null hypothesis: HILDA and baseline have same mean recall
t_stat, p_value = ttest_rel(hilda_recalls, baseline_recalls)
if p_value < 0.05:
    print("Significant difference detected")
```

**Bootstrapping:** For confidence intervals
```python
from scipy.stats import bootstrap

def recall_mean(recalls):
    return np.mean(recalls)

ci = bootstrap((hilda_recalls,), recall_mean, n_resamples=10000, 
               confidence_level=0.95)
print(f"95% CI: [{ci.confidence_interval.low}, {ci.confidence_interval.high}]")
```

**Multiple Comparisons:** Bonferroni correction
```python
alpha = 0.05
num_tests = 6  # 6 experiments
adjusted_alpha = alpha / num_tests  # 0.0083
```

---

**End of Evaluation Framework**

This comprehensive evaluation plan ensures rigorous validation of the HILDA/SUID system across all critical dimensions. Execute experiments in parallel where possible, and iterate based on findings. Good luck! 📊
