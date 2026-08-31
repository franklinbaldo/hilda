# HILDA representation ablation

recall@10 reachable while the *mean* per-query scan stays inside a
budget. `Nr` is the mean number of separate B-tree ranges that scan
takes; `p95` is the 95th-percentile per-query scan fraction, which is
what a single unlucky query actually pays.

As in the table below, depth is chosen on the validation queries and
reported on the test queries.

| encoder | mean scan ≤0.5% | mean scan ≤1.0% | mean scan ≤2.0% | mean scan ≤5.0% |
|---|---|---|---|---|
| `hkmeans-L4xK16` | 0.504 (4r · p95 0.5%) | 0.662 (7r · p95 0.9%) | 0.802 (13r · p95 1.6%) | 0.958 (49r · p95 5.2%) |
| `hkmeans-L5xK16` | 0.504 (4r · p95 0.5%) | 0.662 (7r · p95 0.9%) | 0.802 (13r · p95 1.6%) | 0.958 (49r · p95 5.2%) |
| `hkmeans-L4xK8` | 0.489 (3r · p95 0.4%) | 0.646 (7r · p95 0.8%) | 0.779 (13r · p95 1.5%) | 0.943 (47r · p95 5.5%) |
| `hkmeans-L6xK4` | 0.466 (3r · p95 0.6%) | 0.627 (6r · p95 1.1%) | 0.750 (12r · p95 2.1%) | 0.867 (24r · p95 3.8%) |
| `rvq-L4xK16` | 0.383 (6r · p95 1.1%) | 0.488 (11r · p95 1.6%) | 0.615 (20r · p95 2.4%) | 0.759 (5r · p95 5.9%) |
| `rvq-L5xK16` | 0.383 (6r · p95 1.1%) | 0.488 (11r · p95 1.6%) | 0.615 (20r · p95 2.4%) | 0.759 (5r · p95 5.9%) |
| `rvq-L4xK8` | 0.256 (3r · p95 0.8%) | 0.491 (11r · p95 1.7%) | 0.626 (19r · p95 2.5%) | 0.748 (33r · p95 3.8%) |
| `rvq-L6xK4` | 0.205 (2r · p95 1.2%) | 0.392 (6r · p95 2.0%) | 0.511 (12r · p95 2.6%) | 0.747 (36r · p95 5.0%) |
| `rqvae-L4xK8` | 0.231 (3r · p95 0.7%) | 0.332 (6r · p95 1.2%) | 0.590 (19r · p95 2.9%) | 0.726 (31r · p95 5.1%) |
| `ae+rvq-L4xK8` | 0.282 (7r · p95 0.9%) | 0.395 (12r · p95 1.3%) | 0.523 (21r · p95 2.1%) | 0.633 (36r · p95 3.2%) |
| `hilbert-pca4-b15` | 0.124 (5r · p95 1.2%) | 0.191 (8r · p95 2.0%) | 0.266 (12r · p95 3.1%) | 0.507 (5r · p95 7.1%) |
| `morton-pca4-b15` | 0.124 (6r · p95 1.2%) | 0.191 (11r · p95 2.0%) | 0.266 (17r · p95 3.1%) | 0.507 (6r · p95 7.1%) |
| `hilbert-pca3-b20` | 0.082 (4r · p95 1.0%) | 0.131 (6r · p95 1.7%) | 0.208 (11r · p95 2.9%) | 0.370 (2r · p95 7.8%) |
| `morton-pca3-b20` | 0.082 (5r · p95 1.0%) | 0.131 (9r · p95 1.7%) | 0.208 (15r · p95 2.9%) | 0.370 (2r · p95 7.8%) |
| `hilbert-pca2-b30` | 0.036 (3r · p95 0.6%) | 0.068 (8r · p95 1.1%) | 0.111 (3r · p95 2.3%) | 0.295 (6r · p95 7.2%) |
| `morton-pca2-b30` | 0.036 (4r · p95 0.6%) | 0.068 (13r · p95 1.1%) | 0.111 (4r · p95 2.3%) | 0.295 (9r · p95 7.2%) |
| `hilbert-pca2-b30-minmax` | 0.034 (2r · p95 0.8%) | 0.061 (3r · p95 1.7%) | 0.110 (9r · p95 3.2%) | 0.197 (6r · p95 6.0%) |
| `hilbert-rp2-b30` | 0.015±0.003 (3r · p95 0.5%) | 0.028±0.003 (3r · p95 1.0%) | 0.048±0.007 (5r · p95 1.9%) | 0.084±0.008 (2r · p95 3.6%) |

## At a per-query candidate budget

Every encoder spends the same candidates on *every* query: cells are
visited nearest-first and the boundary cell is truncated in index
order, the way a range scan with a LIMIT would. No averaging hides an
expensive query here.

Depth is chosen on the validation queries and reported on the held-out
test queries, so the number is not selected on what it reports. Only
operating points that filled the budget on every query are eligible.

| encoder | 40 cand (0.5%) | 80 cand (1.0%) | 160 cand (2.0%) | 400 cand (5.0%) |
|---|---|---|---|---|
| `hkmeans-L4xK16` | 0.648 (8r) | 0.776 (15r) | 0.876 (24r) | 0.960 (56r) |
| `hkmeans-L5xK16` | 0.648 (8r) | 0.776 (15r) | 0.876 (24r) | 0.960 (56r) |
| `hkmeans-L4xK8` | 0.635 (7r) | 0.765 (12r) | 0.867 (24r) | 0.955 (54r) |
| `hkmeans-L6xK4` | 0.595 (6r) | 0.710 (11r) | 0.820 (20r) | 0.917 (43r) |
| `rvq-L4xK16` | 0.481 (23r) | 0.631 (46r) | 0.778 (95r) | 0.878 (55r) |
| `rvq-L5xK16` | 0.479 (23r) | 0.631 (46r) | 0.778 (95r) | 0.878 (55r) |
| `rvq-L4xK8` | 0.349 (6r) | 0.507 (11r) | 0.694 (21r) | 0.864 (52r) |
| `rvq-L6xK4` | 0.272 (5r) | 0.443 (10r) | 0.641 (19r) | 0.849 (52r) |
| `rqvae-L4xK8` | 0.305 (6r) | 0.453 (11r) | 0.637 (21r) | 0.834 (49r) |
| `ae+rvq-L4xK8` | 0.325 (9r) | 0.480 (17r) | 0.645 (32r) | 0.830 (72r) |
| `hilbert-pca4-b15` | 0.133 (27r) | 0.222 (52r) | — | 0.541 (43r) |
| `morton-pca4-b15` | 0.134 (28r) | 0.222 (53r) | — | 0.542 (52r) |
| `hilbert-pca3-b20` | 0.091 (6r) | 0.151 (11r) | 0.246 (18r) | 0.453 (37r) |
| `morton-pca3-b20` | 0.089 (8r) | 0.152 (13r) | 0.249 (22r) | 0.453 (45r) |
| `hilbert-pca2-b30-minmax` | 0.064 (9r) | 0.100 (16r) | 0.175 (7r) | 0.356 (13r) |
| `morton-pca2-b30` | 0.065 (32r) | 0.104 (12r) | 0.171 (69r) | 0.334 (37r) |
| `hilbert-pca2-b30` | 0.065 (32r) | 0.099 (3r) | 0.161 (1r) | 0.337 (3r) |
| `hilbert-rp2-b30` | 0.016±0.004 (2r) | 0.032±0.006 (24r) | 0.057±0.005 (29r) | 0.125±0.012 (82r) |

## Run record

- corpus_size: 8000.0000
- normalised: 1.0000
- pca2_explained_variance: 0.0720
- pca3_explained_variance: 0.0962
- pca4_explained_variance: 0.1153
- test_queries: 200.0000
- validation_queries: 200.0000
