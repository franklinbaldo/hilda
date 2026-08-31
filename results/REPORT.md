# HILDA representation ablation

recall@10 reachable while the *mean* per-query scan stays inside a
budget. `Nr` is the mean number of separate B-tree ranges that scan
takes; `p95` is the 95th-percentile per-query scan fraction, which is
what a single unlucky query actually pays.

| encoder | mean scan ≤0.5% | mean scan ≤1.0% | mean scan ≤2.0% | mean scan ≤5.0% |
|---|---|---|---|---|
| `hkmeans-L4xK16` | 0.457 (4r · p95 0.5%) | 0.614 (7r · p95 0.8%) | 0.768 (14r · p95 1.5%) | 0.947 (49r · p95 5.3%) |
| `hkmeans-L5xK16` | 0.457 (4r · p95 0.5%) | 0.614 (7r · p95 0.8%) | 0.768 (14r · p95 1.5%) | 0.947 (49r · p95 5.3%) |
| `hkmeans-L4xK8` | 0.454 (4r · p95 0.4%) | 0.619 (7r · p95 0.8%) | 0.753 (13r · p95 1.5%) | 0.938 (47r · p95 5.6%) |
| `hkmeans-L6xK4` | 0.445 (3r · p95 0.5%) | 0.596 (6r · p95 1.1%) | 0.740 (12r · p95 2.0%) | 0.851 (23r · p95 3.8%) |
| `rvq-L4xK16` | 0.372 (7r · p95 1.0%) | 0.476 (11r · p95 1.5%) | 0.608 (20r · p95 2.3%) | 0.749 (5r · p95 5.7%) |
| `rvq-L5xK16` | 0.372 (7r · p95 1.0%) | 0.476 (11r · p95 1.5%) | 0.608 (20r · p95 2.3%) | 0.749 (5r · p95 5.7%) |
| `rvq-L6xK4` | 0.260 (3r · p95 1.6%) | 0.347 (6r · p95 2.2%) | 0.469 (11r · p95 2.7%) | 0.719 (35r · p95 5.2%) |
| `rqvae-L4xK8` | 0.220 (4r · p95 1.2%) | 0.321 (6r · p95 1.8%) | 0.431 (12r · p95 2.6%) | 0.714 (35r · p95 6.4%) |
| `rvq-L4xK8` | 0.249 (4r · p95 0.9%) | 0.453 (11r · p95 1.7%) | 0.582 (20r · p95 2.5%) | 0.709 (33r · p95 3.9%) |
| `ae+rvq-L4xK8` | 0.267 (6r · p95 0.8%) | 0.378 (12r · p95 1.3%) | 0.502 (21r · p95 1.8%) | 0.628 (36r · p95 3.1%) |
| `hilbert-pca4-b15` | 0.107 (5r · p95 1.2%) | 0.165 (8r · p95 1.9%) | 0.254 (13r · p95 3.1%) | 0.469 (4r · p95 7.0%) |
| `morton-pca4-b15` | 0.107 (6r · p95 1.2%) | 0.165 (11r · p95 1.9%) | 0.254 (17r · p95 3.1%) | 0.469 (6r · p95 7.0%) |
| `hilbert-pca3-b20` | 0.076 (4r · p95 1.0%) | 0.125 (6r · p95 1.7%) | 0.205 (11r · p95 2.9%) | 0.372 (2r · p95 7.8%) |
| `morton-pca3-b20` | 0.076 (5r · p95 1.0%) | 0.125 (8r · p95 1.7%) | 0.205 (14r · p95 2.9%) | 0.372 (2r · p95 7.8%) |
| `hilbert-pca2-b30` | 0.034 (3r · p95 0.6%) | 0.055 (9r · p95 1.2%) | 0.103 (3r · p95 2.5%) | 0.298 (6r · p95 7.2%) |
| `morton-pca2-b30` | 0.034 (4r · p95 0.6%) | 0.055 (13r · p95 1.2%) | 0.103 (4r · p95 2.5%) | 0.298 (9r · p95 7.2%) |
| `hilbert-pca2-b30-minmax` | 0.032 (2r · p95 0.8%) | 0.059 (3r · p95 1.6%) | 0.102 (9r · p95 3.1%) | 0.193 (6r · p95 5.9%) |
| `hilbert-rp2-b30` | 0.012±0.003 (3r · p95 0.5%) | 0.022±0.005 (3r · p95 1.0%) | 0.042±0.007 (5r · p95 1.8%) | 0.079±0.013 (2r · p95 3.6%) |

## Run record

- normalised: 1.0000
- pca2_explained_variance: 0.0720
- pca3_explained_variance: 0.0962
- pca4_explained_variance: 0.1153
