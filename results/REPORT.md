# HILDA representation ablation

recall@10 reachable within a scan budget; (Nr) is the mean number of
separate B-tree ranges that scan takes.

| encoder | ≤0.5% scanned | ≤1.0% scanned | ≤2.0% scanned | ≤5.0% scanned |
|---|---|---|---|---|
| `hkmeans-L4xK8` | 0.442 (4r) | 0.603 (7r) | 0.739 (13r) | 0.931 (46r) |
| `hkmeans-L6xK4` | 0.434 (3r) | 0.589 (6r) | 0.733 (12r) | 0.850 (10r) |
| `rvq-L6xK4` | 0.260 (3r) | 0.347 (6r) | 0.469 (11r) | 0.719 (35r) |
| `rvq-L4xK8` | 0.249 (4r) | 0.453 (11r) | 0.582 (20r) | 0.709 (33r) |
| `rqvae-L4xK8` | 0.272 (6r) | 0.393 (11r) | 0.507 (20r) | 0.632 (35r) |
| `hilbert-pca4-b15` | 0.107 (5r) | 0.165 (8r) | 0.254 (13r) | 0.469 (4r) |
| `morton-pca4-b15` | 0.107 (6r) | 0.165 (11r) | 0.254 (17r) | 0.469 (6r) |
| `hilbert-pca3-b20` | 0.076 (4r) | 0.125 (6r) | 0.205 (11r) | 0.372 (2r) |
| `morton-pca3-b20` | 0.076 (5r) | 0.125 (8r) | 0.205 (14r) | 0.372 (2r) |
| `hilbert-pca2-b30` | 0.034 (3r) | 0.055 (9r) | 0.103 (3r) | 0.298 (6r) |
| `morton-pca2-b30` | 0.034 (4r) | 0.055 (13r) | 0.103 (4r) | 0.298 (9r) |
| `hilbert-rp2-b30` | 0.012±0.003 (3r) | 0.022±0.005 (3r) | 0.042±0.007 (5r) | 0.079±0.013 (2r) |

## Run record

- pca2_explained_variance: 0.0720
- pca3_explained_variance: 0.0962
- pca4_explained_variance: 0.1153
