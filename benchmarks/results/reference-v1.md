<!-- Regenerated after PR review: independent historical repair fixture (seed 29), behavior-preserving preprocessing conversion, bounded-memory silhouette. Earlier repair numbers used held-out labels and are superseded. -->

# Benchmark results: mlsurgie-classical-v1

Seeds: [0, 1, 2]. Track: {'names': 'semantic', 'policy': 'strict'}. Final-test metrics, mean ± sd across seeds.
Baselines are fit on the support split; hyperparameters chosen on development; same preprocessing.

## fraud (binary)

| Method | average_precision ↑ | roc_auc ↑ | log_loss ↓ | accuracy ↑ | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **search_baseline** (editor) | 0.2021 ± 0.1236 | 0.6291 ± 0.1303 | 0.1025 ± 0.0089 | 0.9792 ± 0.0000 | 28 |
| prior | 0.0208 ± 0.0000 | 0.5000 ± 0.0000 | 0.1012 ± 0.0000 | 0.9792 ± 0.0000 | 1 |
| logistic_regression | 0.6336 ± 0.0326 | 0.8800 ± 0.0149 | 0.0550 ± 0.0042 | 0.9865 ± 0.0017 | 28 |
| random_forest (best baseline) | 0.7550 ± 0.0371 | 0.8686 ± 0.0172 | 0.0668 ± 0.0174 | 0.9926 ± 0.0006 | 182500 |
| svm_rbf | 0.7118 ± 0.0194 | 0.8424 ± 0.0338 | 0.0457 ± 0.0013 | 0.9913 ± 0.0011 | 9306 |

## breast_cancer (binary)

| Method | log_loss ↓ | average_precision ↑ | roc_auc ↑ | accuracy ↑ | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **search_baseline** (editor) | 0.6250 ± 0.1141 | 0.5867 ± 0.3189 | 0.6508 ± 0.3094 | 0.7072 ± 0.1406 | 61 |
| prior | 0.6610 ± 0.0000 | 0.3739 ± 0.0000 | 0.5000 ± 0.0000 | 0.6261 ± 0.0000 | 1 |
| logistic_regression | 0.0728 ± 0.0179 | 0.9973 ± 0.0027 | 0.9982 ± 0.0018 | 0.9768 ± 0.0181 | 61 |
| random_forest | 0.1195 ± 0.0194 | 0.9919 ± 0.0024 | 0.9950 ± 0.0015 | 0.9623 ± 0.0050 | 25680 |
| svm_rbf (best baseline) | 0.0668 ± 0.0247 | 0.9968 ± 0.0021 | 0.9978 ± 0.0015 | 0.9768 ± 0.0050 | 4576 |

## wine (multiclass)

| Method | log_loss ↓ | accuracy ↑ | macro_f1 ↑ | Stored numbers |
|---|---:|---:|---:|---:|
| **search_baseline** (editor) | 1.0778 ± 0.2080 | 0.5045 ± 0.1023 | 0.4574 ± 0.1138 | 81 |
| prior | 1.0850 ± 0.0000 | 0.4054 ± 0.0000 | 0.1923 ± 0.0000 | 1 |
| logistic_regression (best baseline) | 0.0636 ± 0.0441 | 0.9820 ± 0.0156 | 0.9834 ± 0.0143 | 81 |
| random_forest | 0.1576 ± 0.0175 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 16034 |
| svm_rbf | 0.0792 ± 0.0048 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1384 |

## diabetes (regression)

| Method | rmse ↓ | mae ↓ | r2 ↑ | Stored numbers |
|---|---:|---:|---:|---:|
| **search_baseline** (editor) | 72.6573 ± 11.1391 | 60.4729 ± 10.2907 | -0.0377 ± 0.2677 | 23 |
| mean | 71.8559 ± 1.8333 | 61.2871 ± 1.7180 | -0.0082 ± 0.0088 | 1 |
| ridge | 56.4234 ± 0.4424 | 45.9182 ± 0.6011 | 0.3775 ± 0.0329 | 23 |
| random_forest (best baseline) | 56.1131 ± 1.8940 | 46.1483 ± 2.6200 | 0.3840 ± 0.0509 | 47394 |
| svr_rbf | 58.5784 ± 1.3627 | 45.8428 ± 1.8535 | 0.3286 ± 0.0524 | 6096 |

## segments (clustering)

| Method | adjusted_rand ↑ | silhouette ↑ | Stored numbers |
|---|---:|---:|---:|
| **search_baseline** (editor) | 0.4591 ± 0.0381 | 0.2584 ± 0.0903 | 52 |
| kmeans | 0.9774 ± 0.0152 | 0.4163 ± 0.0164 | 48 |
| gaussian_mixture (best baseline) | 0.9967 ± 0.0058 | 0.4155 ± 0.0166 | 628 |

## segments_repair (clustering)

| Method | adjusted_rand ↑ | silhouette ↑ | Stored numbers |
|---|---:|---:|---:|
| **search_baseline** (editor) | 0.3597 ± 0.0156 | 0.0471 ± 0.0121 | 52 |
| kmeans | 0.9774 ± 0.0152 | 0.4163 ± 0.0164 | 48 |
| gaussian_mixture (best baseline) | 0.9967 ± 0.0058 | 0.4155 ± 0.0166 | 628 |

