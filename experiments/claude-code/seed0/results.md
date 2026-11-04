# Benchmark results: claude-code-interactive

Seeds: [0]. Track: None. Final-test metrics, mean ± sd across seeds.
Baselines are fit on the support split; hyperparameters chosen on development; same preprocessing.

## fraud (binary)

| Method | average_precision ↑ | roc_auc ↑ | log_loss ↓ | accuracy ↑ | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **interactive_llm** (editor) | 0.7832 | 0.9189 | 0.0355 | 0.9921 | 478 |
| prior | 0.0208 | 0.5000 | 0.1012 | 0.9792 | 1 |
| logistic_regression | 0.6675 | 0.8971 | 0.0506 | 0.9879 | 28 |
| random_forest (best baseline) | 0.7797 | 0.8748 | 0.0487 | 0.9925 | 129708 |
| svm_rbf | 0.7210 | 0.8167 | 0.0460 | 0.9921 | 8961 |

## breast_cancer (binary)

| Method | log_loss ↓ | average_precision ↑ | roc_auc ↑ | accuracy ↑ | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **interactive_llm** (editor) | 0.1159 | 0.9908 | 0.9942 | 0.9478 | 621 |
| prior | 0.6610 | 0.3739 | 0.5000 | 0.6261 | 1 |
| logistic_regression (best baseline) | 0.0799 | 0.9946 | 0.9964 | 0.9565 | 61 |
| random_forest | 0.1388 | 0.9903 | 0.9939 | 0.9565 | 30420 |
| svm_rbf | 0.0933 | 0.9947 | 0.9964 | 0.9739 | 5003 |

## wine (multiclass)

| Method | log_loss ↓ | accuracy ↑ | macro_f1 ↑ | Stored numbers |
|---|---:|---:|---:|---:|
| **interactive_llm** (editor) | 0.0195 | 1.0000 | 1.0000 | 2343 |
| prior | 1.0850 | 0.4054 | 0.1923 | 1 |
| logistic_regression (best baseline) | 0.0264 | 1.0000 | 1.0000 | 81 |
| random_forest | 0.1650 | 1.0000 | 1.0000 | 16914 |
| svm_rbf | 0.0799 | 1.0000 | 1.0000 | 1291 |

## diabetes (regression)

| Method | rmse ↓ | mae ↓ | r2 ↑ | Stored numbers |
|---|---:|---:|---:|---:|
| **interactive_llm** (editor) | 58.8661 | 48.6247 | 0.3025 | 73 |
| mean | 70.4846 | 59.6340 | -0.0000 | 1 |
| ridge | 56.8659 | 45.4222 | 0.3491 | 23 |
| random_forest (best baseline) | 54.1499 | 43.5136 | 0.4098 | 47382 |
| svr_rbf | 57.5097 | 43.7366 | 0.3343 | 6096 |

## segments (clustering)

| Method | adjusted_rand ↑ | silhouette ↑ | Stored numbers |
|---|---:|---:|---:|
| **interactive_llm** (editor) | 0.9812 | 0.4198 | 52 |
| kmeans | 0.9812 | 0.4198 | 48 |
| gaussian_mixture (best baseline) | 1.0000 | 0.4185 | 628 |

## segments_repair (clustering)

| Method | adjusted_rand ↑ | silhouette ↑ | Stored numbers |
|---|---:|---:|---:|
| **interactive_llm** (editor) | 0.9812 | 0.4198 | 52 |
| kmeans | 0.9812 | 0.4198 | 48 |
| gaussian_mixture (best baseline) | 1.0000 | 0.4185 | 628 |

