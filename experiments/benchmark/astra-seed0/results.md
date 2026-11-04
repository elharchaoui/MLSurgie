# Benchmark results: live-astra

Seeds: [0]. Track: {'names': 'semantic', 'policy': 'strict'}. Final-test metrics, mean ± sd across seeds.
Baselines are fit on the support split; hyperparameters chosen on development; same preprocessing.

## breast_cancer (binary)

| Method | log_loss ↓ | average_precision ↑ | roc_auc ↑ | accuracy ↑ | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **openhands** (editor) | 0.1370 | 0.9894 | 0.9926 | 0.9391 | 4403 |
| prior | 0.6610 | 0.3739 | 0.5000 | 0.6261 | 1 |
| logistic_regression (best baseline) | 0.0799 | 0.9946 | 0.9964 | 0.9565 | 61 |
| random_forest | 0.1388 | 0.9903 | 0.9939 | 0.9565 | 30420 |
| svm_rbf | 0.0933 | 0.9947 | 0.9964 | 0.9739 | 5003 |

## wine (multiclass)

| Method | log_loss ↓ | accuracy ↑ | macro_f1 ↑ | Stored numbers |
|---|---:|---:|---:|---:|
| **openhands** (editor) | 0.0075 | 1.0000 | 1.0000 | 7413 |
| prior | 1.0850 | 0.4054 | 0.1923 | 1 |
| logistic_regression (best baseline) | 0.0264 | 1.0000 | 1.0000 | 81 |
| random_forest | 0.1650 | 1.0000 | 1.0000 | 16914 |
| svm_rbf | 0.0799 | 1.0000 | 1.0000 | 1291 |

## diabetes (regression)

| Method | rmse ↓ | mae ↓ | r2 ↑ | Stored numbers |
|---|---:|---:|---:|---:|
| **openhands** (editor) | 57.8487 | 46.2935 | 0.3264 | 23 |
| mean | 70.4846 | 59.6340 | -0.0000 | 1 |
| ridge | 56.8659 | 45.4222 | 0.3491 | 23 |
| random_forest (best baseline) | 54.1499 | 43.5136 | 0.4098 | 47382 |
| svr_rbf | 57.5097 | 43.7366 | 0.3343 | 6096 |

## segments (clustering)

| Method | adjusted_rand ↑ | silhouette ↑ | Stored numbers |
|---|---:|---:|---:|
| **openhands** (editor) | 0.9812 | 0.4198 | 52 |
| kmeans | 0.9812 | 0.4198 | 48 |
| gaussian_mixture (best baseline) | 1.0000 | 0.4185 | 628 |

## segments_repair (clustering)

| Method | adjusted_rand ↑ | silhouette ↑ | Stored numbers |
|---|---:|---:|---:|
| **openhands** (editor) | 0.9812 | 0.4198 | 52 |
| kmeans | 0.9812 | 0.4198 | 48 |
| gaussian_mixture (best baseline) | 1.0000 | 0.4185 | 628 |

