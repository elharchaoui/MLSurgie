# Benchmark results: live-fraud

Seeds: [0]. Track: {'names': 'semantic', 'policy': 'strict'}. Final-test metrics, mean ± sd across seeds.
Baselines are fit on the support split; hyperparameters chosen on development; same preprocessing.

## fraud (binary)

| Method | average_precision ↑ | roc_auc ↑ | log_loss ↓ | accuracy ↑ | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **openhands** (editor) | 0.7458 | 0.8864 | 0.0365 | 0.9938 | 265 |
| prior | 0.0208 | 0.5000 | 0.1012 | 0.9792 | 1 |
| logistic_regression | 0.6675 | 0.8971 | 0.0506 | 0.9879 | 28 |
| random_forest (best baseline) | 0.7797 | 0.8748 | 0.0487 | 0.9925 | 129708 |
| svm_rbf | 0.7210 | 0.8167 | 0.0460 | 0.9921 | 8961 |

