# Benchmark data provenance

These CSV snapshots are committed so a fresh clone can run the suite without downloads.
`checksums.json` records their SHA-256 hashes. Regenerate them from the repository root:

```bash
python benchmarks/make_datasets.py
python benchmarks/make_repair.py
```

Generation requires the `baselines` extra. The current snapshots were generated with
NumPy 2.5.4 and scikit-learn 1.9.1. Numeric values in the public datasets are rounded
to six decimal places; column names and target labels are adapted by the generator.

| File | Rows | Origin |
|---|---:|---|
| `fraud.csv` | 12,000 | Repository synthetic transaction generator, seed 7 |
| `segments.csv` | 1,600 | Repository synthetic customer generator, seed 11 |
| `breast_cancer.csv` | 569 | scikit-learn `load_breast_cancer`; UCI Wisconsin Diagnostic Breast Cancer, Wolberg, Street and Mangasarian |
| `wine.csv` | 178 | scikit-learn `load_wine`; UCI Wine recognition, Forina et al. |
| `diabetes.csv` | 442 | scikit-learn `load_diabetes(scaled=False)`; Efron, Hastie, Johnstone and Tibshirani, *Least Angle Regression* (2004) |

Source descriptions: [breast cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html),
[wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html),
[diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

The repair model is generated separately from **4,000 historical synthetic customers,
seed 29**. It does not read these benchmark CSVs. Its historical preprocessing is
saved beside the initial program and rebased to each run's support encoding using an
affine coordinate change. Historical reference labels are used to construct that
fixture; benchmark reference labels are used only for the final adjusted Rand score.
