# Benchmarking weight-space programming

[← Overview](../README.md) · [Agent harness](agent-harness.md) · [Technical reference](reference.md)

MLSurgie turns "an LLM wrote some weights" into a measured experiment that an ML engineer can compare against the models they already use. This page defines the protocol, the comparison classes (tracks), the task suite, the controls, and how to read the results.

## The protocol, formally

A run is a tuple **(T, E, B, F, S, K)**:

| Symbol | Component | Declared in |
|---|---|---|
| T | Task: dataset, task type, target/reference, features, selection metric | `benchmarks/tasks/*.json` → `Goal`, `Feature` |
| E | Editor: who writes weight programs (agent, JSON LLM, search control, fixture) | `--editor`, editor `identity` |
| B | Budget: iterations, parameters, width, depth, seconds, tokens, rows | `Budget` |
| F | Feedback policy: what one development evaluation reveals | `FeedbackPolicy` |
| S | Selection rule: metric and minimum improvement to replace the incumbent | `SelectionRule` |
| K | Track: semantic/anonymized names × strict/open fitting policy | `Track` |

Execution:

1. Split rows 60/20/20 into **support / development / test** (stratified for classification; random for regression and clustering; disjoint groups when `group_column` is set).
2. Fit preprocessing on support only. This covers standardization, missing-value flags and one-hot encoding.
3. Each iteration, the editor submits one **weight program**. That's the IR: a JSON object of explicit tensor assignments, validated against the architecture and budget. The protocol has no code path that fits weights.
4. A separate process evaluates the network on the development rows. The editor receives what **F** allows: metrics, plus up to *k* worst-error rows, under a cumulative cap on revealed labels.
5. **S** decides whether the candidate replaces the best.
6. After the last iteration, selection is frozen (`selection.json`). The test rows are then evaluated exactly once, and the network, preprocessing, programs, diffs and checksums are exported. `mlsurgie replay` rebuilds the network from the saved programs alone.

The editor never sees development or test rows, except the development rows **F** explicitly reveals. For clustering, benchmark reference labels are hidden even by name and are used only in the final report. The repair fixture was constructed using labels from a separate historical sample.

## Task types and heads

| Task | Network output | Selection metrics | Reported |
|---|---|---|---|
| binary | 1 logit → sigmoid | log loss, Brier, ROC-AUC, average precision, accuracy, balanced accuracy, F1 | plus confusion matrix and Wilson interval |
| multiclass | one logit per class → softmax | log loss, accuracy, balanced accuracy, macro-F1 | confusion matrix |
| regression | 1 value in target units | RMSE, MAE, R² | target mean/std |
| clustering | one score per cluster → argmax | silhouette, inertia (unsupervised) | adjusted Rand index against the hidden reference |

Hidden layers can use `relu`, `tanh`, `sigmoid` or `identity` activations.

## Tracks: which results are comparable

| Track | Meaning |
|---|---|
| `--names semantic` | Real feature names, descriptions and category values. The editor can use domain knowledge and anything it memorized about the dataset. |
| `--names anonymized` | Features renamed `f01…`, categories `v1…`, classes `c1…`, no descriptions, generic goal text. The gap between semantic and anonymized results measures reliance on priors. |
| `--policy strict` | All executed code and every file in the agent workspace is statically audited for fitting signatures: learning libraries, solvers, `scipy.optimize`, gradient updates, `.fit(`. A flagged candidate is rejected and still costs its iteration. |
| `--policy open` | Flags are recorded in the report; the result is labeled, not rejected. |

The audit is a tripwire, not a proof. An agent could implement fitting the audit doesn't recognize, so every source file is retained for review. Compare results only within the same track.

## Controls and baselines

Every suite cell trains predeclared reference models on **the same split and the same encoded inputs**:

| Task | Baselines (scikit-learn) | Trivial |
|---|---|---|
| binary / multiclass | logistic regression (C ∈ {0.1, 1, 10}), random forest (300 trees, min leaf ∈ {1, 5}), RBF SVM (C ∈ {0.3, 1, 3}) | prior |
| regression | ridge (α ∈ {0.1, 1, 10}), random forest, RBF SVR (C ∈ {10, 100, 1000}) | mean |
| clustering | k-means (10 restarts), Gaussian mixture | — |

Baselines are fit on support, choose their hyperparameters on development with the task's selection metric, and are scored once on test. The table also reports **stored numbers**, so a 300-parameter network can be compared honestly with a 100,000-node forest.

`RandomSearchEditor` is the non-LLM control. It hill-climbs over the same IR with the same number of development queries, and has no reasoning. If an LLM editor doesn't beat it, the LLM added nothing beyond evaluation-guided search.

## The suite (`benchmarks/suite.json`)

| Task | Type | Source | What it tests |
|---|---|---|---|
| `fraud` | binary, 2% positives | synthetic, 12,000 transactions | Rare events; conjunctive fraud scenarios (bust-out, cross-border card-not-present, card testing); 20% camouflaged fraud. Selected by average precision. |
| `breast_cancer` | binary | UCI WDBC (scikit-learn copy) | Many correlated numeric features |
| `wine` | multiclass (3) | UCI wine (scikit-learn copy) | Prototype or per-class evidence construction |
| `diabetes` | regression | Efron et al. (scikit-learn copy) | Additive numeric response |
| `segments` | clustering (4) | synthetic retail customers | Unsupervised structure, including an irrelevant feature (age) |
| `segments_repair` | clustering, repair | same, starting from a faulty deployed network | Diagnosis and minimal patching of a model with two planted faults |

The CSV snapshots and their [provenance and checksums](../benchmarks/data/README.md) are committed. Synthetic sets are regenerated by `benchmarks/make_datasets.py` and `benchmarks/make_repair.py`; their planted structure can't appear in any pretraining corpus. The repair starting model uses independent historical customers (seed 29); the evaluation sample uses seed 11. Its saved historical preprocessing is converted to each support encoding without fitting, preserving the imported model’s raw-input behavior. The real sets are familiar public benchmarks, so treat semantic-track results on them as possibly contaminated.

## Running

```bash
pip install -e '.[baselines]'
mlsurgie bench --output runs/bench-baselines                      # baselines only, 3 seeds
mlsurgie bench --editor random-search --output runs/bench-search  # non-LLM control

pip install -e '.[harness,baselines]'
docker build -f containers/Dockerfile.constructor -t mlsurgie-constructor:local .
export OPENROUTER_API_KEY=...
mlsurgie bench --editor agent --model openrouter/VENDOR/MODEL --output runs/bench-agent
mlsurgie bench --editor agent --model openrouter/VENDOR/MODEL --names anonymized --output runs/bench-agent-anon
```

Each run writes `results.md` (mean ± sd across seeds), `results.json`, and a full protocol run directory per task and seed. Use `--seeds 0` and `--iterations 2` for a quick check.

Live agent runs cost money. Each cell is bounded by its `max_editor_tokens` budget (400k in the shipped tasks), so the worst case is roughly 6 tasks × 3 seeds × 400k tokens. Start with one task and one seed.

## Adding a task

1. Add a CSV and a config under `benchmarks/` with `name`, `dataset`, `goal` (`task`, target or `n_clusters`/`reference_column`, `primary_metric`), `features`, `budget`, and optionally `initial_program` for a repair task. For a numeric model built with historical preprocessing, add `initial_preprocessing`, a JSON path to its saved preprocessing state (matching the feature order).
2. Prefer data with planted, documented structure, or data unlikely to be in pretraining corpora.
3. Add it to `suite.json`, run baselines-only, and check that the task neither saturates (all baselines perfect) nor is pure noise (all baselines close to trivial).
