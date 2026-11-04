# Titanic: interactive LLM weight construction

On 2026-11-04, the assistant in this Codex conversation designed and revised a neural network's weights directly. After eight development evaluations, the frozen network classified **144/179 held-out passengers correctly (80.45%)**, with **0.8398 ROC-AUC** and **0.4455 log loss**.

This is an actual execution, not the scripted demo. No optimizer, backpropagation, regression fitting, or numerical parameter search was used. The assistant chose circuit constants from support statistics and development feedback; a deterministic materializer translated those constants into tensors. This is supervised, data-informed model construction, not data-free learning. The run's model identifier is a session label, not independently verified provider metadata; token usage and monetary cost are unavailable.

## Data and protocol

- Source: [Seaborn Titanic CSV](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv), downloaded from its raw GitHub URL; 891 labeled rows.
- Download SHA-256: `81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2`.
- Seven inputs: passenger class, sex, age, siblings/spouses, parents/children, fare, embarkation port. Excluded `alive` (direct target leakage) and all other columns.
- Stratified split, seed 42: 534 support, 178 development, 179 final test.
- Support-only preprocessing; 22 encoded inputs. Editor context offered 32 support examples plus grouped support counts. The assistant inspected support examples and aggregate counts, followed by development metrics and largest probability errors.
- Eight proposals fixed in advance; minimize development log loss; classification threshold fixed at 0.5. Final test opened only after candidate selection. No proposals changed after test results.
- Architecture: 22 inputs → 32 ReLU units → one sigmoid output; 769 stored parameters, including zero padding. Circuits express sex/class combinations, bounded child-age effects, family size, missing age, and embarkation effects.
- Contexts, rationales, complete weight programs, tensor differences, and candidate metrics are retained under `experiments/titanic/codex/iterations/`.

## Development trajectory

| Candidate | Change | Log loss | Accuracy | Selected when evaluated |
|---|---|---:|---:|---|
| 1 | Initial group and demographic circuits | 0.45195 | 79.78% | Yes |
| 2 | Soften group confidence and older-age penalty | 0.44366 | 80.34% | Yes |
| 3 | Fare, family, and child refinements | 0.44264 | 80.90% | Yes |
| 4 | Remove low-fare penalty | 0.43797 | 80.90% | Yes |
| 5 | Further soften two group logits | 0.43943 | 80.90% | No |
| 6 | Increase small-family bonus | 0.44633 | 80.90% | No |
| **7** | **Remove small-family bonus** | **0.43661** | **80.90%** | **Yes** |
| 8 | Increase male-child bonus | 0.43871 | 80.34% | No |

Rejected candidates did not replace the best model. This demonstrates useful feedback and rejection, although improvements are modest on a small development set.

## Final test, opened after selection

The two baseline definitions were recorded in `baseline-plan.json` before opening the final test. Neither baseline uses conventional model training.

| Frozen predictor | Accuracy | ROC-AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| Constant support prevalence | 61.45% | 0.5000 | 0.6667 | 0.2369 |
| Sex rule: female 0.75, otherwise 0.20 | 76.54% | 0.7389 | 0.5440 | 0.1793 |
| LLM-constructed NN, candidate 7 | **80.45%** | **0.8398** | **0.4455** | **0.1414** |

### Trained baselines on the same split (added later)

These were computed afterwards with `mlsurgie.baselines`, using the same 534 support rows, the same 22 encoded inputs, hyperparameters chosen on the 178 development rows, and one test evaluation. They were not part of the predeclared plan; they answer the obvious question of how the network compares with conventionally fitted models. Values are in [`trained-baselines.json`](trained-baselines.json).

| Predictor (seed-42 split) | Accuracy | ROC-AUC | Log loss | Stored numbers |
|---|---:|---:|---:|---:|
| Logistic regression (C=0.1) | 75.98% | 0.8437 | 0.4720 | 23 |
| Random forest (300 trees, min leaf 5) | 78.21% | 0.8410 | 0.4753 | 52,764 |
| RBF SVM (C=3) | **81.01%** | 0.8115 | 0.4828 | 5,406 |
| LLM-constructed NN, candidate 7 (this experiment) | 80.45% | 0.8398 | **0.4455** | 769 |
| Opus 5.5 agent NN ([separate run](opus55/README.md)) | 79.89% | **0.8439** | 0.4551 | 337 |

Both hand-written networks have better log loss than every trained baseline and ROC-AUC on par with logistic regression; the SVM has the best accuracy. With 179 test rows these differences are within sampling noise. The fair reading is that the networks are **competitive with standard trained models on this split**, not better than them.

NN precision: 78.33%; recall: 68.12%; F1: 0.7287; balanced accuracy: 78.15%. Confusion matrix, rows actual and columns predicted, non-survival first: `[[97, 13], [22, 47]]`. Accuracy Wilson 95% interval: **74.02–85.59%**, assuming independent observations. Seven more passengers were classified correctly than by the sex rule; this single run does not establish a statistically reliable advantage.

## Interpretation and limitations

The experiment demonstrates that an LLM can construct a runnable tabular classifier by writing neural weights, improve it using bounded development feedback, and export a replayable model. The sex/class and family structure was designed explicitly: this is not evidence that arbitrary architectures or arbitrary tasks can be solved by direct weight generation.

Titanic is a familiar public benchmark and may have appeared in the assistant's pretraining. The split is IID, so relatives and duplicate feature patterns may cross partitions. Test separation was procedural within a trusted local agent session, not an OS security boundary. Repeated development feedback can overfit. This is one dataset and seed, with no distribution-shift evaluation and no calibrated uncertainty guarantee. Trained baselines were added after the fact (above). Development errors are residual errors, not evidence of temporal data drift.

The next credible evaluation should freeze this protocol on unfamiliar datasets, use multiple seeds and group-aware splits where appropriate, and compare direct-weight construction against fixed heuristics and separately reported trained baselines.

## Artifacts and verification

- Human-readable run: `experiments/titanic/codex/report.html`.
- Machine-readable metrics: `report.json`; baseline metrics: `baselines.json`.
- Deployable network: `weights.npz`, `model.json`, `checksums.json` in the run directory.
- Construction: `submit.py` and `design-01.json` through `design-08.json` in this directory.
- Replay verified the final state hash `029d43a0ba41a4581ca3e405619cb0b2ba9dd60f369acccc8c0331421f0325ce`.
- Saved-model loading and prediction succeeded for all 891 source rows. These predictions include support and development rows; only the 179 held-out rows contribute to the final test metrics above.

```sh
.venv/bin/mlsurgie replay experiments/titanic/codex
.venv/bin/python experiments/titanic/evaluate_baselines.py
.venv/bin/mlsurgie predict experiments/titanic/codex --dataset data/titanic-seaborn.csv --output /tmp/titanic-predictions.csv
```

Replay deterministically reconstructs the saved model without an LLM. `run_interactive.py` needs a fresh output directory and an active editor supplying eight responses; it does not autonomously reproduce the assistant's reasoning. The complete run record is tracked in `experiments/titanic/codex/`, so replay works from a fresh clone. The dataset itself must be downloaded (see the [quickstart](../../docs/quickstart.md)) before re-running baselines or predictions.
