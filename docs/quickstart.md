# MLSurgie quickstart

[← Project overview](../README.md) · [Technical reference](reference.md)

This guide takes you from source installation to an exported neural network. The offline path checks the framework; the live path asks a local LLM to write weights.

## Install

Use Python 3.11+ with a compatible PyTorch wheel. These commands use Python 3.12:

```bash
git clone https://github.com/elharchaoui/MLSurgie.git
cd MLSurgie
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. If PyTorch installation fails, use the [official installation selector](https://pytorch.org/get-started/locally/) for your platform, then retry the editable install. No published PyPI package is assumed by this guide.

## Check the offline pipeline

```bash
mlsurgie demo --output runs/first-demo
mlsurgie replay runs/first-demo
```

Expected behavior: candidate 2 is selected, a final-test report is printed, and replay returns `verified_state_hash`. Open `runs/first-demo/report.html` in your browser. This is a deterministic scripted fixture for a known threshold function, not an LLM performance test.

## Build with a local LLM

### 1. Prepare the editor

Install [Ollama](https://ollama.com/), start its local server, and download a model capable of following structured JSON instructions. In a terminal, replace the placeholder with your installed model's exact name:

```bash
ollama pull YOUR_MODEL_NAME
ollama list
```

If the service is not already running, run `ollama serve` in another terminal. MLSurgie defaults to `http://localhost:11434`. JSON support alone does not guarantee that a model can construct useful weights.

### 2. Download labeled data

From the repository root, using macOS/Linux or another shell with `curl`:

```bash
mkdir -p data
curl --fail --location \
  https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv \
  --output data/titanic-seaborn.csv
```

Source: [Seaborn's example datasets](https://github.com/mwaskom/seaborn-data). The matching [configuration](../examples/titanic-seaborn.json) uses lowercase column names and seven explicit inputs. It excludes `alive`, which reveals the target. The separate `examples/titanic.json` uses Kaggle-style capitalized names; do not mix these configurations.

### 3. Construct the network

```bash
mlsurgie build \
  --config examples/titanic-seaborn.json \
  --dataset data/titanic-seaborn.csv \
  --ollama-model YOUR_MODEL_NAME \
  --output runs/my-classifier
```

This configuration allows 10 submissions, at most 4,096 parameters, and 1,800 seconds for the construction loop. Invalid programs consume an iteration. A slow model may time out; a model can also fail to produce any valid candidate. Both are reported rather than silently replaced with a fitted model.

Only support examples, support preprocessing, current weights, and development feedback enter editor requests. A remote `--ollama-url` sends that context to the chosen server. Final-test evaluation occurs after model selection. The IID split is suitable for exploring the protocol; consider meaningful group identifiers for a stronger benchmark.

### 4. Inspect and reuse

Open `runs/my-classifier/report.html`. Check status, failed attempts, the selected iteration, and `final_test`. A high development score alone does not establish generalization.

```bash
mlsurgie replay runs/my-classifier
mlsurgie predict runs/my-classifier \
  --dataset data/titanic-seaborn.csv \
  --output predictions.csv
```

The prediction command above exercises inference on all source rows, including support and development rows. Use the report's held-out test metrics to assess performance. For deployment, replace the source CSV with new rows containing the declared input columns; labels are unnecessary for prediction.

```python
from mlsurgie import load

predictor = load("runs/my-classifier")
rows = [{"pclass": "3", "sex": "female", "age": "28",
         "sibsp": "0", "parch": "0", "fare": "8.05", "embarked": "S"}]
print(predictor.classes)
print(predictor.predict_proba(rows))
print(predictor.predict(rows))
```

## Bring your own dataset

Copy the example configuration and change `goal.target`, `positive_label`, the description, and `features`. Select only features available at prediction time. The current task must have exactly two nonmissing target labels. Feature kinds are `numeric` or `categorical`; target and optional group identifiers must be excluded from the feature list. Add `group_column` when related records should remain together.

Choose a primary metric (`log_loss`, `brier`, `roc_auc`, `accuracy`, or `balanced_accuracy`) before running, and keep final-test results out of subsequent editor feedback. [See the complete contract](reference.md#inputs-and-supported-scope).

## Troubleshooting

| Symptom | Action |
|---|---|
| Output directory already exists | Choose a fresh run path; existing artifacts are preserved. |
| Missing CSV column or incorrect target | Match the configuration to the exact CSV headers and positive-label string. |
| Ollama connection error | Check the server, installed model name, and `--ollama-url`. |
| No model exported | Read `report.json` and iteration results for invalid proposals, timeouts, or exhausted budget. |
| Final metrics are weak | Inspect development errors and support coverage; compare against a simple baseline and report failures. |
| Need help | [Open an issue](https://github.com/elharchaoui/MLSurgie/issues/new/choose) with versions, a minimal config, and redacted errors. |

## Develop locally

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Core tests need no provider or network. See [contribution guidelines](../CONTRIBUTING.md) for editor and benchmark requirements.
