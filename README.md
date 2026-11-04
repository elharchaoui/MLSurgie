<div align="center">

# MLSurgie

### Weight-space programming for neural networks

**Describe a prediction goal. Let an LLM write the weights. Export a tested network.**

[![Tests](https://github.com/elharchaoui/MLSurgie/actions/workflows/tests.yml/badge.svg)](https://github.com/elharchaoui/MLSurgie/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/built_with-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status: research prototype](https://img.shields.io/badge/status-research_prototype-0F766E)](#current-scope)

[Quickstart](docs/quickstart.md) · [LLM providers](docs/providers.md) · [Technical reference](docs/reference.md) · [Titanic experiment](experiments/titanic/RESULTS.md) · [Contribute](CONTRIBUTING.md)

</div>

![MLSurgie workflow: goal and support data, LLM weight programs, validation and development feedback, then a frozen network with final test metrics.](docs/assets/weight-space-loop.svg)

MLSurgie is a Python framework for **LLM-driven weight-space programming**: constructing small neural networks through explicit assignments to their weights and biases. Supply a goal, labeled tabular data, and an iteration budget. An editor proposes weight programs; MLSurgie checks them, measures development performance, and returns bounded feedback. The best network is frozen before final-test evaluation.

The construction loop uses direct tensor operations without backpropagation, gradient descent, or conventional estimator fitting. Evaluation-guided revisions are still adaptive optimization. Once exported, the network predicts without an LLM.

## Start here

| Your goal | Start with |
|---|---|
| Try the pipeline without a model or credentials | [Offline demo below](#quickstart) |
| Choose a local or hosted LLM | [Provider setup](docs/providers.md) |
| Build a classifier using a local LLM | [Live-editor quickstart](docs/quickstart.md#build-with-a-local-llm) |
| Understand the weight language and artifacts | [Technical reference](docs/reference.md) |
| Inspect an actual LLM construction experiment | [Titanic protocol and results](experiments/titanic/RESULTS.md) |
| Add an editor or contribute a benchmark | [Contributing guide](CONTRIBUTING.md) |

## Quickstart

Requires **Python 3.11+** and a compatible PyTorch wheel; Python 3.12 is used in CI. Install from source:

```bash
git clone https://github.com/elharchaoui/MLSurgie.git
cd MLSurgie
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

mlsurgie demo --output runs/first-demo
mlsurgie replay runs/first-demo
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. Use a new output directory for every run.

The demo is a **scripted known-rule fixture** for `x > 0`. It checks the complete proposal → selection → export → replay path without downloading an LLM. The CLI prints the selected iteration and test metrics. Open `runs/first-demo/report.html` for the full report; replay prints the verified checkpoint hash.

## Choose your LLM provider

There is **no default LLM or model**. Select one provider and an explicit model for each live build:

| Path | LLM used | Configuration |
|---|---|---|
| Offline `mlsurgie demo` | None; scripted known-rule proposals | No provider or key |
| Ollama | Your installed local model | `--ollama-model MODEL_ID`, optional `--ollama-url` |
| OpenRouter | Your chosen hosted model | `--openrouter-model VENDOR/MODEL`, `OPENROUTER_API_KEY` environment variable |

OpenRouter is built in; no extra SDK is required. After downloading the data in the [quickstart](docs/quickstart.md):

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
mlsurgie build \
  --config examples/titanic-seaborn.json \
  --dataset data/titanic-seaborn.csv \
  --openrouter-model YOUR_VENDOR/YOUR_MODEL \
  --output runs/openrouter-classifier
```

Replace the placeholders with your key and an exact [OpenRouter model ID](https://openrouter.ai/models) supporting JSON output. No specific hosted or Ollama model has yet been benchmarked through these packaged adapters. The reported Titanic experiment used a separate interactive assistant bridge. LiteLLM is not a built-in adapter.

Follow the [copyable Titanic walkthrough](docs/quickstart.md) for both provider paths, or read the [provider configuration reference](docs/providers.md).

## How weight-space programming works

1. **Specify behavior.** Declare the goal, target, input semantics, selection metric, and budget. MLSurgie creates support, development, and test partitions.
2. **Construct weights.** The editor sees support examples, preprocessing, constraints, and development feedback. It returns JSON operations on neural weights and biases.
3. **Measure and revise.** MLSurgie rejects invalid programs, scores valid candidates, and retains the best eligible network. The first valid architecture stays fixed throughout the run.
4. **Freeze and deliver.** Final-test evaluation follows selection. Export includes weights, preprocessing, proposal history, checksums, and metrics.

```json
{
  "base": "zero",
  "hidden_sizes": [2],
  "rationale": "Represent positive and negative evidence with two ReLU units.",
  "operations": [
    {"op": "set", "tensor": "layers.0.weight", "value": [[1, 0], [-1, 0]]},
    {"op": "set", "tensor": "layers.1.weight", "value": [[4, -4]]}
  ]
}
```

This example expects two **encoded** inputs; unassigned parameters stay zero. Editors can use `fill`, `set`, `set_entries`, or `add_entries`, and patch the current best checkpoint with its exact hash. See [weight-program semantics](docs/reference.md#weight-programs).

## Evidence: Titanic classifier

In an interactive Codex experiment, the assistant designed a 769-parameter neural network through eight proposals. Development log loss selected candidate 7 before opening the 179-passenger test set.

| Predictor | Held-out accuracy | ROC-AUC | Log loss ↓ |
|---|---:|---:|---:|
| Constant support prevalence | 61.45% | 0.5000 | 0.6667 |
| Fixed sex-based rule | 76.54% | 0.7389 | 0.5440 |
| **LLM-constructed neural network** | **80.45%** | **0.8398** | **0.4455** |

This is **one public dataset and one split**. Titanic may be present in the editor's pretraining; relatives can cross the IID split. Accuracy's Wilson 95% interval is 74.02–85.59%. The result demonstrates execution of the protocol, not general superiority. The interactive editor differs from the packaged Ollama adapter; local-model results will vary. [Read the full methods, revisions, and limitations →](experiments/titanic/RESULTS.md)

## Current scope

| Supported in v0.1.0 | Details |
|---|---|
| Binary tabular classification | Numeric and categorical features; explicit feature allowlist |
| Small feed-forward networks | Linear/ReLU hidden layers and one sigmoid output; zero to two hidden layers by default |
| Editors | Ollama local JSON chat, OpenRouter hosted JSON chat, scripted fixtures, trusted Python editor extensions |
| Evaluation | Log loss, Brier, ROC-AUC, accuracy, balanced accuracy, precision, recall, F1, confusion matrix |
| Reproducibility | Saved programs, tensor diffs, dataset fingerprints, split indices, checksum-verified loading and replay |
| Inference | Python API and CSV CLI; no editor required |

Regression, multiclass classification, changing architecture during a run, and separate user-supplied test sets are future extensions. Run folders contain exposed data examples; review them before sharing. [Full constraints and trust model →](docs/reference.md#inputs-and-supported-scope)

## Use the exported network

```python
from mlsurgie import load

predictor = load("runs/first-demo")
print(predictor.classes)  # probability column order
print(predictor.predict_proba([{"x": 0.7}, {"x": -0.7}]))
print(predictor.predict([{"x": 0.7}]))
```

For a classifier built on your own CSV:

```bash
mlsurgie predict runs/my-classifier --dataset data/new-rows.csv --output predictions.csv
```

[Build from Python](docs/reference.md#python-api) · [Artifact layout](docs/reference.md#artifacts-and-reproducibility)

## Questions

**What inputs do I need?** A natural-language goal, a labeled binary dataset, an explicit list of prediction-time features, and evaluation limits. The goal and examples guide the editor; the selected numeric metric determines the winner.

**Does an LLM generate every weight?** The editor supplies explicit parameter operations starting from zero or the best checkpoint. A program may leave parameters at zero. MLSurgie validates and applies the program; it has no weight-fitting fallback.

**Is this training-free?** It uses no gradients or conventional target-model fitting. It does use labeled support data, support-only preprocessing statistics, and repeated development feedback. Those revisions are adaptive optimization.

**Can I reproduce the reported Titanic result with any local model?** No. Saved programs reproduce a network; rerunning an editor is a new experiment. Model capability, randomness, budget, and dataset split affect results.

**Does the demo demonstrate LLM reasoning?** The offline demo verifies the framework using a known scripted rule. The linked Titanic report documents the separate interactive LLM experiment.

## Research and community

Inspired by [Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). MLSurgie is an independent implementation exploring direct weight construction for small tabular neural networks. Read the [research dossier](docs/weight-space-research.md) and [framework design](docs/framework-plan.md).

Contributions are welcome: editor adapters, unfamiliar benchmark tasks, reproducible failures, and numerical validation. Follow [CONTRIBUTING.md](CONTRIBUTING.md). For help, [open an issue](https://github.com/elharchaoui/MLSurgie/issues/new/choose) with your configuration, versions, and a minimal example. Omit private data and credentials.

Maintained by [@elharchaoui](https://github.com/elharchaoui). Licensing terms have not yet been specified; a license decision is needed before treating this as permissively licensed software.
