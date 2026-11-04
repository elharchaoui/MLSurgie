# MLSurgie technical reference

[← Back to the project](../README.md) · [Quickstart](quickstart.md)

## Inputs and supported scope

- Goal: natural-language description, target, positive label, selection metric, fixed decision threshold, optional target score.
- Dataset: binary labels; explicitly selected numeric/categorical inputs. Numeric blanks use support medians and a missing flag; values are standardized using support statistics. Categories have explicit missing/unknown bins.
- Architecture: zero, one, or two hidden Linear/ReLU layers by default; one output logit interpreted through sigmoid. The first valid candidate freezes the architecture.
- Budget: maximum submitted iterations, parameters, widths, wall time, reported editor tokens, output tokens per response, and evaluated rows. Input-token usage can cause a single call to overshoot its token allowance; subsequent work stops and the overshoot is recorded. Final evaluation is reserved separately in the row budget and may add runtime after the editing deadline.

Default split: support/development/test = 60/20/20, stratified by class. Grouped splits are disjoint and approximately follow these proportions by group count. All three must contain both classes. At least five examples per class are needed even to attempt the IID split; that minimum is not enough for reliable generalization estimates.

Preprocessing estimates distribution statistics, not neural weights. Its state is disclosed in the artifact. Numeric validation and split creation use the supplied table; only support rows determine preprocessing. Final-test predictions are computed only after selection.

The current version supports binary classification only. Regression, multiclass, custom preservation constraints, activations as feedback, user-supplied separate test sets, architecture changes during a run, and other hosted-provider adapters are future extensions. Selection currently uses one primary metric, not an arbitrary natural-language acceptance condition.

## Weight programs

Editors submit JSON, never Python. Each submission specifies its base, architecture, rationale, and explicit operations. Unassigned entries of a `zero` construction are zero. Weight matrices use `[out, in]` convention.

```json
{
  "base": "zero",
  "hidden_sizes": [2],
  "rationale": "Use two ReLU units to retain positive and negative evidence.",
  "operations": [
    {"op": "set", "tensor": "layers.0.weight", "value": [[1, 0], [-1, 0]]},
    {"op": "set", "tensor": "layers.1.weight", "value": [[4, -4]]}
  ]
}
```

This particular example requires two encoded input features. `layers.N.bias` is also editable. Supported operations are `fill`, `set`, `set_entries`, and `add_entries`. An entry has `index` and `value`. A patch can use `base: "best"` with the exact `base_hash` provided in the context. Stale bases, unknown operations/tensors, invalid shapes/indices, architecture changes, and nonfinite or excessive weights are rejected.

There is no expression evaluation, code execution, optimizer API, or fitting fallback in the construction language. This enforces the local operation surface; it does not prove how a remote LLM internally derived a numeric constant. Custom Python editor plugins are trusted extensions and do not constitute an adversarial sandbox.

One submitted candidate counts as one iteration, including invalid proposals. Each valid candidate receives one full development evaluation and a bounded list of largest probability errors. There are no hidden extra probes or retries. Provider errors stop the run and are recorded. The best eligible candidate is frozen before the final test; an unsuccessful target score is reported as such.

## Python API

```python
from mlsurgie import Budget, Feature, Goal, OllamaEditor, Table, build, load

goal = Goal("Predict survival", target="Survived", positive_label="1")
features = [Feature("Age", "numeric"), Feature("Sex", "categorical")]
table = Table.from_csv("data/titanic.csv", features, goal)

result = build(
    table=table,
    editor=OllamaEditor("YOUR_MODEL_NAME"),
    output="runs/experiment-001",
    budget=Budget(iterations=10, max_seconds=1800),
)
print(result.report["final_test"])
if result.model is not None:
    predictor = load(result.path)
    print(predictor.predict_proba([{"Age": 30, "Sex": "female"}]))
```

`predict_proba` returns columns `[negative_label, positive_label]`; inspect `predictor.classes`. `predict` returns original label strings. The native PyTorch module is `predictor.network`. No LLM is needed for prediction.

```bash
mlsurgie predict runs/titanic-001 \
  --dataset data/passengers.csv --output predictions.csv
```

## Artifacts and reproducibility

Each run stores:

```text
manifest.json           goal, feature semantics, support preprocessing, splits,
                        dataset fingerprint, budgets, versions, editor identity
iterations/001/
  context.json          exactly the data/weights/feedback exposed to the editor
  proposal.json         submitted construction/patch
  reply.txt             raw response when supplied by the provider
  result.json           acceptance, diagnostics summary, token usage
  diff.npz              changed indices and old/new values relative to declared base
selection.json          frozen checkpoint identity before final evaluation
model.json              inference contract and preprocessing
weights.npz             non-executable arrays, loaded with allow_pickle=False
checksums.json          artifact integrity checks
report.json             development and final metrics, full attempt history
report.html             local, self-contained readable report
```

`mlsurgie replay` rebuilds every evaluated candidate from its recorded program and checks the final checkpoint identity. It does not rerun the editor or reevaluate the dataset. Dataset contents are not bundled in full; reproducing scores requires the exact original data and split manifest. Context logs intentionally include the examples shown to the editor: treat run folders as dataset artifacts when sharing them.

Reports include log loss, Brier score, ROC-AUC, accuracy, balanced accuracy, precision, recall, F1, and confusion matrix. Only accuracy currently has a Wilson 95% interval, which assumes independent observations. Other metric uncertainty is not yet implemented. Degenerate metrics are explicitly null where appropriate; zero predicted positives gives precision 0.

Checksums detect accidental mismatch, not malicious replacement of an entire run and its checksum file. Provider monetary cost is currently unknown/null; token counts and elapsed time are recorded. The first example's hand-designed success is not evidence of autonomous classifier synthesis.

## Development

See [CONTRIBUTING.md](../CONTRIBUTING.md), the [framework design](framework-plan.md), and [research dossier](weight-space-research.md). The architecture has separate data, network, proposer, evaluation, and artifact modules. Core tests require no network or model provider.
