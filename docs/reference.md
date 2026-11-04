# MLSurgie technical reference

[← Back to the project](../README.md) · [Quickstart](quickstart.md) · [LLM providers](providers.md)

## Inputs and supported scope

- Goal: natural-language description, `task` (`binary`, `multiclass`, `regression`, `clustering`), target (or `n_clusters` plus an optional hidden `reference_column` for clustering), positive label, selection metric, fixed decision threshold, optional target score.
- Dataset: explicitly selected numeric/categorical inputs. Numeric blanks use support medians and a missing flag; values are standardized using support statistics. Categories have explicit missing/unknown bins.
- Architecture: zero, one, or two hidden layers by default (raise `max_hidden_layers`), each Linear followed by `relu`, `tanh`, `sigmoid` or `identity`. The output head depends on the task: sigmoid logit, softmax logits, a raw value, or argmax cluster scores. In JSON-only `build`, the first valid candidate freezes the architecture. The optional [agent harness](agent-harness.md) permits topology changes between candidates.
- Budget: maximum submitted iterations, parameters, widths, output tokens per response, and evaluated rows. Wall time (`max_seconds`) and total editor tokens (`max_editor_tokens`) are optional spend limits, unset (unlimited) by default. When a token limit is set, input-token usage can cause a single call to overshoot it; subsequent work stops and the overshoot is recorded. Final evaluation is reserved separately in the row budget and may add runtime after the editing deadline.

Default split: support/development/test = 60/20/20, stratified by class for classification and random for regression and clustering. Grouped splits are disjoint and approximately follow these proportions by group count. For classification, all three must contain every class. At least five examples per class are needed even to attempt the IID split; that minimum is not enough for reliable generalization estimates.

Preprocessing estimates distribution statistics, not neural weights. Its state is disclosed in the artifact. Numeric validation and split creation use the supplied table; only support rows determine preprocessing. Final-test predictions are computed only after selection.

Protocol components are declared and recorded in the manifest: `FeedbackPolicy` (error rows per evaluation, cumulative cap on revealed development labels), `SelectionRule` (metric, minimum improvement), `Track` (semantic/anonymized, strict/open), and an optional `initial_program` for repair tasks, evaluated as iteration 0. An optional `initial_preprocessing` numeric preprocessing state preserves an imported model’s raw-input behavior when rebasing its first-layer weights to the run’s support encoding; no fitting is performed. Config files specify a path to that state, while `build()` accepts the state directly. Features must match in order; categorical historical encodings are not supported for this conversion. Anonymization is applied by `build()` itself, including for direct Python callers. Baseline comparisons use the same effective selection metric, minimum improvement and input-feature budget. See [Benchmarking](benchmark.md). Custom preservation constraints, user-supplied separate test sets, arbitrary custom layers and additional model families (additive models, scorecards) are future extensions.

## Weight programs

Editors submit JSON, never Python. Each submission specifies its base, architecture (`hidden_sizes`, optional `activations`), rationale, and explicit operations. The output layer's size is fixed by the task: 1 for binary and regression, the number of classes, or `n_clusters`. Unassigned entries of a `zero` construction are zero. Weight matrices use `[out, in]` convention.

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

There is no expression evaluation, code execution, optimizer API, or fitting fallback in the JSON construction language. The optional [agent harness](agent-harness.md) separately executes Python in a construction container to produce these JSON artifacts. This enforces the local operation surface; it does not prove how a remote LLM internally derived a numeric constant. Custom Python editor plugins are trusted extensions and do not constitute an adversarial sandbox.

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

See [CONTRIBUTING.md](../CONTRIBUTING.md), the [framework design](archive/framework-plan.md), and [research dossier](weight-space-research.md). The architecture has separate data, network, proposer, evaluation, and artifact modules. Core tests require no network or model provider.
