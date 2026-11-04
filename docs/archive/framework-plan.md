# MLSurgie: weight-space programming only

Version 4 — construction-first extension of version 3; supersedes [version 2](framework-plan-v2.md). Proposed design, not implemented.

## Scope

The primary workflow constructs a neural network from scratch for a supplied goal and dataset: an LLM proposes an architecture within declared bounds, writes its weights, probes its behavior, and revises the weights within an iteration budget. The framework also enables inspection, explanation, and direct editing of existing model weights. It does not orchestrate conventional training, fit classical estimators, tune hyperparameters, or calibrate models. Historical training data supplies context and diagnostic evidence, never an implicit training job. Classical ML refers to prediction tasks such as tabular classification and regression; the initial target representation is a small explicit neural network.

Existing training methods can be external experimental baselines. They are not framework operations. Direct weight edits selected using observed performance are adaptive optimization even without gradients; describe them accurately.

## Three input contracts

### ModelSpec: what the parameters mean

Require the executable architecture, exact checkpoint, parameter names/shapes/dtypes, forward computation, activation functions, buffers, preprocessing state, feature order, units, missing/category handling, target semantics, and output decoding/class order. Validate all inputs against the executable model. A checkpoint alone does not identify a model's computation or its feature semantics.

Separate immutable architecture and preprocessing from mutable weight regions. Initial support is fixed-architecture Linear/ReLU MLPs with identity, sigmoid, or softmax output semantics. More architectures require explicit adapters. Do not flatten arbitrary estimator state into a tensor and assume that editing it has neural-weight semantics.

For construction, accept an architecture and initialization policy instead of a checkpoint. The agent writes the parameters directly. Treat architecture search as a later explicitly distinct capability.

### GoalSpec: what behavior should change

Record the natural-language request plus a machine-readable objective, target slices, preservation checks, permitted tensors, edit-size constraints, numerical validity requirements, and budgets. Metrics and acceptance thresholds are specified before editing. Translate ambiguity into an explicit unresolved field; never silently invent a domain requirement or utility tradeoff.

Goals may target prediction quality, a systematic error, monotonic behavior on a defined domain, or a known algorithm to construct. Explanations and correctness claims require probes or supported proofs; comments about weights alone are hypotheses.

### DataContext: what evidence is available

Keep separate roles for historical training data, editor-visible diagnostic data, candidate-selection validation, and final test data. Attach dataset versions/fingerprints, provenance, label definitions, original preprocessing, time/group identities, and the original training split when known.

Historical data explains distributions and failure cases but is not independent evidence of generalization. Subdividing the original training data after training does not make it unseen by the base model. If new held-out data is unavailable, report in-sample/diagnostic findings and explicitly leave generalization unestablished.

Data may remain local. Expose summaries, chosen examples, predictions, residuals, intermediate activations, and counterfactual probes through a metered service. Record exactly which information reached the editor. Feature semantics, labels, and access to data are distinct permissions. Do not send whole datasets to a hosted LLM by default.

Missing training data is allowed: use model metadata, synthetic domain probes, and user examples. A zero-probe run can return a constructed candidate but cannot establish empirical accuracy. Synthetic labels derived from an assumed rule test that rule, not real-world predictive validity.

## Execution

1. Validate and fingerprint ModelSpec and checkpoint.
2. Resolve GoalSpec into checks and explicit acceptance criteria.
3. Register data roles and isolate the final evaluator.
4. Compute a baseline and expose only permitted diagnostics.
5. The LLM inspects parameter slices and approved weight algebra; proposes a concrete weight patch with a mechanism hypothesis.
6. Validate the patch, apply it to a checkpoint copy, and run budgeted diagnostic probes.
7. Repeat within budget; retain all attempted candidates and observations.
8. Select using allowed validation, freeze the candidate, and evaluate once on the final test.
9. Return the checkpoint, parameter diff, input contracts, probe history, and evidence report. Return no accepted edit when criteria fail.

The editor cannot call fit, backward, an optimizer, regression solvers, or distillation to derive weights from targets. Strict mode uses allowlisted tensor operations and structured patches; arbitrary Python is a separately labeled research mode requiring isolated execution and transcript audit. Merely hiding a model object does not prevent manually implementing inference or fitting.

Evaluation feedback is metered by calls, rows, model executions, and time. Selection data repeatedly exposed to the agent becomes development data. Final-test feedback cannot be used to revise a candidate while retaining a claim of untouched testing.

## Construction and editing

Edit: existing architecture and weights plus goal and optional historical context.

Construct: specified architecture plus desired function and input/output semantics; the LLM writes weights from initialization. Data can validate proposals or provide limited diagnostic examples under the declared protocol, without fitting routines. Unknown statistical relationships cannot be guaranteed from feature descriptions alone.

Both paths use the same tensor patch mechanism, contracts, and evaluator. No conventional training fallback occurs inside the framework.

## Initial implementation and community interface

Build ModelSpec validation, checkpoint snapshots, weight inspection, structured patches, and a separate evaluator for one small MLP family. Add goal and dataset manifests, replay, and an optional editor provider interface. Ship examples for constructing a specified function and repairing a naturally occurring error with separate historical and unseen data.

Contributors add architectures, parameter-inspection tools, benchmark cases, or editor strategies through explicit contracts. Every benchmark records the editor's information, tools, budgets, and all failed attempts. Provide offline replay fixtures clearly identified as fixtures, alongside live WSP experiments. Native prediction from the resulting checkpoint requires no editor LLM.

Adoption depends on reproducible direct weight-programming results. Generic estimator fitting and EBM editing workflows from previous versions are removed from scope.


## Construction-first workflow: Titanic example

### Minimum user inputs

- Goal: predict the target, define the positive class and intended prediction-time information.
- Labeled table: a path or local data object and target column; unlabeled rows alone cannot score predictive correctness.
- Budget: maximum proposal/evaluation iterations N, plus runtime, editor-token/cost, and evaluated-row limits.

Optional inputs are feature semantics, grouping/time identifiers, desired metric, acceptance thresholds, architecture/resource limits, and user-supplied untouched test data. Defaults are resolved visibly into a run manifest before iteration. The user need not provide weights or a complete architecture.

For a Titanic example, the target could be Survived with binary classification and probability output. The framework should validate the supplied file rather than assume a particular edition of the dataset. Propose log loss as the primary probabilistic objective, with ROC-AUC, accuracy, balanced accuracy, precision, recall, F1, Brier score, and confusion matrix as secondary reports where defined. Define the decision threshold before final testing; any threshold selection uses development data and is recorded.

### Dataset contract for a new network

Accept a full labeled table where available and split before editor access. The editor receives a support portion containing labels and schema. Development probes return metrics and bounded diagnostic examples. A final test split remains outside editor access. All data repeatedly used for feedback is development data regardless of its filename.

A small support sample is enough to attempt synthesis, not enough to guarantee reliable generalization. When sample size permits, a visible default might be stratified 60/20/20 support/development/test, overridden for grouping or time dependencies. Record split seeds, row identities, missingness, class counts, and provenance. If the data is too small for useful split estimates, request additional labels or report exploratory results rather than manufacture certainty.

Preprocessing is explicit and fixed before weight iteration. Missing-value rules, categorical vocabulary, numeric transformations, and any data-derived preprocessing statistics are constructed only from support data. Such statistics are disclosed data preparation, not neural-weight fitting. A strict all-programmed variant can use user-specified constants instead. Unknown categories and missing inputs must have defined behavior.

For Titanic, exclude outcome-derived or post-outcome inputs; group related passengers when appropriate to the intended evaluation. A public benchmark may have been encountered by the editor during pretraining. Do not claim uncontaminated discovery from Titanic alone; include fresh/private tasks in research evaluation. Predictive feature associations are not causal explanations.

### Architecture and weight construction

Allow the LLM to select an architecture from a small supported family, for example one or two hidden Linear/ReLU layers with sigmoid/logit binary output, under parameter and latency limits. Show the proposed architecture in the run manifest. Freeze it for the initial experiment. Architecture revisions can later be enabled as separately counted proposals with complete cost/provenance.

The LLM produces a structured construction program describing parameter assignments and compositions of approved tensor operations. The framework materializes the complete initial checkpoint, including explicit zero values for unspecified entries where allowed. Every parameter must have defined provenance. No pretrained target weights, optimizer, gradient, least-squares fit, or distillation is invoked.

Programmatic assignments avoid writing thousands of raw numbers in a prompt. They are still direct construction when constants and expressions are designed by the LLM. Unrestricted code can conceal fitting, so the default runtime uses an allowlisted construction language. Experiments admitting arbitrary Python must be labeled and audited separately.

### Iteration semantics

One iteration consists of one candidate submission, validity checks, and a fixed bounded development evaluation. N caps submissions, including invalid submissions; retries and infrastructure failures are recorded separately. Multiple diagnostic tool calls within an iteration have their own shared row/call budget. The initial candidate counts as iteration 1.

At each iteration the editor receives primary/secondary scores, class- and slice-level errors where sample counts support them, and optional requested activations. It proposes a specific mechanism and direct parameter changes. Diagnose residual error, overconfidence, saturation, or subgroup regression; use “distribution drift” only for a measured distribution change.

Keep the best eligible development candidate, not necessarily the last candidate. Stop at N, budget exhaustion, a predeclared success condition, or a predeclared lack-of-progress rule. Repeated development evaluation can overfit even without gradients; report the number of candidates and retain a truly untouched test.

### Deliverable

Freeze the selected candidate and evaluate it once on the final test. Return architecture/configuration, weights, preprocessing, a local predict/predict_proba interface, the goal and data manifests, proposal history, scores with sample counts and suitable uncertainty, resource use, and known limitations. Distinguish final-test from development scores. If no candidate meets the target, return the best valid network labeled as failing the target, or no model if all candidates are invalid.

There is no training fallback. Successful improvement is an empirical question. The complete output should make both a useful network and an unsuccessful synthesis attempt reproducible.
