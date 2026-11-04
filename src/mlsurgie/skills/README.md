# Weight-space programming playbook

You are building a neural network the way an engineer builds a circuit: by understanding the
data and placing every weight on purpose. No training loop will run. These skills are knowledge,
not code — write your own tools in `/workspace` (they persist across turns).

## Workflow of a weight-space engineer

1. **Read the brief.** `/inputs/README.md`, `/inputs/task.json` (task type, output size, encoding,
   budgets), `/inputs/state.json` (iteration, best network, feedback).
2. **Build your bench.** Write a small module, e.g. `/workspace/bench.py`, with: data loading, a
   NumPy forward pass that mirrors the network exactly (see `network-ir.md`), the task's metric,
   and a writer for `candidate.json`. Verify your forward pass on a trivial network first.
3. **Profile the data.** Base rates, per-category rates, per-feature quantiles, conditional rates in
   bins, class-conditional means, correlations, missingness patterns, outliers.
4. **Hypothesize mechanisms.** Which features carry signal, in what shape (monotone, threshold,
   bump, interaction)? Each mechanism should map to a circuit (`circuits.md`).
5. **Compose weights.** Translate each mechanism into units and output weights. Keep a ledger of
   which unit implements what; it becomes your rationale.
6. **Check locally.** Score your candidate on support rows. Hold out a slice of support for an
   honest local estimate — development feedback is limited and revealing.
7. **Submit and read feedback.** One candidate per turn. Development metrics tell you whether a
   mechanism helped. Change one idea at a time so feedback is attributable.
8. **Stop when it no longer pays.** Every submission costs budget, and repeated tuning on the
   development split overfits it. Call finish_run when further changes are not justified.

## Skill index

| Skill | Use it when |
|---|---|
| `network-ir.md` | Always: weight-program format, tensor shapes, heads, encoding math |
| `policy.md` | Always: what counts as reasoning vs. fitting |
| `circuits.md` | Thresholds, intervals, AND/OR, absolute value, piecewise-linear shapes |
| `binary-evidence.md` | Binary classification: log-odds evidence, calibration, interactions |
| `rare-events.md` | Imbalanced problems such as fraud or anomaly detection |
| `multiclass.md` | Several classes: per-class logits, prototypes |
| `regression.md` | Numeric targets: baselines, segment means, piecewise-linear response |
| `clustering.md` | No labels: centroid layers, density reasoning, silhouette |
| `repair.md` | A starting network is provided: diagnose and patch it |
