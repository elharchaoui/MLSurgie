# Inputs (read-only)

- `task.json`: goal, task type (regression), classes, output size, feature declarations,
  preprocessing state, encoded feature names, budgets, track, and the weight-program format.
- `state.json`: refreshed every turn: iteration, remaining budgets, best network (weights and hash),
  best development metrics, and last development feedback.
- `data/support.csv`: 265 labeled support rows with raw feature values. `target` is the numeric target in original units.
- `data/support_encoded.csv`: the same rows as the network actually sees them
  (columns in `encoded_features` order), plus the same label column.

The network input is exactly the encoded vector. Encoding (from support rows only):
numeric `x` -> `x:standardized` = (x_or_median - mean) / scale, and `x:missing` in {0,1};
categorical `c` -> one-hot `c='value'` columns, plus `c:missing` and `c:unknown`.
Preprocessing statistics are in task.json `preprocessing` (mean, scale, median, categories).
To express a raw threshold t on numeric x, use (t - mean) / scale on `x:standardized`.

Development and final-test rows are not available here. Write `/workspace/candidate.json`.
