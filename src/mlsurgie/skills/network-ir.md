# Network IR: what you are writing

A feed-forward network: `x -> [Linear -> activation]* -> Linear -> head`.

```json
{"base": "zero", "hidden_sizes": [16, 8], "activations": ["relu", "relu"],
 "rationale": "unit ledger: ...", "operations": [ ... ]}
```

- `hidden_sizes`: widths; `[]` means the output is a linear function of the inputs.
- `activations`: per hidden layer, one of `relu`, `tanh`, `sigmoid`, `identity` (default relu).
- Tensors: `layers.{i}.weight` with shape `[out, in]`, `layers.{i}.bias` with shape `[out]`.
  Layer 0 reads the encoded inputs (`len(encoded_features)` columns, in that order).
- Every entry starts at zero for `base: "zero"`. Unassigned entries stay zero.
- Operations: `fill` (scalar), `set` (full tensor), `set_entries` / `add_entries` (sparse
  `{"index": [row, col], "value": v}`).
- `base: "best"` + `base_hash` (from state.json `best.hash`) patches the current best; the
  architecture must match exactly.
- Limits in task.json: `max_parameters`, `max_width`, `max_hidden_layers`, `max_abs_weight`.

## Output heads (task.json `task`, `output_size`)

| Task | Outputs | Interpretation |
|---|---|---|
| binary | 1 | logit z; P(positive) = 1/(1+e^-z); class = P >= threshold |
| multiclass | number of classes | softmax over outputs, ordered as task.json `classes` |
| regression | 1 | the prediction in the target's original units |
| clustering | n_clusters | one score per cluster; assignment = argmax |

## Encoding (exact)

- numeric `x`: `x:standardized = (value_or_median - mean) / scale`; `x:missing` is 1 when blank.
- categorical `c`: one-hot `c='v'` per support category, plus `c:missing`, `c:unknown`.
- A raw threshold t on `x` is the standardized threshold `(t - mean) / scale`.
- A raw-unit slope a on `x` is weight `a * scale` on `x:standardized`.

Your NumPy forward pass must reproduce this exactly; float32 rounding is the only difference.
