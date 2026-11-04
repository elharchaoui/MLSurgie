"""Feed-forward networks constructed entirely by checked JSON weight programs.

The weight program is MLSurgie's intermediate representation (IR). Editors may author
it however their track allows (JSON directly, or code in a sandbox), but only the IR
is validated, hashed, replayed, and exported.
"""
import hashlib
import json
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .contracts import Budget

ACTIVATIONS = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": lambda x: x}


def sigmoid(logits):
    # float32 torch sigmoid keeps format-1 artifact metrics bit-identical and overflow-safe.
    return torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).numpy().astype(float)


def integer(value, label):
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def parameter_count(input_size, hidden_sizes, output_size=1):
    sizes = [input_size, *hidden_sizes, output_size]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, budget=None, output_size=1, activations=None):
        super().__init__()
        budget = budget or Budget()
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) > budget.max_hidden_layers:
            raise ValueError("Too many hidden layers or invalid hidden_sizes")
        for width in hidden_sizes:
            if not 1 <= integer(width, "width") <= budget.max_width:
                raise ValueError("Hidden width exceeds budget")
        if not 1 <= integer(output_size, "output_size") <= 1024:
            raise ValueError("Invalid output size")
        activations = ["relu"] * len(hidden_sizes) if activations is None else activations
        if (not isinstance(activations, list) or len(activations) != len(hidden_sizes)
                or any(a not in ACTIVATIONS for a in activations)):
            raise ValueError(f"activations must list one of {sorted(ACTIVATIONS)} per hidden layer")
        if parameter_count(input_size, hidden_sizes, output_size) > budget.max_parameters:
            raise ValueError("Parameter count exceeds budget")
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.output_size = output_size
        self.activations = list(activations)
        sizes = [input_size, *hidden_sizes, output_size]
        # Meta construction avoids even temporary random weight initialization.
        self.layers = nn.ModuleList([nn.Linear(a, b, device="meta") for a, b in zip(sizes, sizes[1:])])
        self.to_empty(device="cpu")
        with torch.no_grad():
            for param in self.parameters():
                param.zero_()
                param.requires_grad_(False)
        self.eval()

    @property
    def default_architecture(self):
        return self.output_size == 1 and all(a == "relu" for a in self.activations)

    def forward(self, x):
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = ACTIVATIONS[activation](layer(x))
        return self.layers[-1](x)

    def outputs(self, x):
        """Raw output matrix [rows, output_size]; task heads interpret it."""
        with torch.inference_mode():
            raw = self(torch.as_tensor(np.asarray(x), dtype=torch.float32))
            if not torch.isfinite(raw).all():
                raise ValueError("Network produced nonfinite outputs")
            return raw.numpy().astype(float)

    def probabilities(self, x):
        """Binary convenience: sigmoid of the single output logit."""
        if self.output_size != 1:
            raise ValueError("probabilities() applies to single-output networks")
        return sigmoid(self.outputs(x)[:, 0])

    def arrays(self):
        return {name: p.detach().cpu().numpy().copy() for name, p in self.named_parameters()}

    def architecture(self):
        return {"hidden_sizes": self.hidden_sizes, "activations": self.activations, "output_size": self.output_size}

    def fingerprint(self):
        # Default (format 1) architectures keep their original hash for artifact compatibility.
        header = ([self.input_size, self.hidden_sizes] if self.default_architecture else
                  [self.input_size, self.hidden_sizes, self.output_size, self.activations])
        h = hashlib.sha256(json.dumps(header).encode())
        for name, value in self.arrays().items():
            h.update(name.encode())
            h.update(value.astype("<f4").tobytes())
        return h.hexdigest()

    def snapshot(self):
        return {"hash": self.fingerprint(), **self.architecture(),
                "parameters": {k: v.tolist() for k, v in self.arrays().items()}}


def rebase_numeric_program(program, historical, current, budget, output_size=1):
    """Preserve an imported model's raw-input behavior under new numeric encoding.

    Only an affine change of coordinates, including median imputation, is applied;
    no weights are fitted. Feature order must match the historical artifact.
    The returned ordinary IR is recorded at iteration zero and replays independently.
    """
    if len(historical) != len(current) or any(p["kind"] != "numeric" for p in [*historical, *current]):
        raise ValueError("Historical preprocessing rebasing requires matching numeric features in order")
    model = apply_program(program, 2 * len(historical), budget, output_size=output_size)
    arrays = model.arrays()
    old_weight = arrays["layers.0.weight"].astype(float)
    weight = old_weight.copy()
    bias = arrays["layers.0.bias"].astype(float)
    for i, (old, new) in enumerate(zip(historical, current)):
        column = 2 * i
        weight[:, column] = old_weight[:, column] * new["scale"] / old["scale"]
        weight[:, column + 1] += old_weight[:, column] * (old["median"] - new["median"]) / old["scale"]
        bias += old_weight[:, column] * (new["mean"] - old["mean"]) / old["scale"]
    arrays.update({"layers.0.weight": weight, "layers.0.bias": bias})
    result = {"base": "zero", "hidden_sizes": model.hidden_sizes, "activations": model.activations,
              "rationale": program["rationale"] + " Rebased from historical numeric preprocessing without fitting.",
              "operations": [{"op": "set", "tensor": name, "value": value.tolist()} for name, value in arrays.items()]}
    # Validate transformed tensors against the same construction limits.
    apply_program(result, 2 * len(current), budget, output_size=output_size)
    return result


def apply_program(program, input_size, budget, best=None, frozen_architecture=None, output_size=1):
    if not isinstance(program, dict):
        raise ValueError("Program must be a JSON object")
    allowed = {"base", "base_hash", "hidden_sizes", "activations", "rationale", "operations"}
    if set(program) - allowed:
        raise ValueError("Unknown program keys; executable code is not accepted")
    if not isinstance(program.get("rationale"), str) or not program["rationale"].strip():
        raise ValueError("A mechanism rationale is required")
    hidden = program.get("hidden_sizes")
    activations = program.get("activations")
    if frozen_architecture is not None and hidden != frozen_architecture:
        raise ValueError("Architecture is frozen after the first valid candidate")
    base = program.get("base")
    if base == "zero":
        if program.get("base_hash") not in (None, ""):
            raise ValueError("Zero construction cannot reference a checkpoint")
        model = Network(input_size, hidden, budget, output_size, activations)
    elif base == "best":
        if best is None or program.get("base_hash") != best.fingerprint():
            raise ValueError("Missing or stale base_hash")
        if hidden != best.hidden_sizes or (activations is not None and activations != best.activations):
            raise ValueError("Patch architecture mismatch")
        model = deepcopy(best)
    else:
        raise ValueError("base must be zero or best")
    operations = program.get("operations")
    if not isinstance(operations, list) or len(operations) > 10_000:
        raise ValueError("operations must be a bounded list")
    tensors = dict(model.named_parameters())
    with torch.no_grad():
        for op in operations:
            if not isinstance(op, dict) or op.get("tensor") not in tensors:
                raise ValueError(f"Unknown tensor; available: {sorted(tensors)}")
            target = tensors[op["tensor"]]
            action = op.get("op")
            if action in {"fill", "set"}:
                if set(op) != {"op", "tensor", "value"}:
                    raise ValueError("fill/set needs only op, tensor, value")
                try:
                    value = torch.as_tensor(op["value"], dtype=torch.float32)
                except (TypeError, RuntimeError, ValueError) as exc:
                    raise ValueError("Invalid tensor value") from exc
                if not torch.isfinite(value).all() or (value.abs() > budget.max_abs_weight).any():
                    raise ValueError("Invalid or out-of-bounds weights")
                if action == "fill":
                    if value.ndim != 0:
                        raise ValueError("fill needs a scalar")
                    target.fill_(value.item())
                else:
                    if value.shape != target.shape:
                        raise ValueError(f"set shape mismatch for {op['tensor']}: expected {list(target.shape)}, got {list(value.shape)}")
                    target.copy_(value)
            elif action in {"set_entries", "add_entries"}:
                if set(op) != {"op", "tensor", "entries"} or not isinstance(op["entries"], list):
                    raise ValueError("entries operation needs op, tensor, entries")
                for entry in op["entries"]:
                    if not isinstance(entry, dict) or set(entry) != {"index", "value"}:
                        raise ValueError("Each entry needs index and value")
                    index = entry["index"]
                    if not isinstance(index, list) or len(index) != target.ndim:
                        raise ValueError("Index rank mismatch")
                    for i, size in zip(index, target.shape):
                        if not 0 <= integer(i, "index") < size:
                            raise ValueError("Index out of bounds")
                    value = entry["value"]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError("Entry value must be a finite number")
                    if abs(value) > budget.max_abs_weight:
                        raise ValueError("Entry exceeds weight bound")
                    if action == "set_entries":
                        target[tuple(index)] = value
                    else:
                        target[tuple(index)] += value
            else:
                raise ValueError("Only fill, set, set_entries, add_entries are supported")
        if any(not torch.isfinite(p).all() or (p.abs() > budget.max_abs_weight).any() for p in model.parameters()):
            raise ValueError("Final weights exceed numerical bounds")
    return model


HEADS = {
    "binary": "output_size is 1: a logit; probability = sigmoid(logit), class = probability >= threshold.",
    "multiclass": "output_size equals the number of classes (in the order of `classes`); probabilities = softmax(outputs).",
    "regression": "output_size is 1: the prediction itself, in the target's original units.",
    "clustering": "output_size equals n_clusters: one score per cluster; assignment = argmax. A nearest-centroid "
                  "layer is score_k = 2*c_k.x - |c_k|^2 (weights 2*c_k, bias -|c_k|^2).",
}


def program_guide(task="binary", dynamic_architecture=False):
    architecture = ("Architecture can change on any base=zero candidate. base=best requires matching architecture and hash."
                    if dynamic_architecture else
                    "After the first valid submission hidden_sizes is frozen. You can restart its weights with base=zero.")
    return f"""Return a JSON object, no code or markdown:
{{"base":"zero", "hidden_sizes":[8], "activations":["relu"], "rationale":"mechanism hypothesis", "operations":[]}}
hidden_sizes is a list of hidden layer widths (empty means a direct linear output).
activations is optional: one of relu, tanh, sigmoid, identity per hidden layer (default relu).
Output head: {HEADS[task]}
Every tensor starts at zero when base=zero; unassigned entries remain zero.
Tensors are layers.0.weight, layers.0.bias, ...; weights have shape [out,in].
Allowed operations:
{{"op":"fill","tensor":"layers.0.bias","value":0}}
{{"op":"set","tensor":"layers.0.weight","value":[[1,2],[3,4]]}}
{{"op":"set_entries","tensor":"layers.0.weight","entries":[{{"index":[0,1],"value":2}}]}}
{{"op":"add_entries","tensor":"layers.0.bias","entries":[{{"index":[0],"value":0.2}}]}}
Use base=best with base_hash equal to the supplied best hash to patch that checkpoint.
{architecture}
Design weight values directly by reasoning. No fitting, optimizers, gradient steps,
pretrained target weights, executable code, or requests for final-test data.
"""


PROGRAM_GUIDE = program_guide("binary")
