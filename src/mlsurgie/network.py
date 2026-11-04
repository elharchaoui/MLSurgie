"""Fixed-architecture networks constructed entirely by checked JSON operations."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .contracts import Budget


def integer(value, label):
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, budget=None):
        super().__init__()
        budget = budget or Budget()
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) > budget.max_hidden_layers:
            raise ValueError("Too many hidden layers or invalid hidden_sizes")
        for width in hidden_sizes:
            if not 1 <= integer(width, "width") <= budget.max_width:
                raise ValueError("Hidden width exceeds budget")
        sizes = [input_size, *hidden_sizes, 1]
        count = sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))
        if count > budget.max_parameters:
            raise ValueError("Parameter count exceeds budget")
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        # Meta construction avoids even temporary random weight initialization.
        self.layers = nn.ModuleList([nn.Linear(a, b, device="meta") for a, b in zip(sizes, sizes[1:])])
        self.to_empty(device="cpu")
        with torch.no_grad():
            for param in self.parameters():
                param.zero_()
                param.requires_grad_(False)
        self.eval()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x).squeeze(-1)

    def probabilities(self, x):
        with torch.inference_mode():
            logits = self(torch.as_tensor(x, dtype=torch.float32))
            if not torch.isfinite(logits).all():
                raise ValueError("Network produced nonfinite logits")
            return torch.sigmoid(logits).numpy().astype(float)

    def arrays(self):
        return {name: p.detach().cpu().numpy().copy() for name, p in self.named_parameters()}

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([self.input_size, self.hidden_sizes]).encode())
        for name, value in self.arrays().items():
            h.update(name.encode())
            h.update(value.astype("<f4").tobytes())
        return h.hexdigest()

    def snapshot(self):
        return {"hash": self.fingerprint(), "hidden_sizes": self.hidden_sizes,
                "parameters": {k: v.tolist() for k, v in self.arrays().items()}}


def apply_program(program, input_size, budget, best=None, frozen_architecture=None):
    if not isinstance(program, dict):
        raise ValueError("Program must be a JSON object")
    allowed = {"base", "base_hash", "hidden_sizes", "rationale", "operations"}
    if set(program) - allowed:
        raise ValueError("Unknown program keys; executable code is not accepted")
    if not isinstance(program.get("rationale"), str) or not program["rationale"].strip():
        raise ValueError("A mechanism rationale is required")
    hidden = program.get("hidden_sizes")
    if frozen_architecture is not None and hidden != frozen_architecture:
        raise ValueError("Architecture is frozen after the first valid candidate")
    base = program.get("base")
    if base == "zero":
        if program.get("base_hash") not in (None, ""):
            raise ValueError("Zero construction cannot reference a checkpoint")
        model = Network(input_size, hidden, budget)
    elif base == "best":
        if best is None or program.get("base_hash") != best.fingerprint():
            raise ValueError("Missing or stale base_hash")
        if hidden != best.hidden_sizes:
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
                raise ValueError("Unknown tensor")
            target = tensors[op["tensor"]]
            action = op.get("op")
            if action in {"fill", "set"}:
                if set(op) != {"op", "tensor", "value"}:
                    raise ValueError("fill/set needs only op, tensor, value")
                try:
                    value = torch.as_tensor(op["value"], dtype=torch.float32)
                except (TypeError, RuntimeError) as exc:
                    raise ValueError("Invalid tensor value") from exc
                if not torch.isfinite(value).all() or (value.abs() > budget.max_abs_weight).any():
                    raise ValueError("Invalid or out-of-bounds weights")
                if action == "fill":
                    if value.ndim != 0:
                        raise ValueError("fill needs a scalar")
                    target.fill_(value.item())
                else:
                    if value.shape != target.shape:
                        raise ValueError("set shape mismatch")
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


PROGRAM_GUIDE = """Return a JSON object, no code or markdown:
{"base":"zero", "hidden_sizes":[8], "rationale":"mechanism hypothesis", "operations":[]}
hidden_sizes is a list of hidden layer widths (empty means a linear logit).
All layers are Linear, with ReLU between layers, and a sigmoid probability output.
Every tensor starts at zero when base=zero; unassigned entries remain zero.
Tensors are layers.0.weight, layers.0.bias, ...; weights have shape [out,in].
Allowed operations:
{"op":"fill","tensor":"layers.0.bias","value":0}
{"op":"set","tensor":"layers.0.weight","value":[[1,2],[3,4]]}
{"op":"set_entries","tensor":"layers.0.weight","entries":[{"index":[0,1],"value":2}]}
{"op":"add_entries","tensor":"layers.0.bias","entries":[{"index":[0],"value":0.2}]}
Use base=best with base_hash equal to the supplied best hash to patch that checkpoint.
After the first valid submission hidden_sizes is frozen. You can restart its weights with base=zero.
Design weight values directly by reasoning. No fitting, optimizers, gradient steps,
pretrained target weights, executable code, or requests for final-test data.
"""
