"""Independent inference worker: only validated tensors and evaluation rows enter."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .contracts import Budget
from .network import apply_program

SOURCE_ROOT = str(Path(__file__).resolve().parents[1])


def evaluate(model, x, timeout):
    """Raw outputs [rows, output_size], computed in a separate interpreter."""
    program = {"base": "zero", "hidden_sizes": model.hidden_sizes, "activations": model.activations,
               "rationale": "Evaluate frozen tensors", "operations": [
                   {"op": "set", "tensor": name, "value": value.tolist()}
                   for name, value in model.arrays().items()]}
    request = {"program": program, "input_size": model.input_size, "output_size": model.output_size,
               "limits": {"max_parameters": sum(p.numel() for p in model.parameters()),
                          "max_hidden_layers": max(1, len(model.hidden_sizes)),
                          "max_width": max([1, *model.hidden_sizes]),
                          "max_abs_weight": max(1., max(float(abs(v).max(initial=0)) for v in model.arrays().values()))},
               "x": np.asarray(x).tolist()}
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
    # Source-tree runs (no installation) still resolve the package; nothing else is inherited.
    env.update(OMP_NUM_THREADS="1", PYTHONNOUSERSITE="1", PYTHONPATH=SOURCE_ROOT)
    result = subprocess.run([sys.executable, "-m", "mlsurgie.evaluator"],
                            input=json.dumps(request, allow_nan=False), capture_output=True,
                            text=True, timeout=timeout, env=env)
    if result.returncode:
        raise RuntimeError("Independent evaluator failed")
    outputs = np.asarray(json.loads(result.stdout), dtype=float)
    if outputs.shape != (len(x), model.output_size) or not np.isfinite(outputs).all():
        raise ValueError("Invalid evaluator output")
    return outputs


if __name__ == "__main__":
    request = json.load(sys.stdin)
    model = apply_program(request["program"], request["input_size"], Budget(**request["limits"]),
                          output_size=request["output_size"])
    print(json.dumps(model.outputs(request["x"]).tolist(), allow_nan=False))
