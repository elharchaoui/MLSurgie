"""Independent inference worker: only validated tensors and evaluation rows enter."""
import json
import os
import subprocess
import sys

from .contracts import Budget
from .network import apply_program


def evaluate(model, x, timeout):
    program = {"base": "zero", "hidden_sizes": model.hidden_sizes,
               "rationale": "Evaluate frozen tensors", "operations": [
                   {"op": "set", "tensor": name, "value": value.tolist()}
                   for name, value in model.arrays().items()]}
    request = {"program": program, "input_size": model.input_size,
               "limits": {"max_parameters": sum(p.numel() for p in model.parameters()),
                          "max_hidden_layers": max(1, len(model.hidden_sizes)),
                          "max_width": max([1, *model.hidden_sizes]),
                          "max_abs_weight": max(1., max(float(abs(v).max(initial=0)) for v in model.arrays().values()))},
               "x": x.tolist()}
    result = subprocess.run([sys.executable, '-m', 'mlsurgie.evaluator'],
                            input=json.dumps(request, allow_nan=False), capture_output=True,
                            text=True, timeout=timeout,
                            env={**{key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'WINDIR')
                                    if key in os.environ},
                                 'OMP_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1'})
    if result.returncode:
        raise RuntimeError('Independent evaluator failed')
    import numpy as np
    probabilities = np.asarray(json.loads(result.stdout), dtype=float)
    if probabilities.shape != (len(x),) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid evaluator output')
    return probabilities


if __name__ == '__main__':
    request = json.load(sys.stdin)
    model = apply_program(request['program'], request['input_size'], Budget(**request['limits']))
    print(json.dumps(model.probabilities(request['x']).tolist(), allow_nan=False))
