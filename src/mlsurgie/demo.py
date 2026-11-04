"""Known-rule fixture. Does not simulate or claim an LLM reasoning result."""
import numpy as np

from .contracts import Budget, Feature, Goal
from .data import Table
from .editors import Reply
from .runner import build


class ThresholdFixture:
    identity = {"kind": "scripted_fixture", "model": "known-rule x > 0"}

    def propose(self, context, timeout):
        i = context["iteration"]
        if i > 3:
            raise StopIteration
        operations = []
        if i == 2:
            f = context["preprocessing"][0]
            # Program the known rule x > 0 using ReLU(z) - ReLU(-z).
            operations = [
                {"op": "set", "tensor": "layers.0.weight", "value": [[1, 0], [-1, 0]]},
                {"op": "set", "tensor": "layers.1.weight", "value": [[8 * f["scale"], -8 * f["scale"]]]},
                {"op": "set", "tensor": "layers.1.bias", "value": [8 * f["mean"]]},
            ]
        return Reply({"base": "zero", "hidden_sizes": [2],
                      "rationale": "Known rule x > 0, represented by a signed ReLU pair" if i == 2 else "Zero-logit fixture control",
                      "operations": operations}, {"input_tokens": 0, "output_tokens": 0})


def demo(output):
    rng = np.random.default_rng(17)
    values = rng.uniform(-2, 2, 400)
    rows = [{"x": float(x), "label": str(int(x > 0))} for x in values]
    table = Table(rows, [Feature("x", description="Input to the specified rule x > 0")],
                  Goal("Construct the known function x > 0", "label"))
    return build(table=table, editor=ThresholdFixture(), output=output, budget=Budget(iterations=3))
