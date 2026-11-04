# Contributing

The project is a research prototype focused exclusively on direct weight-space programming. Keep conventional target-model training out of the construction loop.

Useful contributions are reproducible failure cases, editor adapters, narrowly specified network architectures, and numerical checks. Start with a small case before broadening abstractions.

## Local checks

```bash
pip install -e '.[dev]'
pytest -q
mlsurgie demo --output runs/my-demo
mlsurgie replay runs/my-demo
```

Tests must not contact a live provider. Mock HTTP at the adapter boundary; label scripted candidate generators as fixtures. For live experiments, record model version, request parameters, full usage, and failed attempts.

## Adding an editor

Implement an `identity` JSON object and `propose(context, timeout)` returning `mlsurgie.editors.Reply`. The context contains only support examples, development feedback, the current best parameters, and declared constraints. Return JSON operations; never return executable code. Honor the timeout and output-token allowance. `StopIteration` means that no more proposals are available.

Custom editor plugins execute trusted Python in the process. Do not claim OS-level isolation. External built-in HTTP adapters receive only serialized contexts.

## Benchmark case template

Include dataset provenance and sharing rights, feature descriptions, target definition, intended prediction-time inputs, split/group recipe, metric, limits, editor identity, and all candidate outcomes. Separate planted defects from natural failures. Keep final test data away from editor feedback. Share minimal synthetic cases when real data cannot be redistributed.

## Review expectations

Demonstrate that patches preserve unrelated tensors, invalid submissions cannot mutate the baseline, preprocessing uses support data only, final evaluation cannot influence selection, and save/load/replay agree. New architecture support must include explicit parameter semantics and numerical edge cases.

Do not advertise an LLM result from fixture execution. Include null/negative findings. Agree on a license before distributing third-party data, checkpoints, or code in this repository.
