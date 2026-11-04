# Contributing

The project is a research prototype focused exclusively on direct weight-space programming. Keep conventional target-model training out of the construction loop.

Useful contributions are benchmark tasks with planted structure, weight-space programming skills, editor adapters, reproducible failure cases, and numerical checks. Start with a small case before broadening abstractions.

## Local checks

```bash
pip install -e '.[dev]'
ruff check .
pytest -q
MLSURGIE_DOCKER_TEST=1 pytest -q tests/test_harness.py   # needs Docker and the constructor image
mlsurgie demo --output runs/my-demo
mlsurgie replay runs/my-demo
```

Tests must not contact a live provider. Mock HTTP at the adapter boundary; label scripted candidate generators as fixtures. For live experiments, record model version, request parameters, full usage, and failed attempts.

## Adding an editor

Implement an `identity` JSON object and `propose(context, timeout)` returning `mlsurgie.editors.Reply`. The context contains the task, support examples, development feedback allowed by the feedback policy, the current best parameters, and declared constraints. Optionally implement `prepare(bundle)`: it is called once with the full support data, and is how the agent harness writes data files instead of prompts. Return JSON weight programs; never return executable code to the protocol. Honor the timeout and output-token allowance. Report real token usage; adapters must fail rather than report zero. `StopIteration` means that no more proposals are available.

## Adding a benchmark task or a skill

Tasks live in `benchmarks/tasks/` (see [docs/benchmark.md](docs/benchmark.md#adding-a-task)). Run `mlsurgie bench --editor none` and confirm the task is neither saturated nor noise. Skills are Markdown knowledge in `src/mlsurgie/skills/` and are indexed in its `README.md`. They must not contain runnable construction scripts; the agent writes its own tools.

Custom editor plugins execute trusted Python in the process. Do not claim OS-level isolation. External built-in HTTP adapters receive only serialized contexts.

## Benchmark case template

Include dataset provenance and sharing rights, feature descriptions, target definition, intended prediction-time inputs, split/group recipe, metric, limits, editor identity, and all candidate outcomes. Separate planted defects from natural failures. Keep final test data away from editor feedback. Share minimal synthetic cases when real data cannot be redistributed.

## Review expectations

Demonstrate that patches preserve unrelated tensors, invalid submissions cannot mutate the baseline, preprocessing uses support data only, final evaluation cannot influence selection, and save/load/replay agree. New architecture support must include explicit parameter semantics and numerical edge cases.

Do not advertise an LLM result from fixture execution. Include null/negative findings. Agree on a license before distributing third-party data, checkpoints, or code in this repository.
