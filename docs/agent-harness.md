# Agentic weight-space construction

[← Overview](../README.md) · [Quickstart](quickstart.md) · [Providers](providers.md)

The optional **OpenHands SDK harness** lets a coding agent investigate support data, run Python construction programs, change network depth and width between candidates, receive independent development feedback, and deliver a frozen network. The existing `mlsurgie build` command remains the lightweight JSON-only editor path.

This is the first implementation milestone, not an unrestricted neural architecture system. Export currently supports numeric/categorical binary classification with feed-forward Linear/ReLU networks and one sigmoid output. Architecture is no longer frozen in harness mode. Width, depth, parameters, iterations, tokens, and time remain explicit resource budgets. Custom activations, arbitrary PyTorch modules, and graph topologies require future export/runtime extensions.

## Install and prepare

Harness mode requires **Python 3.12+**, a running Docker engine on Linux or macOS, and a model supporting tool calls. The core package continues to support Python 3.11+.

```bash
git clone https://github.com/elharchaoui/MLSurgie.git
cd MLSurgie
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[harness]'
docker build -f containers/Dockerfile.constructor -t mlsurgie-constructor:local .
```

The optional dependency is pinned to OpenHands SDK 1.59.0, whose tool API is tested. No separate `openhands-tools` package is needed: MLSurgie registers its own construction tool. The supplied construction image contains Python and NumPy. Build your own compatible image and pass `--image IMAGE_NAME` to supply other libraries. Images are local and are never automatically pulled during agent execution.

## Give the agent a goal and data

Download the labeled Titanic dataset using the [quickstart](quickstart.md#2-download-labeled-data), then:

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
mlsurgie agent \
  --config examples/titanic-agent.json \
  --dataset data/titanic-seaborn.csv \
  --model openrouter/YOUR_VENDOR/YOUR_MODEL \
  --output runs/agent-titanic
```

Replace the model placeholder with a tool-capable model's exact ID. Here the **`openrouter/` prefix is required**, following the SDK's LiteLLM provider convention. It differs from the direct adapter's `build --openrouter-model VENDOR/MODEL` argument. There is no default model.

For other SDK-supported providers, use their LiteLLM model ID and set `--api-key-env YOUR_KEY_VARIABLE`. `--base-url` supplies an optional provider endpoint. Credential requirements and model compatibility depend on the provider. This harness uses the SDK's provider integration; the direct JSON adapters are separate.

The example allows eight evaluated submissions, up to 65,536 parameters, eight hidden layers, 256 neurons per layer, and 3,600 seconds for construction. `--agent-steps 12` bounds SDK steps per submission. The evaluator retains one primary metric and a fixed threshold declared in the config.

By default, the `agent` command exposes all support rows as labeled examples. Set `budget.support_examples` to reduce the context for large datasets. Development and final-test datasets are not placed in the construction container. Feature declarations, preprocessing, task classes, and the goal remain user-defined inputs.

## The execution loop

1. MLSurgie splits the supplied table and computes support-only preprocessing.
2. OpenHands receives the goal, support context, budgets, prior best weights, and bounded development feedback.
3. The agent calls the construction tool with Python source. Each execution gets a fresh writable `/work` and read-only `/inputs/task.json` and `/inputs/source.py`.
4. Python writes `/work/candidate.json` with explicit tensor operations. Output and errors return to the agent; it can repair code before finishing its turn. Scratch files are not carried between executions; the conversation retains tool history.
5. The last successful candidate is submitted when the agent ends its turn. A later failed execution clears that candidate. No candidate produces an invalid submission; it does not trigger a fitting fallback.
6. MLSurgie validates the tensors and architecture, runs inference in a separate Python process, computes development metrics, and supplies feedback on the next turn. Each submission, including invalid ones, consumes an iteration.
7. The agent may set `finish_run: true` on its construction tool to stop early, submitting the current candidate or retaining the prior best. On stop or budget exhaustion, the best candidate is frozen. Final-test inference runs independently, without another agent call. A final-evaluation failure is reported explicitly while preserving the frozen model.

The agent can change the architecture on any `base: "zero"` proposal. `base: "best"` patches still require matching topology and the best checkpoint's exact hash.

## Example construction source

The agent writes code such as the following inside its tool call. This example is a zero-logit control, not a useful classifier:

```python
import json
from pathlib import Path

task = json.loads(Path('/inputs/task.json').read_text())
program = {
    'base': 'zero',
    'hidden_sizes': [16, 8],
    'rationale': 'Zero-logit control before testing a mechanism hypothesis.',
    'operations': [],
}
Path('/work/candidate.json').write_text(json.dumps(program))
```

The source can perform arithmetic, inspect support examples, and construct complete tensors or sparse edits. The required export language is documented in [weight programs](reference.md#weight-programs).

## Deliverables

```bash
mlsurgie replay runs/agent-titanic
mlsurgie predict runs/agent-titanic \
  --dataset data/titanic-seaborn.csv --output predictions.csv
```

The normal network, preprocessing, checksums, metrics, and replay artifacts remain compatible with `load(...)`. Additional records include:

- `construction/ITERATION/CALL/inputs/source.py`: exact executed source, mounted read-only.
- `construction/ITERATION/CALL/inputs/task.json`: exact support and feedback context.
- `construction/ITERATION/CALL/execution.log`: construction output and errors.
- `construction/ITERATION/CALL/work/candidate.json`: submitted tensor artifact.
- `agent-events/`: persisted OpenHands conversation events.
- `manifest.json`: dynamic-architecture and independent-evaluation flags.
- `report.json`: construction assurance, actual metrics, usage, and errors.

Replay reconstructs tensors, not the agent's reasoning or arbitrary construction code. Generated runs contain dataset examples and transcripts; keep private runs private.

## Execution boundaries and evidence

Only the construction-call directory is mounted into its container. No Docker socket, host project tree, dataset path, provider key, or evaluator process is mounted or forwarded. Construction containers have no network, a read-only root, dropped capabilities, and CPU/memory/PID limits. The SDK orchestrates on the host with a custom construction tool and a finish tool; it receives no host-shell or file-editor tool. The evaluator executes trusted tensor inference, not agent-authored model code.

The instruction to avoid fitting is **a policy, not a proof**. Arbitrary Python can implement numerical fitting without importing an optimizer. Reports therefore label harness assurance as source-recorded and unverified for no-fitting compliance. Resource controls do not make this a proven adversarial sandbox; writable artifact storage and logs also require host capacity monitoring.

Token usage is checked between SDK events and submissions; one in-flight request can overshoot, and an SDK call can finish after its deadline. Candidates arriving beyond the outer construction budget are not evaluated. Final evaluation has its own timeout after selection. Monetary totals are not enforced as a spending budget. Provider errors and partial artifacts remain visible; do not infer success solely from a saved checkpoint.

The local verification covers actual SDK tool initialization, mocked model/runtime behavior, topology changes, separate-process inference, replay, and failure handling. The Docker smoke test in CI exercises actual container construction with a **scripted runtime**, not an LLM benchmark. Neither the earlier Titanic score nor the offline fixtures establish this harness's autonomous model quality. That requires a new live run with an explicitly selected model.

## Troubleshooting

- **Docker prerequisite error:** start Docker and build the local construction image. MLSurgie never falls back to host execution.
- **Missing SDK:** install `.[harness]` using Python 3.12+.
- **Missing credential:** set the named environment variable; `.env` is not loaded automatically by MLSurgie.
- **No valid candidate:** inspect iteration results, source, logs, and the agent transcript.
- **Model error:** use a model with tool-call support and confirm the SDK/LiteLLM provider ID and account access.

Implementation references: [OpenHands custom tools](https://docs.openhands.dev/sdk/guides/custom-tools), [SDK agent example](https://docs.openhands.dev/sdk/guides/hello-world), and [LLM API](https://docs.openhands.dev/sdk/api-reference/openhands.sdk.llm).
