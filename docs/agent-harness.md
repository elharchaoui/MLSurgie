# The weight-space programming agent

[← Overview](../README.md) · [Benchmark](benchmark.md) · [Providers](providers.md) · [Reference](reference.md)

The optional **OpenHands harness** gives a coding agent what an ML engineer has: a goal, labeled data, a computer, a place to keep their own tools, and a playbook. MLSurgie provides no prebuilt construction scripts; the agent writes all of its tooling itself. What's controlled is the information boundary (no development or test rows) and the deliverable: a weight program that's validated, evaluated independently, frozen and exported.

## What the agent gets

| Location | Access | Contents |
|---|---|---|
| `/inputs/README.md` | read-only | What every file contains and the exact encoding math |
| `/inputs/task.json` | read-only | Goal, task type, classes, output size, features, preprocessing state, budgets, track, weight-program format |
| `/inputs/state.json` | read-only, refreshed each turn | Iteration, remaining budgets, best network (weights and hash), development metrics, feedback rows |
| `/inputs/data/support.csv` | read-only | All support rows: raw features plus label (or target) |
| `/inputs/data/support_encoded.csv` | read-only | The same rows exactly as the network sees them |
| `/skills/*.md` | read-only | The weight-space programming playbook: [IR](../src/mlsurgie/skills/network-ir.md), [policy](../src/mlsurgie/skills/policy.md), [ReLU circuits](../src/mlsurgie/skills/circuits.md), and skills for binary evidence, rare events/fraud, multiclass, regression, clustering and repair |
| `/workspace` | read-write, **persists across the whole run** | The agent's own modules, notes and scripts; deliverable at `/workspace/candidate.json` |

The tool runs **Python or shell** in a Docker container with numpy, pandas and scipy. The container has no network, a read-only root filesystem, all capabilities dropped, PID, memory and CPU limits, and a non-root user. `/workspace` is on `PYTHONPATH`, so a module written in one execution can be imported in the next.

The per-turn prompt is a short brief: the goal on the first turn, then development metrics and remaining budget. The data and full state live in files. Earlier versions put every support row into the prompt each turn, which exhausted token budgets after one candidate.

## A turn

1. The protocol refreshes `/inputs/state.json`. A previous turn's `candidate.json` is moved to `previous_candidate.json`, so it can never be silently resubmitted.
2. The agent explores, builds tools, constructs weights and checks them locally, using up to `--agent-steps` tool calls.
3. When the turn ends, the valid `/workspace/candidate.json` that exists at that moment is submitted. No file, or invalid JSON, counts as an invalid proposal; there's never a fitting fallback.
4. The fitting audit scans the workspace and this turn's executed sources. On the strict track, a flagged candidate is rejected with the flagged lines returned as feedback.
5. MLSurgie validates the program and evaluates it in a separate process. The feedback the policy allows arrives on the next turn.
6. `finish_run=true` ends the experiment. The best candidate is frozen, and the final test runs outside the agent's reach.

## Run it

```bash
python -m pip install -e '.[harness,baselines]'     # Python 3.12+
docker build -f containers/Dockerfile.constructor -t mlsurgie-constructor:local .
export OPENROUTER_API_KEY="YOUR_API_KEY"

# One task:
mlsurgie agent --config benchmarks/tasks/fraud.json \
  --model openrouter/YOUR_VENDOR/YOUR_MODEL --output runs/agent-fraud

# The suite, against trained baselines:
mlsurgie bench --editor agent --model openrouter/YOUR_VENDOR/YOUR_MODEL --output runs/bench-agent
```

The `openrouter/` prefix follows the SDK's LiteLLM convention. For other providers, pass their LiteLLM model ID with `--api-key-env` and optionally `--base-url`. Options include `--names anonymized` (hide feature semantics), `--policy open` (record audit flags without rejecting), `--image` (a custom construction image with other analysis libraries), and `--agent-steps N`.

## Deliverables

```bash
mlsurgie replay runs/agent-fraud      # rebuild the network from programs; verify hashes
mlsurgie predict runs/agent-fraud --dataset new.csv --output predictions.csv
```

The run directory also contains `agent/workspace/` (every tool the agent wrote), `construction/<iteration>/<call>/` (each executed source and its log), `agent-events/` (the SDK conversation), and audit results per iteration in `report.json`.

## Limits

- The no-fitting policy is enforced by a static audit and full source retention, not by mechanical proof.
- Exports are feed-forward networks (Linear plus a chosen activation per layer). Custom layers and graph topologies need future export support.
- The agent receives labeled support data and revised feedback; this is supervised, data-informed construction with adaptive model selection, not data-free learning.
