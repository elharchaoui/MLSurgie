# LLM provider configuration

[← Overview](../README.md) · [Getting started](quickstart.md) · [Technical reference](reference.md)

MLSurgie supports **Ollama** and **OpenRouter** as built-in live editors. The user always chooses an explicit model. The same evaluator, JSON weight language, budgets, and final-test protocol apply to both. No extra provider SDK is needed for `build`. The optional [`agent` harness](agent-harness.md) uses OpenHands SDK and its LiteLLM provider integration; its model argument uses `provider/model` notation.

## Which model runs?

| Entry point | Model used |
|---|---|
| `mlsurgie demo` | No LLM. Deterministic scripted proposals for a known threshold rule. |
| `build --programs proposals.json` | No LLM. User-supplied JSON proposals, labeled as a scripted fixture. |
| `build --ollama-model MODEL` | The exact installed Ollama model name supplied by the user. |
| `agent --model openrouter/VENDOR/MODEL` | OpenHands harness using its LiteLLM/OpenRouter integration, with containerized construction. |
| `build --openrouter-model VENDOR/MODEL` | The explicit OpenRouter model ID supplied by the user; OpenRouter routes to its serving providers. |
| Published Titanic experiment | An interactive assistant submitting programs through a file bridge; this is separate from the packaged live adapters. |

Neither a default model nor an automatically recommended model is bundled. The direct JSON adapter tests mock HTTP and have no model-specific benchmark. The optional OpenHands harness has a [live Opus 5.5 experiment](../experiments/titanic/opus55/README.md) on Titanic; its result does not establish general model quality. A successful JSON response can still contain invalid or poor weights.

## Ollama

Start your Ollama service and install the model you want. Pass its exact name from `ollama list`:

```bash
mlsurgie build \
  --config examples/titanic-seaborn.json \
  --dataset data/titanic-seaborn.csv \
  --ollama-model YOUR_INSTALLED_MODEL \
  --ollama-url http://localhost:11434 \
  --output runs/local-classifier
```

`--ollama-url` defaults to the local URL shown. The adapter sends JSON-mode chat requests without tools. Remote Ollama endpoints receive the same data context; the built-in adapter does not provide an authentication setting for them. [Ollama chat API](https://docs.ollama.com/api/chat).

## OpenRouter

Set an API key in the environment and specify an exact model ID from the [OpenRouter model catalog](https://openrouter.ai/models):

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
mlsurgie build \
  --config examples/titanic-seaborn.json \
  --dataset data/titanic-seaborn.csv \
  --openrouter-model YOUR_VENDOR/YOUR_MODEL \
  --output runs/hosted-classifier
```

PowerShell: `$env:OPENROUTER_API_KEY = "YOUR_API_KEY"`.

| Setting | Default / behavior |
|---|---|
| Model | Required `--openrouter-model`; no default |
| Credential | Environment variable `OPENROUTER_API_KEY` |
| Alternative key variable | `--openrouter-api-key-env MY_KEY_VARIABLE` reads that variable's value |
| Endpoint | Fixed `https://openrouter.ai/api/v1/chat/completions` |
| Output | Non-streaming JSON object, temperature 0; output limit from the run budget |
| Routing | `provider.require_parameters: true` asks OpenRouter to require support for requested parameters |
| Usage | Prompt/completion token counts recorded per iteration; response model recorded when returned |
| Errors | HTTP status recorded without provider response bodies; no application-level retries |

The key is never included in editor identity, serialized contexts, or run manifests. Do not place it in JSON configurations or version control. `.env` files are **not automatically loaded**; export variables yourself or load them before launching MLSurgie.

An unsupported model/parameter combination can fail. The adapter does not silently disable JSON mode or select a different model. OpenRouter's own host routing/fallback behavior is governed by OpenRouter. Provider errors stop the run; malformed weight programs are recorded and rejected by the normal validation loop.

API sources: [chat completions](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion), [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

## Python API

Choose one editor:

```python
from mlsurgie import OllamaEditor

editor = OllamaEditor(
    model="YOUR_INSTALLED_MODEL",
    base_url="http://localhost:11434",
)
```

Or, with `OPENROUTER_API_KEY` set in the process environment:

```python
from mlsurgie import OpenRouterEditor

editor = OpenRouterEditor(model="YOUR_VENDOR/YOUR_MODEL")
```

Pass the chosen editor as `editor=...` to `build(...)`. For an alternative environment variable:

```python
from mlsurgie import OpenRouterEditor

hosted_editor = OpenRouterEditor(
    model="YOUR_VENDOR/YOUR_MODEL",
    api_key_env="MY_KEY_VARIABLE",
)
```

See the [complete build example](reference.md#python-api).

## Data and cost

Live editors receive support examples, support preprocessing statistics, current best weights, and bounded development diagnostics. Hosted OpenRouter calls send that context to OpenRouter and its serving provider. Final-test examples are excluded from serialized editor contexts. Custom Python editors remain trusted in-process extensions.

Run budgets constrain submitted iterations, output tokens per response, and evaluated rows. Total-token (`max_editor_tokens`) and runtime (`max_seconds`) budgets are optional and unset by default. When set, a single request can overshoot a total-token budget; subsequent work stops. Token usage is recorded, while monetary totals remain unknown and there is no currency budget enforcement. Use provider-side spending controls for paid inference.

## Other integrations

The lightweight `build` command has no LiteLLM adapter or direct vendor SDK. The optional `agent` command uses LiteLLM through OpenHands SDK. OpenRouter remains available as a direct JSON editor without that optional dependency. New trusted editors can implement `identity` and `propose(context, timeout) -> Reply`; see [contributing](../CONTRIBUTING.md#adding-an-editor).
