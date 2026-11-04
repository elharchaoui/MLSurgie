# Live OpenRouter / Claude Opus 5.5 harness test

This run used the supplied credential in process memory to authenticate against OpenRouter, then requested `anthropic/claude-opus-5.5` through the actual OpenHands SDK. No scripted model responses or manually supplied candidate weights were used. Construction code ran in the local Docker sandbox. Source review of the selected construction found explicit constants, arithmetic, and a support-scoring pass, with no parameter-fitting loop.

Titanic: 891 rows; split seed 42; 534 support / 178 development / 179 final test. The agent received 64 support examples plus support preprocessing and development feedback. The model is 22 inputs → 14 ReLU units → 1 sigmoid output, with 337 stored parameters.

| Metric | Development | Final test |
|---|---:|---:|
| Accuracy | 82.02% | 79.89% |
| ROC-AUC | 0.8344 | 0.8439 |
| Log loss | 0.4404 | 0.4551 |
| Brier | 0.1390 | 0.1436 |
| F1 | 0.7460 | 0.7231 |

Final test: 143/179 correct. Confusion matrix (actual rows, predicted columns; non-survival first): `[[96,14],[22,47]]`. Accuracy Wilson 95% interval: 73.42–85.10%, assuming independent observations.

## Budget and execution

Limits: three proposals, five SDK steps per proposal, 40,000 reported tokens, 4,096 output tokens per request, ten-minute construction deadline. First proposal was evaluated and selected. A second proposal exceeded the token budget and was not evaluated; the run froze the first candidate before final-test evaluation. Reported token usage reached 75,233, demonstrating the documented possibility of overshooting a total-token budget by an in-flight request. This was not a completed three-candidate comparison or an observed architecture-search result.

Elapsed construction/final evaluation: approximately 50.5 seconds. SDK-reported cost: $0.2812194; the provider invoice is authoritative. No automatic second run was made after opening the final test.

Replay verified hash `bd7332a5293d07c06d2531387fb020c7e0d7fdb70ac5867c3171aa629259ad68`; saved-model reload and inference also succeeded. A credential scan found no plaintext credential in run artifacts. Credential material was neither committed nor included in the construction container.

## Limits of the evidence

This demonstrates a functioning live model → construction tool → independent evaluator → export path. It is one familiar public dataset and one IID split. Titanic may be present in the model's pretraining, relatives can cross partitions, and arbitrary Python's no-fitting compliance is not mechanically proven. It does not establish general model quality or an improvement over the earlier interactive-assistant experiment.

## Published evidence and reproducibility

This run used framework commit `7ea4e6c`, before the request-token preflight improvement. The measured results and token overshoot above are unchanged historical results; they are not evidence of running the subsequent fix with a live model. The baseline comparisons in the earlier experiment use the same dataset and seed, but this run was not selected through a fresh multi-seed comparison.

Published files in this directory contain the selected model, explicit proposal, exact construction source, metrics, split manifest, and integrity checks. Raw examples, task contexts, SDK events/state, and credentials are excluded. `construction.py` is the exact sandbox source and expects `/inputs/task.json`; it is published for inspection, not as a standalone host executable.

From the repository root:

```bash
mlsurgie replay experiments/titanic/opus55
```

```python
from mlsurgie import load
predictor = load("experiments/titanic/opus55")
```

Replay needs no LLM, key, Docker, or original CSV. It verifies the saved tensors, not the original model's reasoning. The dataset and seed/split manifest are necessary to reproduce metrics separately; a new live Opus run can produce different programs and scores.
