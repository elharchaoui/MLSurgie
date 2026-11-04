# Live run: Opus 5.5 agent on the fraud benchmark

On 2026-11-04 the OpenHands harness called `openrouter/anthropic/claude-opus-5.5` on `benchmarks/tasks/fraud.json` (seed 0, semantic names, strict track). No scripted responses or hand-supplied weights were used. The agent read the support data, wrote its own tools (`run/agent/workspace/bench.py`, `build.py`, `load.py`: 79 lines in total), constructed one network, and was stopped by the token preflight before a second candidate.

## Result (final test, 2,401 transactions, 50 frauds)

| Method | Average precision ↑ | ROC-AUC ↑ | Log loss ↓ | Accuracy | Stored numbers |
|---|---:|---:|---:|---:|---:|
| **Opus 5.5 agent, written weights** | 0.7458 | 0.8864 | **0.0365** | **99.38%** | **265** |
| Logistic regression | 0.6675 | **0.8971** | 0.0506 | 98.79% | 28 |
| Random forest (300 trees) | **0.7797** | 0.8748 | 0.0487 | 99.25% | 129,708 |
| RBF SVM | 0.7210 | 0.8167 | 0.0460 | 99.21% | 8,961 |
| Prior | 0.0208 | 0.5000 | 0.1012 | 97.92% | 1 |

Baselines were fit on the same 7,199 support rows and 27 encoded inputs, with hyperparameters chosen on development. The network's confusion matrix is `[[2350, 1], [14, 36]]`: 97.3% precision and 72% recall at threshold 0.5. Development AP was 0.7414.

**Reading:** a network written entirely by reasoning, with no fitting, beat logistic regression and the SVM on average precision. It had the best log loss and accuracy of every model, while using 0.2% of the random forest's size. The random forest kept the best average precision. This is **one seed and one candidate**, so treat it as a demonstration, not a ranking.

## What the agent built

From its rationale (`run/iterations/001/proposal.json`):

- **Layer 1:** 8 sigmoid soft-step detectors: new account (< 35 days), card not present, risky merchant (electronics, gaming, gift cards), large amount (> $180), velocity (> 6.5 transactions in 24h), tiny amount (< $11), night (< 5:30), and far from home (> 250 km).
- **Layer 2:** 4 ReLU AND units for three signatures: new account + card not present + risky merchant + large amount; card testing (velocity + tiny amount + card not present); night + far + card not present.
- **Output:** bias −5.4 (log-odds of the residual fraud rate outside the signatures). Signature weights were set from the log-odds of each signature cell, shrunk toward the base rate.

The audit found no fitting patterns across 20 scanned files. Thresholds and weights were chosen from support-data profiling (rates, conditional counts, calibration bins), which the retained sources show.

## Caveats you must read

- **The skill text hinted at the planted structure.** At the time of this run, `skills/rare-events.md` listed example fraud signals ("new account + high amount, card-not-present + foreign, bursts of transactions, unusual hours"). These overlap with the scenarios planted by `benchmarks/make_datasets.py`; the same author wrote both. The agent never saw the generator, and the thresholds it chose came from the data. Still, the hint may have directed its search. The skill has since been neutralized. A clean rerun, and the `--names anonymized` track, are required before claiming the agent discovered the signatures unaided.
- **One candidate.** The first turn consumed 322k input tokens over 16 tool calls, because the SDK resends the growing conversation on every call. The 400k budget then stopped the run. Total usage was 370,888 tokens (355,487 input, 15,401 output). Monetary cost wasn't recorded by the SDK in this run.
- One seed, one split, and a synthetic dataset. Fraud data in the wild is messier.

## Reproduce and verify

```bash
mlsurgie replay experiments/fraud/opus55-live/run    # 1766a4d1…cb18ff6
mlsurgie bench --editor agent --model openrouter/anthropic/claude-opus-5.5 \
  --suite <suite with benchmarks/tasks/fraud.json> --seeds 0 --output runs/rerun   # a new experiment
```

The copied support CSVs were removed from `run/agent/inputs/data/` to save space; they are a deterministic function of `benchmarks/data/fraud.csv` and the split indices in `run/manifest.json`. `run/agent-events/` holds the full SDK conversation.
