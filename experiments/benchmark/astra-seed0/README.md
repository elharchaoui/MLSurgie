# Live runs: GPT-6 Astra agent on five benchmark tasks

On 2026-11-04 the OpenHands harness called `openrouter/openai/gpt-6-astra` on five suite tasks: seed 0, semantic names, strict track, with each task's token budget capped at **140,000** by `--max-editor-tokens` so the whole run fit the API key's spending limit. No scripted responses or hand-supplied weights were used. In every task the agent wrote its own tooling in `/workspace`: 160–224 lines per task, including its own forward pass, metrics, profiling scripts and versioned construction scripts. All are retained in `<task>/run/agent/workspace/`. **The fitting audit flagged nothing in any task.**

Total spend was **$3.72** by OpenRouter key usage, about 0.74M tokens.

## Results (final test, opened once after selection)

Baselines were fit on the same support rows and encoded inputs, with hyperparameters chosen on development (`<task>/baselines.json`).

| Task | Metric | **Astra network** | Logistic / ridge | Random forest | SVM / SVR | k-means / GMM | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| breast_cancer | log loss ↓ | 0.137 | **0.080** | 0.139 | 0.093 | — | Loses to LR and SVM, ties RF |
| | ROC-AUC ↑ | 0.993 | **0.996** | 0.994 | **0.996** | — | |
| wine | log loss ↓ | **0.0075** | 0.026 | 0.165 | 0.080 | — | Best log loss; all reach 100% accuracy |
| diabetes | RMSE ↓ | 57.85 | 56.87 | **54.15** | 57.51 | — | Behind every trained model; mean predictor scores 70.5 |
| segments | ARI ↑ | 0.981 | — | — | — | 0.981 / **1.000** | Ties k-means, below GMM |
| segments_repair | ARI ↑ | 0.981 (faulty start 0.383) | — | — | — | 0.981 / **1.000** | Repaired to the k-means level with a 2-value patch |

Stored numbers: breast_cancer 4,403 (71 hidden units), wine 7,413 (247 hidden units), diabetes 23 (linear), segments 52. The baselines go up to 47k for the random forests. See `results.md` for every metric.

## What the agent built

- **breast_cancer:** six "physical evidence channels" (log worst radius, worst concave points, texture, smoothness, symmetry, relative radius error). Each is centred between its two class means and scaled by pooled spread. Logarithms are approximated with ReLU secant segments, which is why there are 71 hidden units.
- **wine:** the first candidate used pooled Gaussian prototypes (development log loss 0.105). After reading the development errors, the agent gave each cultivar its own chemical spread (0.049). This is the one task where feedback produced a measured improvement.
- **diabetes:** a linear additive circuit. Slopes per standard deviation come from upper-minus-lower half contrasts within strata of the other risk factors: BMI 32.4, log-triglycerides 35.1, blood pressure 13.9, HDL −13.0. Overlapping evidence was discounted by hand. The development RMSE of 51.7 didn't carry over to test (57.85), consistent with a small development split (88 rows).
- **segments:** histogram troughs define four regions (online share > 0.55; otherwise basket > $100; otherwise spend > $2,250). The mean of each region is taken once and written as a nearest-centroid layer, and the irrelevant `age` feature gets weight 0. This follows the policy (no reassignment loop), but it's close to a single k-means step from well-chosen seeds.
- **segments_repair:** the agent diagnosed the faults and patched with `base: "best"`, changing 2 values. It found that cluster 1's intercept "swallows 615/960 support rows" and reset it to the exact analytic value −‖c‖². It also found cluster 3's "impossible negative basket prototype (about −$100)" and reversed that sign. It **missed** the third planted error, a small flipped `online_share` coefficient (0.50), which barely changes assignments.

## Caveats

- **One seed, mostly one candidate per task.** In every task the final turn exceeded the 140k cap and was correctly discarded (`budget_exceeded`). Only wine got two evaluated candidates. Higher budgets would buy iteration at roughly $1 per 140k tokens.
- **Familiar datasets.** breast_cancer, wine and diabetes are well-known public datasets that ran on the semantic track; Astra may know them from pretraining. The anonymized track is needed to separate reasoning from recall.
- **Same author.** The synthetic segments generator and the repair fixture were written by the same author as the skills. The clustering and repair skills describe general methods, not the planted segments.
- **Wine is near-saturated.** Every method reaches 100% accuracy, so only log loss separates them.

## Verify

```bash
for t in breast_cancer wine diabetes segments segments_repair; do
  mlsurgie replay experiments/benchmark/astra-seed0/$t/run
done
```

Support CSVs copied into `agent/inputs/data/` were removed; they are a deterministic function of `benchmarks/data/` and each manifest's split indices.
