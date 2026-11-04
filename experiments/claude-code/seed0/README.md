# Interactive run: Claude Opus 5.5 (Claude Code) as the weight-space programmer

On 2026-11-04 Claude Opus 5.5, working in a Claude Code session, built networks for all six suite tasks by hand-writing their weights. It drove the standard agent harness through a file bridge (`../bridge.py`). The protocol was unchanged:
- support data as files, and a persistent workspace;
- every line of code executed in the Docker sandbox (no network, read-only inputs);
- the strict fitting audit;
- independent development evaluation;
- a sealed final test, scored once after the editor stopped.

No budget caps were set, and each task stopped when Claude called `finish`. Token usage isn't available from the host session.

`bridge-transcripts/<task>/turn-NNN/` holds every executed source (`exec-*.py`), its full output (`exec-*.out`), and the brief received at each turn. That is exactly what the editor saw.

## Results (final test)

| Task | Metric | **Claude (this run)** | Astra (GPT-6) | Opus agent | Best trained baseline | Other baselines |
|---|---|---:|---:|---:|---:|---|
| fraud | avg. precision ↑ | **0.783** | — | 0.746 | RF 0.780 | SVM 0.721, LR 0.668 |
| | log loss ↓ | **0.0355** | — | 0.0365 | SVM 0.046 | |
| breast_cancer | log loss ↓ | 0.116 | 0.137 | — | **LR 0.080** | SVM 0.093, RF 0.139 |
| wine | log loss ↓ | 0.0195 | **0.0075** | — | LR 0.026 | SVM 0.080, RF 0.165 |
| diabetes | RMSE ↓ | 58.87 | 57.85 | — | **RF 54.15** | ridge 56.87, SVR 57.51 |
| segments | ARI ↑ | 0.981 | 0.981 | — | **GMM 1.000** | k-means 0.981 |
| segments_repair | ARI ↑ | 0.981 (from 0.383) | 0.981 | — | **GMM 1.000** | k-means 0.981 |

All on seed 0 with identical splits and encoded inputs. Astra and Opus figures come from their published runs in `../../benchmark/astra-seed0/` and `../../fraud/opus55-live/`. The Astra fraud attempt is unpublished because it was cut off by insufficient provider credits.

**Scorecard against trained baselines:**
- **Beat every trained model:** fraud (AP and log loss) and wine (log loss).
- **Matched k-means:** segments and the repair task.
- **Lost:** breast cancer (beat the random forest, lost to LR and SVM) and diabetes (behind all three trained models).

Audit: 49 files scanned, **0 violations**. All six networks replay to their recorded hashes.

## What was built, and how feedback was used

| Task | Construction | Development rounds |
|---|---|---|
| fraud (478 params) | 11 sigmoid soft-step detectors → ReLU AND units for four signatures found in support lift tables, plus pass-through evidence units. Weights are weight-of-evidence (shrunk) and cell log-odds. | **2.** Round 1 AP 0.764. Feedback showed legitimate look-alikes at p = 0.999 (stacked evidence) and a $152 bust-out missed by a $180 step, in a range where support had no data. Round 2 recalibrated each signature analytically to its cell rate and moved the step to $125: AP 0.795. Remaining errors were camouflaged fraud, so Claude stopped. |
| breast_cancer (621) | Four near-independent channels (size, concave points, texture, smoothness). Each weight is the Gaussian log-likelihood-ratio slope Δμ/σ² with a 1/(1+ρ) correlation discount. ReLU hinges add the quadratic term where malignant spread exceeds benign spread (perimeter, area error). | 1 (development log loss 0.072). The rest of the iteration happened on support: calibration tables showed underconfidence, which the hinges fixed (support 0.142 → 0.102). |
| wine (2,343) | Per-cultivar diagonal Gaussian log-likelihoods over six channels. The quadratic term uses an exact ReLU piecewise-linear x² (knots every 0.5; the slope changes by 1 at each knot). Class variances are shrunk 30% toward pooled. | 1 (development log loss 0.009, 100% accuracy). A pooled-variance linear version was rejected on support first (0.124 → 0.093), because the broad cultivar 2 was being pulled into the tight ones. |
| diabetes (73) | Additive response in target units. Slopes were read once each, as top-40% minus bottom-40% contrasts of the residual from the previous features. Plus an HDL threshold hinge. | **2.** Development RMSE 53.5 → 52.9. The errors showed under-predicted low-HDL patients, and support residuals confirmed a threshold, plus under-weighted blood pressure. |
| segments (52) | Histogram gaps define four regions: online share > 0.6; basket > $87 (empty gap $66–115); spend > $2,200. Their one-pass means become a nearest-centroid layer, with age at 0. | 1 (development silhouette 0.428). |
| segments_repair (3 values changed) | `base: best` patch from three diagnostics. (1) The bias identity bias = −‖c‖² flagged cluster 1 (+6.22 excess). (2) Plausibility flagged an implied −$100 basket. (3) Implied coordinate vs the mean of its assigned rows, testing only a sign-flip hypothesis, flagged `online_share` in cluster 3. | 1 (development silhouette 0.059 → 0.428). All three planted faults were fixed; Astra fixed two. |

## Caveats: read before citing

- **Strong contamination on the synthetic tasks.** The same assistant wrote `benchmarks/make_datasets.py` and `benchmarks/make_repair.py`, earlier in this project. It knew the planted fraud scenarios, the four customer segments, and the exact repair faults. Every threshold and weight was still derived in the sandbox from support statistics, and the transcripts show it. But knowing the answer shapes which questions get asked; check 3 of the repair, the sign-flip test, is the clearest case. **The fraud, segments and repair results are not evidence of unaided discovery.**
- **Other contamination.** Before this run, Claude had seen the baseline test scores and the Astra and Opus rationales. It may also know breast cancer, wine and diabetes from pretraining.
- **The isolation was procedural for the editor itself.** Its code ran sandboxed, but Claude Code had host access, so not reading development or test rows was a commitment, not an OS boundary. The bridge transcripts are the audit trail.
- **One seed.** Test sets are small (37 wines, 89 diabetes rows, 115 breast-cancer rows), so most differences in the table are within sampling noise. Development estimates were optimistic for breast cancer (0.072 → 0.116 on test) and diabetes (52.9 → 58.9).

The fair conclusion: hand-reasoned networks can reach trained-model quality, beating it where the structure is rule-like (fraud) or prototype-like (wine), and losing where a smooth, many-feature boundary helps (breast cancer, diabetes). An uncontaminated test needs an agent that didn't write the tasks, multiple seeds, and the anonymized track.

## Verify

```bash
for t in fraud breast_cancer wine diabetes segments segments_repair; do
  mlsurgie replay experiments/claude-code/seed0/$t/run
done
```
