# Weight-space programming for classical ML: research dossier

Research date: 4 November 2026. Scope: the linked research post, released experimental records, independent experiments, existing tooling, and relevant classical ML predecessors. This is a public-source survey, not an experimental replication. Repository descriptions and reported scores have been inspected; their results have not been rerun here.

## Assessment

There is enough evidence to justify a focused research framework. There is not yet evidence to promise a general replacement for fitting classical ML models. The most credible initial applications are constructing models from explicit domain knowledge, repairing localized parameter faults, and editing interpretable models under measured constraints.

I interpret “classical ML” broadly as supervised classification and regression on tabular data, including linear models, additive models, trees, boosting, and small MLPs. Clustering, forecasting, and kernel methods should follow only after the supervised workflow is established.

The opportunity is a bridge between domain specifications and deployable estimators, with reproducible evidence for every edit. Generic tensor editing is already available. A new library needs stronger classical model semantics, construction primitives, and evaluation than a generic LLM wrapper.

## What the original work demonstrates

The linked work is a research blog post by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, published 2 November 2026. It defines weight-space programming as an LLM reasoning about and writing neural parameters without an optimizer. Its experiments use materially different information and tool budgets. [Original post](https://rikiyat.github.io/blog/weight-space-programming/)

| Experiment | Capability reported | Important boundary |
|---|---|---|
| 1: game policies | Identifies four games from raw weights | Tiny models; observation/action semantics supplied |
| 2: diffusion models | Identifies three generated shapes | Small synthetic setting |
| 3: character models | Identifies language and copying heads | Python analysis permitted; inference prohibited |
| 4: car controller | Repairs a failing policy in one reply | Scene and failure trace supplied |
| 5: MinAtar | Writes working networks from zero | Game rules and source supplied |
| 6: Go | Constructs and improves a larger policy | Python and budgeted evaluation |
| 7: sycophancy | Static diagnosis fails; attention probes help | Weights alone insufficient |
| 8: Qwen, plus Gemma | Repairs looping; no established Gemma gain | Does not establish general capability creation |

Source for this experiment map: [released records overview](https://github.com/RikiyaT/lm_understand-logs/blob/main/README.md). These are demonstrations on specific networks, not classical tabular benchmarks.

### The large-model result needs its full protocol

The Qwen experiment reports MATH-500 improving from 7.9% to 48.1% with the 50-pass edit. The underlying records show 158 turns, 80 Python executions, and 20.3 hours elapsed, including a 4.7-hour service interruption. A “pass” could generate answers to hundreds of problems or obtain activations from many texts. It is not one prediction. The edit addresses a severe looping failure under the chosen generation configuration. [Experiment 8 protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)

That record also distinguishes 131,072 touched parameters from 130,188 values that actually differ after bfloat16 rounding. This matters for our patch accounting: report touched entries, changed serialized entries, update rank, and functional effect separately. [Experiment 8 edit specification](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)

In Go, zero-feedback construction is separate from iterative editing. The submitted zero-pass network beat smaller PPO opponents; the 50-pass version beat the same-size PPO baseline. This supports compiling an algorithm into a network when the task has known structure, rather than learning unknown statistical relationships from a description. [Experiment 6 protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/README.md)

### Four operations our framework should distinguish

1. **Inspect:** infer or explain mechanisms from parameters and model metadata.
2. **Compile:** translate an explicit function or rule into parameters.
3. **Repair:** diagnose a concrete failure and apply a bounded parameter patch.
4. **Fit:** estimate unknown relationships from observations, with a solver, optimizer, or adaptive search.

These are proposed engineering categories. Compilation can work without labeled observations when a specification determines the function. Generic prediction problems usually contain unknown relationships; feature names alone do not determine coefficients. Also, avoiding gradients does not eliminate statistical learning or optimization: repeated validation-guided edits are adaptive model selection.

## Community implementations and experiments

These projects are extremely recent. All findings below describe what their authors report or what the inspected code contains.

| Project | Work observed | Relevance to this framework | Evidence limit |
|---|---|---|---|
| [Alan Turing Institute: weights_editing](https://github.com/alan-turing-institute/weights_editing) | Replicates reading and MinAtar writing across editors; adds shuffled tensors and altered games | Reuse its controls and failure accounting | Small run counts; subset of original experiments |
| [weightspace](https://github.com/leofmarciano/weightspace) | Python tensor-inspection/editing harness, probes, patches, rollback, replay, multiple providers | Evaluate as an optional backend | Early library; generic tensor focus |
| [Chandan Singh: autoresearch](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html) | Hand-constructed transformer programs for small tasks | Source of interpretable circuits and construction examples | Language evaluations have disclosed leakage |
| [Simon Willison](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/) | Local open-model experiment identifying game policies from weights | Test local editor feasibility | Reading demonstration, not classical model generation |
| [Model Booster Lab](https://github.com/lusob/model-booster) | Browser MLPs, LLM-written weights, practice budget, held-out levels | Useful interaction design for edit inspection | Game demo, not a tabular benchmark |
| [Source-checked tutorial](https://github.com/az9713/weight-space-programming) | Explains released records and includes weight-analysis labs | Onboarding material | Tutorial, not new model-performance evidence |

### Independent replication is promising but uneven

The Turing repository reports near-perfect small-network reading for two frontier editors. Its Astra MinAtar means reproduce strong Breakout and Asterix behavior, while Seaquest is about half the original score. Some other-editor writing runs exhaust the output budget. Shuffled-weight controls and action/board modifications are useful, although the shuffled conditions themselves have very few trials. These findings support capability on selected tasks and significant editor dependence. [Replication results](https://github.com/alan-turing-institute/weights_editing)

For our benchmark, retain unfinished submissions, invalid patches, timeouts, and regressions in the results. Do not report only successful edits. A behavior-preserving hidden-unit permutation is a different control from randomly shuffling tensor entries: the former tests representation dependence; the latter destroys the function.

### Existing harness: useful components, limited differentiation

The `weightspace` library offers tensor summaries/slices, Python execution, budgeted probes, sparse patch export, checkpoints, provider adapters, and transcript replay. Its README explicitly says its in-process sandbox is not a security boundary. It reports a Qwen editing experiment and a strong unedited thinking-disabled baseline; the edited models do not improve that non-thinking configuration. This reinforces the need to compare against cheap configuration changes before attributing value to parameter editing. [Library and experiments](https://github.com/leofmarciano/weightspace)

Its [quadrant-repair example](https://github.com/leofmarciano/weightspace/blob/main/examples/fix_quadrants.py) trains a small classifier, corrupts an output bias, and repairs it. Code inspection shows that the probe uses the same `x, y` returned by training, despite a “held-out” docstring. Without an LLM argument it uses a scripted response. It is a wiring illustration, not independent evidence of generalization or autonomous discovery. Use a fresh private dataset in our equivalent example.

### Algorithmic generation and evaluation failure

Singh reports successful construction on ten algorithmic tasks, with explanations of channels, heads, and neurons. The report also discloses contaminated language evaluation: evaluation examples were present in an accessible pool, and SNLI used logistic fitting on the evaluation examples. Some compression runs moved constants into buffers, gaming parameter count. [Author's report and disclosures](https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html)

Consequences for our design: keep final evaluation outside editor access, count buffers and lookup tables as well as trainable parameters, and distinguish analytic construction from fitting in the run manifest. An explanatory comment is a hypothesis about a mechanism until interventions or equivalence checks support it.

### Criticism that changes the engineering plan

Matthew Honnibal argues that the headline combines raw-weight reading with tool-assisted analysis under one capability claim. Sean Goedecke argues that targeted weight analysis and intervention are within human ML engineering practice. Neither objection implies the automation has no value. They imply that we should specify the allowed tools, information, baseline, and budget for every result. [Honnibal](https://honnibal.dev/blog/astra-read-write-misleading), [Goedecke](https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/)

## What exists for classical ML already

### Directly editable additive models: strongest practical connection

InterpretML documents direct editing of an EBM's learned term-score arrays. It also provides monotonicity controls and postprocessing. Thus parameter-level editing of classical predictors is already an established capability. The proposed new contribution is translating a domain request into an appropriate, evaluated edit. [InterpretML FAQ](https://interpret.ml/docs/faq.html)

Use the documented `term_scores_` attribute from the [API reference](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html); the FAQ contains inconsistent pluralization in places. Treat missing bins, feature interactions, intercept centering, and cached uncertainty statistics as part of an adapter's responsibilities. Editing one univariate term alone cannot guarantee global monotonicity when interactions still involve that feature.

Neural Additive Models combine feature-specific networks additively. They offer a useful neural target with localized feature effects, but their original method trains those networks. They are prior art for architecture and interpretation, not proof of LLM parameter synthesis. [NAM project](https://neural-additive-models.github.io/)

TensorFlow Lattice supplies constrained layers supporting shape restrictions such as monotonicity and convexity. It is a reference for ensuring properties through parameterization or projection rather than relying on natural-language assurances. [Lattice repository](https://github.com/tensorflow/lattice)

### Generating classifiers without per-task gradient training

HyperFast uses a meta-trained hypernetwork to generate task-specific classifier parameters from a labeled support dataset. This is particularly relevant: it is a concrete implementation of model-weight generation for tabular classification. It differs from a general LLM reasoning from instructions and has substantial upstream training. [HyperFast paper](https://arxiv.org/abs/2402.14335), [implementation](https://github.com/AI-sandbox/HyperFast)

TabPFN is a necessary baseline for fast tabular learning. It predicts using a pretrained foundation model and dataset context; ordinary usage does not synthesize a fresh small model by directly writing its weights. Pin a checkpoint and protocol. The current repository defaults to 3.5 and distinguishes code licensing from model-weight licensing. [TabPFN implementation](https://github.com/PriorLabs/TabPFN), [2025 foundational paper](https://www.nature.com/articles/s41586-024-08328-6)

Agentic-imodels evolves scikit-learn-compatible regressors optimized for accuracy and agent-facing interpretability. Its connection is readable model representations and automated model development; that is adjacent to, rather than equivalent to, direct parameter programming. [Research paper](https://arxiv.org/abs/2605.03808)

### Program-to-weight compilers

Tracr converts RASP programs into transformer weights, providing known mechanisms for interpretation. Its original repository is archived. It is a conceptual reference for an intermediate representation and deterministic lowering, rather than a dependency to adopt uncritically. [Tracr](https://github.com/google-deepmind/tracr)

torchwright converts Python computation graphs into transformer checkpoints. Its documentation describes both exact and approximated operators, range-dependent numerical checks, and graph-level probes. It shows how a compiler can expose an error budget. It is not evidence that a transformer is the efficient deployment format for a tabular scorecard. [torchwright](https://github.com/physicsrob/torchwright)

Hummingbird converts trained classical models and preprocessing into tensor computations, including tree ensembles. This bridges deployment formats, but conversion does not make tree routing parameters interchangeable with ordinary MLP weights. Preserve model structure and comparison semantics when editing. [Hummingbird](https://github.com/microsoft/hummingbird), [compiler paper](https://arxiv.org/abs/2010.04804)

### Learning over weights and combining models

Neural Functional Networks process other networks' parameters while respecting hidden-unit permutations. Universal Neural Functionals extend this architectural approach. They motivate structural tensor metadata and permutation tests in a weight editor, but involve learned weight-processing models. [NFNs](https://papers.neurips.cc/paper_files/paper/2023/hash/4e9d8aeeab6120c3c83ccf95d4c211d3-Abstract-Conference.html), [UNFs](https://arxiv.org/abs/2402.05232)

Git Re-Basin aligns hidden units before weight-space merging. It illustrates why naive averaging across independently trained networks is unsafe. Alignment alone does not guarantee compatible features, labels, or behavior. Defer cross-model merging from the initial framework. [Git Re-Basin](https://github.com/samuela/git-re-basin)

ROME supplies a targeted model-editing precedent, but its implementation includes optimization. A structured low-rank update is useful prior art; it should not be labeled training-free merely because the final write is compact. [ROME implementation](https://github.com/kmeng01/rome/blob/main/rome/README.md)

## Coverage and remaining uncertainty

Searches covered the exact phrase and variants with classical ML, tabular, regression, classification, scikit-learn/sklearn, XGBoost, random forest, replication, framework, and GitHub, followed by primary papers and source repositories. Primary sources take precedence over news summaries. Search results included many repeated headlines; those are not independent evidence.

I did not locate a convincing public benchmark demonstrating LLM weight-space programming broadly improving classical tabular estimators. This is a finding about this search, not proof that no such work exists. In particular, neither game networks nor arithmetic circuits establish performance on unknown real-world statistical relationships.

A ProgramAsWeights comparison page appeared in search, but its article could not be retrieved, and its homepage exposed insufficient detail. It is a watchlist item, not evidence used for this architecture. Further conjectural image-restoration proposals were outside the classical ML scope and do not establish implemented capability.

Major unanswered questions:

- Does reasoning about parameters beat a deterministic repair with equivalent information?
- When does meaningful feature metadata help, and when does it introduce unjustified priors?
- How much does adaptive probe selection overfit the development set?
- Do edits survive feature reordering, hidden-unit permutations, and serialization?
- Can the approach help naturally occurring failures, beyond planted corruptions?
- Does total editing cost beat refitting, calibration, or a small hyperparameter search?
- Does domain-guided construction work with few labels on fresh tasks, without benchmark memorization?

These are the framework's initial research objectives. The implementation plan is in [framework-plan.md](archive/framework-plan.md).
