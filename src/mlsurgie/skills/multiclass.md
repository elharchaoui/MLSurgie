# Multiclass classification

Outputs: one logit per class in task.json `classes` order; prediction = softmax.

- **Priors:** output bias_k = log(prior_k) (differences matter, not the absolute level).
- **Class evidence:** for each informative feature, read the class-conditional mean (numeric) or
  rate (categorical). Logit contributions can be read as log P(feature region | class).
- **Prototype construction (Gaussian-like classes):** with class means m_k and a shared scale,
  logit_k = (m_k · x) / σ² - |m_k|² / (2σ²) + log prior_k. That is a linear layer: weight m_k/σ²,
  bias -|m_k|²/(2σ²) + log prior_k. Use features where classes separate; zero the others.
  Choose σ² from the within-class spread you measured (larger σ² = softer probabilities).
- **One-vs-rest rules:** when one class is defined by a threshold (e.g. a feature very high), add a hidden
  unit for that condition feeding a positive weight to that class and negative weights to others.
- Check per-class recall on support; confusion between two classes points to the feature that
  separates exactly those two.
