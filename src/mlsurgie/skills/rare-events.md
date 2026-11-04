# Rare events: fraud, defaults, anomalies

With 1–3 % positives, accuracy is meaningless (predicting "never" scores 97 %+). Use
average precision (PR-AUC), recall at a precision, or log loss.

- **Start from the base rate:** bias = log(p/(1-p)) ≈ -3.5 to -4.5. A network that only knows the base
  rate is the right "zero" for a fraud model.
- **Hunt for signatures.** Profile positives separately: which conditions are many times more
  common among positives than negatives? Lift = P(condition | positive) / P(condition | negative).
  Let the data name the signatures; do not assume a particular pattern.
- **Encode signatures as AND units.** Rare events are often conjunctive: one condition alone is weak, the
  combination is strong. Use `ReLU(sum(conditions) - (n - 1))` and give it a large positive output
  weight ≈ log(lift of the combination).
- **Soft thresholds on amounts/velocity.** Use hinges on standardized amount or counts rather than hard
  steps, so ranking within the suspicious region is preserved (PR-AUC rewards good ordering).
- **Do not over-predict.** Every signature unit also fires on some legitimate rows; check its precision
  on support before raising its weight. Overlapping signatures should not simply add up.
- **Small positive counts.** With ~100 positives in support, conditional rates are noisy: prefer
  signatures supported by many positives, and shrink weights of rare cells.
