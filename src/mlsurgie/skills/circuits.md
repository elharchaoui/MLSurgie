# Circuit cookbook (ReLU unless stated)

Write s = standardized input, t = standardized threshold. ReLU(u) = max(0, u).

| Shape | Construction |
|---|---|
| Linear evidence | output weight w on the input directly (no hidden unit needed) |
| Step at t (soft) | unit `ReLU(k(s - t)) - ReLU(k(s - t) - 1)` → 0 below t, 1 above t + 1/k (two units) |
| Hinge (effect only above t) | `ReLU(s - t)`; slope = output weight |
| Hinge below t | `ReLU(t - s)` (weight -1, bias t) |
| Interval / bump [a, b] | step at a minus step at b (four units), or `ReLU(1 - k|s - c|)` via two units |
| Absolute value | `|s - c| = ReLU(s - c) + ReLU(c - s)` |
| AND of binary features (one-hot) | `ReLU(f1 + f2 - 1)` → 1 only when both are 1 |
| AND of n conditions | `ReLU(sum(f_i) - (n - 1))` |
| OR (saturating) | `min(1, f1 + f2) = (f1 + f2) - ReLU(f1 + f2 - 1)` |
| Condition AND threshold | `ReLU(k(s - t) + M(f - 1))` with large M: active only if f = 1 and s > t |
| Min / max | `min(a,b) = a - ReLU(a - b)`; `max(a,b) = b + ReLU(a - b)` |
| Piecewise-linear f(s) | knots t_j: f(s) = f(t_0) + slope_0 (s - t_0) + Σ_j Δslope_j ReLU(s - t_j) |

Tips
- Identity pass-through in a hidden layer: `ReLU(s) - ReLU(-s) = s` (two units).
- Keep magnitudes moderate (|w| < ~20) unless you need hard gates; large weights make probabilities
  overconfident and log loss explodes on mistakes.
- Two hidden layers let layer 1 compute conditions and layer 2 combine them (AND of ORs).
- `tanh`/`sigmoid` hidden layers give smooth saturating gates when you want bounded evidence.
