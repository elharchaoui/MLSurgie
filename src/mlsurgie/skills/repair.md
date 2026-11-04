# Repairing a provided network

state.json `best` holds the starting network (weights, architecture, hash) and its development
metrics. Your job is to find what is wrong and fix it with a minimal, explained patch.

1. **Reproduce.** Load the weights into your NumPy forward pass and score them on support.
2. **Localize.** Inspect each unit: its input weights (which features), bias, and output weights.
   Compute each unit's activation statistics on support. Look for units with implausible
   magnitudes, wrong signs, dead units, or outputs that dominate.
3. **Attribute errors.** Group support errors by which unit/cluster/class they involve.
4. **Patch.** Use `base: "best"` with the exact `base_hash` and `set_entries` on the faulty entries.
   Smaller patches are better evidence of understanding than rebuilding from zero.
5. **Explain.** The rationale names the fault and why the patch fixes it.
