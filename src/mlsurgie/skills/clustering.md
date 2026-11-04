# Clustering

No labels. Outputs: one score per cluster; assignment = argmax. Selection metric: silhouette
on development rows (higher is better). Reference labels, if any, are used only in the final report.

## Nearest-centroid layer (exact)
argmin_k |x - c_k|² = argmax_k (2 c_k · x - |c_k|²), so a single linear layer with
weight row 2 c_k and bias -|c_k|² implements nearest-centroid assignment in encoded space.
Per-feature emphasis: scale both weights and centroid coordinates by w_j (Mahalanobis-diagonal).

## Choosing centroids by reasoning
- Inspect 1-D histograms of each standardized feature: multimodality reveals segment structure.
- Cross-tabulate two informative features in coarse bins to see joint modes.
- Place each centroid at a mode you identified (bin centers, or the mean of rows inside a region you
  defined by explicit thresholds). Describe each cluster in the rationale ("high spend, low visits").
- Features with no structure add noise to distances: give them weight 0.
- Iterating "assign then recompute means" is k-means fitting — not allowed on the strict track.
  Defining regions by your own thresholds and taking their means once is analysis.

## Check
Cluster sizes (avoid empty or tiny clusters), silhouette on support, and whether clusters are
interpretable. Use exactly `n_clusters` outputs.
