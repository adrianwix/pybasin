"""Fixed-k, fixed-iteration k-means, built to trace under torch.export with a dynamic batch dim.

Demo-only approximation of the production HDBSCANClusterer/TunedClusterer (density-based,
auto-discovers cluster count and shape): this trades that adaptivity for a fixed k and a
static iteration count, which is what makes it embeddable in an exported computation graph
alongside feature extraction and classification (see case_study_graph.py).
"""

import torch
import torch.nn.functional as functional


def torch_kmeans(features: torch.Tensor, k: int, n_iters: int) -> torch.Tensor:
    """Cluster `features` with Lloyd's algorithm, run for a fixed number of iterations.

    Centroids are initialized from evenly-spaced positions in the batch itself (a plain
    slice, not a random draw, so the graph has no dependency on export-uncertain RNG
    support). Empty clusters keep their previous centroid rather than becoming NaN via a
    zero-count mean division.

    :param features: (B, F) feature tensor.
    :param k: Fixed number of clusters.
    :param n_iters: Fixed number of refinement iterations.
    :return: (B,) cluster index per sample, values in [0, k).
    """
    batch_size = features.shape[0]
    init_idx = torch.arange(k, device=features.device) * (batch_size // k)
    centroids = features[init_idx]

    for _ in range(n_iters):
        distances = torch.cdist(features, centroids, compute_mode="donot_use_mm_for_euclid_dist")
        assignment = torch.argmin(distances, dim=-1)

        one_hot = functional.one_hot(assignment, num_classes=k).to(features.dtype)
        counts = one_hot.sum(dim=0)
        sums = one_hot.t() @ features
        new_centroids = sums / counts.clamp_min(1).unsqueeze(-1)
        has_points = (counts > 0).unsqueeze(-1)
        centroids = torch.where(has_points, new_centroids, centroids)

    distances = torch.cdist(features, centroids, compute_mode="donot_use_mm_for_euclid_dist")
    return torch.argmin(distances, dim=-1)
