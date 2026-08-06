"""Exportable graph: features -> label/cluster index. Stage 3 of 3 -- see
integrate_graph.py's module docstring for why each pipeline stage is its own `.pte` (a demo
UI's classifier-choice control should only re-run this stage, reusing already-computed
features from stage 2).

- Supervised: nearest fixed template (real physical trajectories with known labels, e.g.
  "chaotic1") -- ``cdist`` + ``argmin`` against ``reference_features``, identical math to the
  production ``KNeighborsClassifier(n_neighbors=1)`` every case study already uses.
  ``reference_features`` is a runtime input, not a value baked into the graph at export time --
  since neither `n` (batch) nor `F` (feature count, which the demo UI's feature-selection
  control changes freely, see feature_extract_graph.py) is fixed, one exported
  `classify_supervised.pte` is reusable by every case study: each just passes its own
  offline-computed template features at call time.
- Unsupervised: a fresh :func:`torch_kmeans` clustering of *this batch's* features, fixed to
  the same cluster count as the case study's known number of attractors. Unlike supervised
  mode, a cluster index carries no fixed semantic identity across separate runs -- Lloyd's
  algorithm run fresh on a new batch has no reason to assign attractor A to index 0 twice in a
  row. That is an honest property of real unsupervised clustering (no external labels), not a
  bug to paper over; the demo UI should present unsupervised results as "N distinct groups
  found", not as stable named attractors. `n_clusters` is fixed at export time (one `.pte` per
  distinct cluster count, shared across case studies with the same attractor count).
"""

import torch
import torch.nn as nn
from torch_kmeans import torch_kmeans


class ClassifyGraph(nn.Module):
    """Classify (or cluster) a feature tensor, in either supervised or unsupervised mode.

    :param unsupervised: Selects clustering mode instead of nearest-template classification.
    :param n_clusters: Unsupervised mode only -- fixed cluster count for :func:`torch_kmeans`.
    :param n_kmeans_iters: Unsupervised mode only -- fixed refinement iteration count.
    """

    def __init__(
        self,
        unsupervised: bool = False,
        n_clusters: int | None = None,
        n_kmeans_iters: int = 10,
    ) -> None:
        super().__init__()
        if unsupervised and n_clusters is None:
            raise ValueError("n_clusters is required when unsupervised=True.")
        self.unsupervised = unsupervised
        if unsupervised:
            assert n_clusters is not None
            self.n_clusters = n_clusters
            self.n_kmeans_iters = n_kmeans_iters

    def forward(
        self, features: torch.Tensor, reference_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        :param features: ``(n, F)`` feature tensor to classify.
        :param reference_features: Supervised mode only -- ``(R, F)`` feature vectors of
            known-labeled template trajectories.
        :return: ``(n,)`` predicted label/cluster index per sample.
        """
        if self.unsupervised:
            return torch_kmeans(features, k=self.n_clusters, n_iters=self.n_kmeans_iters)

        assert reference_features is not None
        distances = torch.cdist(
            features, reference_features, compute_mode="donot_use_mm_for_euclid_dist"
        )
        return torch.argmin(distances, dim=-1)
