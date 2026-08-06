"""Export the unsupervised classification (clustering) graph, once per distinct cluster
count -- see classify_graph.py's module docstring. `F` (feature count) is dynamic, like
classify_supervised.pte, so this is also shared across every case study with the same
attractor count, not tied to one case study's fixed feature spec.
"""

from pathlib import Path

import torch
from classify_graph import ClassifyGraph
from export_utils import MAX_BATCH, check_eager_vs_exported, export_and_save, mark


def export_for_n_clusters(n_clusters: int) -> None:
    mark(f"=== n_clusters={n_clusters} ===")
    module = ClassifyGraph(unsupervised=True, n_clusters=n_clusters)

    n_features_example = 6
    features_example = torch.randn(5 * n_clusters, n_features_example)
    with torch.no_grad():
        label_idx = module(features_example)
    mark(f"eager forward, label_idx={label_idx.tolist()}")

    batch_dim = torch.export.Dim("batch", min=n_clusters, max=MAX_BATCH)
    feature_dim = torch.export.Dim("n_features", min=1, max=200)
    pte_path = Path(__file__).with_name(f"classify_unsupervised_k{n_clusters}.pte")
    exported, _ = export_and_save(
        module,
        (features_example,),
        {"features": {0: batch_dim, 1: feature_dim}},
        pte_path,
    )

    features_test = torch.randn(11 * n_clusters, n_features_example)
    check_eager_vs_exported(module, exported, (features_test,))


def main() -> None:
    for n_clusters in (2,):
        export_for_n_clusters(n_clusters)


if __name__ == "__main__":
    main()
