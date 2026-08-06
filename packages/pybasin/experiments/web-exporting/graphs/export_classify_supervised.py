"""Export the shared supervised-classification graph.

`reference_features` is a runtime input, not baked into the graph (see classify_graph.py's
module docstring) -- this exports ONE `.pte`, with `n` (batch), `F` (feature count, since the
demo UI's feature-selection control changes it freely) and `R` (template count) all dynamic,
reused by every case study. No case-study-specific data is exported here.
"""

from pathlib import Path

import torch
from classify_graph import ClassifyGraph
from export_utils import MAX_BATCH, check_eager_vs_exported, export_and_save, mark


def main() -> None:
    module = ClassifyGraph(unsupervised=False)
    mark("ClassifyGraph construction")

    n_features_example = 6
    n_templates_example = 3
    features_example = torch.randn(5, n_features_example)
    reference_example = torch.randn(n_templates_example, n_features_example)
    with torch.no_grad():
        label_idx = module(features_example, reference_example)
    mark(f"eager forward, label_idx={label_idx.tolist()}")

    batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    feature_dim = torch.export.Dim("n_features", min=1, max=200)
    template_dim = torch.export.Dim("n_templates", min=1, max=50)
    pte_path = Path(__file__).with_name("classify_supervised.pte")
    exported, _ = export_and_save(
        module,
        (features_example, reference_example),
        {
            "features": {0: batch_dim, 1: feature_dim},
            "reference_features": {0: template_dim, 1: feature_dim},
        },
        pte_path,
    )

    features_test = torch.randn(37, n_features_example)
    reference_test = torch.randn(n_templates_example, n_features_example)
    check_eager_vs_exported(module, exported, (features_test, reference_test))


if __name__ == "__main__":
    main()
