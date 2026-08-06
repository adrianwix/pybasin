"""Export the shared feature-extraction graph, once per distinct `state_dim`.

`FeatureExtractGraph` depends only on `state_dim` (see its module docstring for why), so this
exports one `.pte` per state count actually used across case studies, not one per case study.
`t_steps` is standardized to `T_STEPS` for every case study so a shared export isn't
accidentally blocked by a mismatched save-point count.
"""

from pathlib import Path

import torch
from export_utils import MAX_BATCH, check_eager_vs_exported, export_and_save, mark
from feature_extract_graph import N_FEATURES_PER_STATE, FeatureExtractGraph

T_STEPS = 50


def export_for_state_dim(state_dim: int) -> None:
    mark(f"=== state_dim={state_dim} ===")
    module = FeatureExtractGraph(state_dim=state_dim)

    y_example = torch.randn(5, state_dim, T_STEPS)
    with torch.no_grad():
        features = module(y_example)
    mark(f"eager forward, features={tuple(features.shape)}")
    assert features.shape[1] == state_dim * N_FEATURES_PER_STATE

    batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    pte_path = Path(__file__).with_name(f"extract_features_dim{state_dim}.pte")
    exported, _ = export_and_save(module, (y_example,), {"y": {0: batch_dim}}, pte_path)

    y_test = torch.randn(37, state_dim, T_STEPS)
    check_eager_vs_exported(module, exported, (y_test,))


def main() -> None:
    for state_dim in (2, 3):
        export_for_state_dim(state_dim)


if __name__ == "__main__":
    main()
