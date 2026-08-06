# pyright: basic
from __future__ import annotations

import torch
import torch.nn as nn
from torch_kmeans import torch_kmeans


def test_separates_well_separated_clusters() -> None:
    torch.manual_seed(0)
    cluster_a = torch.randn(50, 2) + torch.tensor([10.0, 10.0])
    cluster_b = torch.randn(50, 2) + torch.tensor([-10.0, -10.0])
    cluster_c = torch.randn(50, 2) + torch.tensor([10.0, -10.0])
    features = torch.cat([cluster_a, cluster_b, cluster_c], dim=0)

    assignment = torch_kmeans(features, k=3, n_iters=10)

    assert assignment.shape == (150,)
    # Each well-separated group should be assigned a single, consistent cluster id.
    assert len(assignment[:50].unique()) == 1
    assert len(assignment[50:100].unique()) == 1
    assert len(assignment[100:].unique()) == 1
    assert assignment[0] != assignment[50]
    assert assignment[0] != assignment[100]
    assert assignment[50] != assignment[100]


def test_traces_under_torch_export() -> None:
    class Wrapper(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch_kmeans(features, k=3, n_iters=5)

    module = Wrapper()
    module.eval()

    torch.manual_seed(0)
    example = torch.randn(500, 4)
    batch_dim = torch.export.Dim("batch", min=3, max=1000)

    exported = torch.export.export(module, (example,), dynamic_shapes={"features": {0: batch_dim}})

    test_input = torch.randn(50, 4)
    eager_out = module(test_input)
    exported_out = exported.module()(test_input)
    assert torch.equal(eager_out, exported_out)
