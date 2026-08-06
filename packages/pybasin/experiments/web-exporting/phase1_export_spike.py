"""Phase 1 spike: does torch.export trace TorchDiffEqSolver's fixed-step rk4 path?

Standalone script (not a pytest test) — run manually and record the result in README.md.
Uses the same harmonic-oscillator system as tests/solvers/test_torchdiffeq_solver.py so the
only variable under test is export-tracing mechanics, not ODE-specific behavior.
"""

import torch
import torch.nn as nn
from executorch.exir import to_edge  # type: ignore[import-untyped]
from torch.export import Dim
from torchdiffeq import odeint  # type: ignore[import-untyped]


class HarmonicStepModule(nn.Module):
    """Wraps a single fixed-step rk4 odeint call as an export target.

    t_span/t_eval/step_size are fixed at construction time (mirrors how a real case study's
    time window and integration step are fixed per exported graph, not runtime inputs) — only
    the batch of initial conditions is a runtime input.
    """

    def __init__(self, t_span: tuple[float, float], t_eval: tuple[float, float], step_size: float):
        super().__init__()
        save_ts = torch.linspace(t_eval[0], t_eval[1], 5)
        if t_eval[0] > t_span[0]:
            anchor = torch.tensor([t_span[0]])
            ts = torch.cat([anchor, save_ts])
        else:
            ts = save_ts
        self.register_buffer("ts", ts)
        self.step_size = step_size

    def ode_func(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, v = y[..., 0], y[..., 1]
        return torch.stack([v, -x], dim=-1)

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        y_all = odeint(
            self.ode_func,
            y0,
            self.ts,
            method="rk4",
            options={"step_size": self.step_size},
        )
        # Drop the prepended anchor point, matching TorchDiffEqSolver._integrate's slicing.
        n_save = self.ts.shape[0] - 1
        return y_all[-n_save:].permute(1, 2, 0).contiguous()


def main() -> None:
    module = HarmonicStepModule(t_span=(0.0, 20.0), t_eval=(18.0, 20.0), step_size=0.0005)
    module.eval()

    y0_example = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    batch_dim = Dim("batch", min=1, max=1000)

    print("Attempting torch.export.export...")
    exported = torch.export.export(
        module,
        (y0_example,),
        dynamic_shapes={"y0": {0: batch_dim}},
    )
    print("torch.export.export SUCCEEDED")
    print(exported.graph_module.graph)

    y0_test = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    eager_out = module(y0_test)
    exported_out = exported.module()(y0_test)
    max_abs_diff = (eager_out - exported_out).abs().max().item()
    print(f"Max abs diff eager vs exported (batch=3, differs from example batch=2): {max_abs_diff}")

    print("Attempting executorch.exir.to_edge(...).to_executorch()...")
    edge_program = to_edge(exported)
    et_program = edge_program.to_executorch()
    pte_path = "/tmp/harmonic_rk4_spike.pte"
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"ExecuTorch lowering SUCCEEDED, wrote {pte_path}")


if __name__ == "__main__":
    main()
