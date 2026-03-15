from typing import TypedDict

import torch

from pybasin.solvers.torch_ode_system import ODESystem


class SimpleParams(TypedDict):
    a: float


class SimpleODE(ODESystem[SimpleParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return p[..., 0:1] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['a']} * y"


def test_ode_system_initialization():
    params: SimpleParams = {"a": -0.5}
    ode = SimpleODE(params)

    # Parameters stored correctly in ODE system
    assert ode.params == params


def test_ode_system_forward():
    params: SimpleParams = {"a": -0.5}
    ode = SimpleODE(params)

    t = torch.tensor(0.0)
    y = torch.tensor([[1.0]])
    dydt = ode(t, y)

    # Output shape matches input shape
    assert dydt.shape == y.shape
    # Derivative = -0.5 × 1.0 = -0.5 (correct calculation)
    assert torch.allclose(dydt, torch.tensor([[-0.5]]))


def test_ode_system_batched():
    params: SimpleParams = {"a": 2.0}
    ode = SimpleODE(params)

    t = torch.tensor(0.0)
    y = torch.tensor([[1.0], [2.0], [3.0]])
    dydt = ode(t, y)

    # Batched output has shape (3 batches, 1 state)
    assert dydt.shape == (3, 1)
    # Derivatives are 2×y for each batch (2×1=2, 2×2=4, 2×3=6)
    assert torch.allclose(dydt, torch.tensor([[2.0], [4.0], [6.0]]))
