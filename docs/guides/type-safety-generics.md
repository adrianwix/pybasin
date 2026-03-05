# Type Safety and Generics in pybasin

## Overview

pybasin uses Python's generic type system to provide strong type safety for ODE parameters across the entire library. This guide explains how to use generics effectively when extending pybasin with your own ODE systems.

## Why Generics?

The generic type system provides:

1. **Type Safety**: Static type checkers (mypy, pyright) can verify parameter types at development time
2. **IDE Autocomplete**: Your IDE will know exactly which parameters are available
3. **Self-Documentation**: Types serve as documentation for what parameters an ODE system expects
4. **Refactoring Safety**: Renaming or changing parameter types will show errors across your codebase

## How to Define a New ODE System

### Step 1: Define Your Parameter Type

Use `TypedDict` to define the exact parameters your ODE system needs:

```python
from typing import TypedDict

class MyODEParams(TypedDict):
    """Parameters for my ODE system."""
    alpha: float      # damping coefficient
    beta: float       # forcing amplitude
    omega: float      # forcing frequency
    initial_mass: float  # initial mass
```

**Benefits:**

- Type checkers will enforce that all required keys are present
- IDE will autocomplete parameter names
- Docstrings on each field document what they mean

### Step 2: Create Your ODE System

Inherit from `ODESystem[YourParamsType]`:

```python
from pybasin.ode_system import ODESystem
import torch

class MyODE(ODESystem[MyODEParams]):
    def __init__(self, params: MyODEParams):
        super().__init__(params)
        # self.params is now typed as MyODEParams!

    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Type checker knows these keys exist
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        omega = self.params["omega"]

        # ... your ODE logic here ...
        dy_dt = -alpha * y + beta * torch.sin(omega * t)
        return dy_dt

    def get_str(self) -> str:
        return f"My ODE with α={self.params['alpha']}"
```

### Step 3: Use Type-Safe Parameters Everywhere

When creating classifiers or other components, you can pass your typed parameters:

```python
from pybasin.predictors.knn_classifier import KNNClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create your parameters with full type safety
params: MyODEParams = {
    "alpha": 0.1,
    "beta": 1.0,
    "omega": 2.0,
    "initial_mass": 1.5,
}

# Type checker ensures params matches MyODEParams
ode_system = MyODE(params)

# Pass the parameters to the classifier
knn_classifier = KNNClassifier(
    classifier=KNeighborsClassifier(n_neighbors=3),
    template_y0=[[0.0, 1.0], [1.0, 0.0]],
    labels=["stable", "unstable"],
    ode_params=params,
)
```

## Complete Example: Pendulum System

Here's the pendulum example showing full type safety:

```python
# 1. Define parameters
from typing import TypedDict

class PendulumParams(TypedDict):
    """Parameters for the pendulum ODE system."""
    alpha: float  # damping coefficient
    T: float      # external torque
    K: float      # stiffness coefficient

# 2. Create ODE system
from pybasin.ode_system import ODESystem
import torch

class PendulumODE(ODESystem[PendulumParams]):
    def __init__(self, params: PendulumParams):
        super().__init__(params)

    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # IDE autocompletes these parameter names!
        alpha = self.params["alpha"]
        T = self.params["T"]
        K = self.params["K"]

        theta = y[..., 0]
        theta_dot = y[..., 1]

        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + T - K * torch.sin(theta)

        return torch.stack([dtheta_dt, dtheta_dot_dt], dim=-1)

    def get_str(self) -> str:
        return (
            f"Pendulum ODE:\n"
            f"  dθ/dt = ω\n"
            f"  dω/dt = -{self.params['alpha']}ω + "
            f"{self.params['T']} - {self.params['K']}sin(θ)"
        )

# 3. Use with type safety
def setup_pendulum():
    # Type checker verifies all required keys are present
    params: PendulumParams = {
        "alpha": 0.1,
        "T": 0.5,
        "K": 1.0,
    }

    # If you forget a parameter or misspell it, you'll get a type error!
    # params: PendulumParams = {"alpha": 0.1}  # ERROR: Missing 'T' and 'K'

    ode = PendulumODE(params)
    return ode
```

## Comparison with TypeScript

If you're familiar with TypeScript, here's how the concepts map:

### TypeScript:

```typescript
interface PendulumParams {
  alpha: number;
  T: number;
  K: number;
}

class PendulumODE extends ODESystem<PendulumParams> {
  constructor(params: PendulumParams) {
    super(params);
    // this.params is typed as PendulumParams
  }
}
```

### Python (pybasin):

```python
class PendulumParams(TypedDict):
    alpha: float
    T: float
    K: float

class PendulumODE(ODESystem[PendulumParams]):
    def __init__(self, params: PendulumParams):
        super().__init__(params)
        # self.params is typed as PendulumParams
```

The main difference is that Python uses `TypedDict` instead of `interface`, and square brackets `[]` for generics instead of angle brackets `<>`.

## Type Checking

To verify your types are correct, run:

```bash
# With pyright (recommended for VS Code)
uv run pyright

# Or with mypy
uv run mypy src/
```

## Common Patterns

### Optional Parameters

```python
from typing import TypedDict, NotRequired

class OptionalParams(TypedDict):
    alpha: float                    # Required
    beta: NotRequired[float]        # Optional (Python 3.11+)
```

### Multiple Parameter Types

If you need to support multiple parameter configurations:

```python
from typing import Union

ParamVariant1 = TypedDict("ParamVariant1", {"a": float, "b": float})
ParamVariant2 = TypedDict("ParamVariant2", {"x": float, "y": float})

class FlexibleODE(ODESystem[Union[ParamVariant1, ParamVariant2]]):
    def ode(self, t, y):
        if "a" in self.params:
            # Handle variant 1
            pass
        else:
            # Handle variant 2
            pass
```

## Best Practices

1. **Always use TypedDict for parameters**: Don't use plain `dict[str, float]`
2. **Document your parameter fields**: Add docstrings or comments to each field
3. **Be specific with types**: Use `float`, `int`, `str` instead of `Any`
4. **Run type checkers regularly**: Integrate `pyright` or `mypy` into your workflow
5. **Keep parameters immutable**: Don't modify `self.params` after initialization

## Troubleshooting

### "Type is not assignable to TypeVar"

If you see this error, make sure:

- Your parameter type is a `TypedDict` or plain `dict`
- You're consistently using the same generic type throughout

### IDE not showing autocomplete

- Restart your Python language server
- Ensure your `TypedDict` is properly defined
- Check that you're using the latest version of your type checker

## Further Reading

- [Python TypedDict documentation](https://docs.python.org/3/library/typing.html#typing.TypedDict)
- [PEP 589 – TypedDict](https://peps.python.org/pep-0589/)
- [Pyright documentation](https://microsoft.github.io/pyright/)
