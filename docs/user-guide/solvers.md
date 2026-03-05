# Solvers

Solvers numerically integrate ODE systems from batches of initial conditions. Every solver in pybasin conforms to `SolverProtocol`, accepts PyTorch tensors as input, returns PyTorch tensors as output, and supports persistent disk caching -- regardless of the underlying numerical backend.

!!! info "Unified Tensor Interface"
All solvers accept `torch.Tensor` inputs and return `torch.Tensor` outputs. Internal conversions to JAX arrays, NumPy arrays, or other formats happen transparently. You do not need to handle backend-specific tensor types.

## Available Solvers

| Class                 | Backend       | CPU | GPU (CUDA) | Event Functions | Recommended For                                                      |
| --------------------- | ------------- | --- | ---------- | --------------- | -------------------------------------------------------------------- |
| `TorchDiffEqSolver`   | torchdiffeq   | Yes | Yes        | No              | **Default solver** -- works out of the box                           |
| `JaxSolver`           | JAX/Diffrax   | Yes | Yes        | Yes             | **Fastest on GPU** (requires `pybasin[jax]`)                         |
| `TorchOdeSolver`      | torchode      | Yes | Yes        | No              | Independent per-trajectory step sizes (requires `pybasin[torchode]`) |
| `ScipyParallelSolver` | scipy/sklearn | Yes | No         | No              | Debugging, reference baselines                                       |

### ODE System Pairing

Each solver backend expects a specific ODE system base class:

| Solver                | ODE System Class | Why                                                                  |
| --------------------- | ---------------- | -------------------------------------------------------------------- |
| `JaxSolver`           | `JaxODESystem`   | Uses pure JAX operations for JIT compilation and `jax.vmap` batching |
| `TorchDiffEqSolver`   | `ODESystem`      | Wraps a `torch.nn.Module` with standard PyTorch tensor operations    |
| `TorchOdeSolver`      | `ODESystem`      | Same PyTorch interface as `TorchDiffEqSolver`                        |
| `ScipyParallelSolver` | `ODESystem`      | Converts PyTorch ODE to NumPy internally                             |

`JaxODESystem` subclasses define `ode(t, y)` using `jax.numpy` operations, which enables JIT compilation and vectorized GPU execution. `ODESystem` subclasses define `ode(t, y)` using `torch` operations and inherit from `torch.nn.Module`, so their parameters can be moved between devices with `.to(device)`.

When `solver=None` is passed to `BasinStabilityEstimator`, the solver is chosen automatically based on the ODE class:

- `JaxODESystem` &rarr; `JaxSolver` (only if `pybasin[jax]` is installed)
- `ODESystem` &rarr; `TorchDiffEqSolver`

If the ODE inherits from `ODESystem`, `TorchDiffEqSolver` is always selected -- even if JAX is installed. To use `JaxSolver`, the ODE must inherit from `JaxODESystem`.

## Common Parameters

All solvers share these constructor parameters:

| Parameter   | Type                  | Default            | Description                                   |
| ----------- | --------------------- | ------------------ | --------------------------------------------- |
| `time_span` | `tuple[float, float]` | `(0, 1000)`        | Integration interval `(t_start, t_end)`       |
| `n_steps`   | `int`                 | `1000`             | Number of evaluation points                   |
| `device`    | `str` or `None`       | `None`             | `"cuda"`, `"cpu"`, or `None` for auto-detect  |
| `rtol`      | `float`               | `1e-8`             | Relative tolerance for adaptive stepping      |
| `atol`      | `float`               | `1e-6`             | Absolute tolerance for adaptive stepping      |
| `cache_dir` | `str` or `None`       | `".pybasin_cache"` | Cache directory path. `None` disables caching |

## Generic Solver API

All solvers expose three capabilities through `SolverProtocol`:

### `integrate(ode_system, y0)`

Solves the ODE system for a batch of initial conditions. The `y0` tensor must be 2D with shape `(batch, n_dims)`, where `batch` is the number of initial conditions and `n_dims` is the number of state variables.
A `ValueError` is raised if `y0` is not 2D -- the error message suggests using `y0.unsqueeze(0)` for single trajectories.

Returns a tuple `(t_eval, y_values)`:

- `t_eval` has shape `(n_steps,)` -- the time points at which solutions are evaluated
- `y_values` has shape `(n_steps, batch, n_dims)` -- the state at each time point for every initial condition

```python
t_eval, y_values = solver.integrate(ode_system, y0)
# t_eval.shape   -> (n_steps,)
# y_values.shape -> (n_steps, batch, n_dims)
```

### `clone(*, device, n_steps_factor, cache_dir)`

Creates a copy of the solver with optionally overridden settings. Useful for creating a high-resolution variant for plotting while keeping the original solver for computation:

```python
plot_solver = solver.clone(n_steps_factor=10, device="cpu")
```

| Parameter        | Type                      | Default | Description                                                                      |
| ---------------- | ------------------------- | ------- | -------------------------------------------------------------------------------- |
| `device`         | `str` or `None`           | `None`  | Override the device. `None` keeps current device                                 |
| `n_steps_factor` | `int`                     | `1`     | Multiply `n_steps` by this factor                                                |
| `cache_dir`      | `str`, `None`, or `UNSET` | `UNSET` | Override cache directory. `None` disables caching. `UNSET` keeps current setting |

### `device` attribute

A `torch.device` indicating where output tensors are placed. Always reflects the normalized device (e.g. `torch.device("cuda:0")`, never bare `"cuda"`).

## Behavior Notes

Several behaviors apply to all solvers and happen automatically. Understanding them prevents unexpected surprises.

### Device Auto-Detection

When `device=None`, the solver checks whether CUDA is available via `torch.cuda.is_available()` (or the JAX equivalent for `JaxSolver`). If a GPU is found, `"cuda:0"` is selected; otherwise the solver falls back to `"cpu"`. The string `"cuda"` is always normalized to `"cuda:0"` internally.

### Dtype and Precision

Solvers do not enforce a specific dtype. Time evaluation points are created with the same dtype as `y0`, so `float32`, `float64`, and other dtypes are accepted. No automatic casting is performed -- the solver logs a warning only when `y0` is on a different device than the solver, not for dtype differences.

!!! tip "float32 for GPU workloads"
GPU solvers (`JaxSolver`, `TorchDiffEqSolver`, `TorchOdeSolver`) perform best with `float32` tensors. CUDA devices process `float32` significantly faster than `float64`. When using `float32`, tighten tolerances accordingly -- the default `rtol=1e-8` targets `float64` precision and will cause the adaptive stepper to stagnate with single-precision arithmetic. Values such as `rtol=1e-5`, `atol=1e-6` are more appropriate for `float32`. Use `float64` when higher accuracy is required.

### ODE System Device Transfer

PyTorch-based solvers (`TorchDiffEqSolver`, `TorchOdeSolver`, `ScipyParallelSolver`) call `ode_system.to(self.device)` before integration. This moves the ODE system's parameters to the solver's device automatically. `JaxSolver` does not perform this transfer because JAX ODE systems are stateless and device placement is handled through `jax.device_put`.

## Caching

Every solver caches integration results to disk so that repeated runs with identical inputs skip the numerical integration entirely. Caching is controlled through the `cache_dir` constructor parameter, which defaults to `".pybasin_cache"`. Pass `None` to disable it.

```python
# Default -- cache under .pybasin_cache/ at the project root
solver = JaxSolver(time_span=(0, 1000), n_steps=5000)

# Explicit subfolder for a specific system
solver = JaxSolver(time_span=(0, 1000), n_steps=5000, cache_dir=".pybasin_cache/pendulum")

# No caching
solver = JaxSolver(time_span=(0, 1000), n_steps=5000, cache_dir=None)
```

### Path Resolution

Relative paths (like `".pybasin_cache"` or `".pybasin_cache/pendulum"`) are resolved from the project root, which is located by walking up the directory tree until a `pyproject.toml` or `.git` marker is found. Absolute paths are used as-is. The directory is created automatically if it does not exist.

### Cache Keys

The cache key is an MD5 hash built from six components: the solver class name, the ODE system's source representation (via `get_str()`), the ODE system parameters, the serialized `y0` and `t_eval` tensors, and solver-specific configuration (tolerances, method, etc.). Changing any of these produces a different key, so stale results are never returned.

### Storage Format

Cached tensors are stored using [safetensors](https://huggingface.co/docs/safetensors/), which provides fast, zero-copy loading without the security concerns of pickle. On cache load, tensors are moved to the solver's current device. Corrupted files are detected and deleted automatically rather than raising exceptions.

---

## JaxSolver

The recommended solver for most workloads. It uses [Diffrax](https://docs.kidger.site/diffrax/) (Kidger, 2021) for numerical integration, with `jax.vmap` for batch processing and JIT compilation for performance. On GPU, `JaxSolver` achieves near-constant integration time regardless of sample count -- roughly 11.5 seconds for N ranging from 5,000 to 100,000 in [benchmark tests](../benchmarks/solvers.md). It is also the only solver that supports per-trajectory event-based early termination, which is critical for systems with unbounded trajectories.

!!! tip "Default Solver"
`TorchDiffEqSolver` is the default solver and ships with the core `pybasin` install.
When JAX and Diffrax are available (`pip install pybasin[jax]`) **and** the ODE system
inherits from `JaxODESystem`, `BasinStabilityEstimator` automatically selects `JaxSolver`.
If the ODE inherits from `ODESystem`, `TorchDiffEqSolver` is used regardless of whether
JAX is installed. `JaxSolver` delivers the best GPU performance and is the only solver
supporting event functions for early trajectory termination. For CPU-only workloads at
large sample sizes (N >= 100k), `TorchDiffEqSolver` is faster. See the
[Solver Comparison benchmark](../benchmarks/solvers.md) for detailed numbers.

`JaxSolver` does not inherit from the `Solver` base class. It implements `SolverProtocol` independently with its own device handling and caching logic. Two construction modes are available: a generic API for standard ODE integration, and a `solver_args` mode that passes arguments directly to `diffrax.diffeqsolve()`.

### Generic API

```python
from pybasin.solvers import JaxSolver
from diffrax import Dopri5

solver = JaxSolver(
    time_span=(0, 1000),
    n_steps=5000,
    device="cuda",
    method=Dopri5(),       # Diffrax solver instance
    rtol=1e-8,
    atol=1e-6,
    max_steps=16**5,       # Maximum integrator steps
    event_fn=None,         # Optional early termination
)
```

#### Constructor Parameters

| Parameter   | Type                     | Default            | Description                                               |
| ----------- | ------------------------ | ------------------ | --------------------------------------------------------- |
| `time_span` | `tuple[float, float]`    | `(0, 1000)`        | Integration interval `(t_start, t_end)`                   |
| `n_steps`   | `int`                    | `1000`             | Number of evaluation points                               |
| `device`    | `str` or `None`          | `None`             | `"cuda"`, `"gpu"`, `"cpu"`, or `None` for auto-detect     |
| `method`    | Diffrax solver or `None` | `None`             | Diffrax solver instance. Defaults to `Dopri5()` if `None` |
| `rtol`      | `float`                  | `1e-8`             | Relative tolerance for `PIDController`                    |
| `atol`      | `float`                  | `1e-6`             | Absolute tolerance for `PIDController`                    |
| `max_steps` | `int`                    | `16**5`            | Maximum number of integrator steps (1,048,576)            |
| `event_fn`  | `Callable` or `None`     | `None`             | Event function for per-trajectory early termination       |
| `cache_dir` | `str` or `None`          | `".pybasin_cache"` | Cache directory path. `None` disables caching             |

!!! note "Device String"
Unlike PyTorch-based solvers, `JaxSolver` also accepts `"gpu"` as a device string (mapped to JAX's GPU backend). Both `"cuda"` and `"gpu"` resolve to the same GPU device.

#### Tensor Conversion

`JaxSolver` converts between PyTorch and JAX tensors at the integration boundary. On GPU, this conversion uses DLPack for zero-copy transfer -- no data is duplicated in device memory. On CPU, the conversion falls back to a NumPy intermediate. Input tensors are PyTorch; output tensors are PyTorch. You never interact with JAX arrays directly.

### solver_args Mode

For advanced use cases (SDEs, CDEs, custom step-size controllers, or any configuration not exposed by the generic API), you can pass a dictionary of keyword arguments directly to [`diffrax.diffeqsolve()`](https://docs.kidger.site/diffrax/api/diffeqsolve/):

```python
from diffrax import Dopri5, ODETerm, PIDController, SaveAt
import jax.numpy as jnp

solver = JaxSolver(
    solver_args={
        "terms": ODETerm(lambda t, y, args: -y),
        "solver": Dopri5(),
        "t0": 0,
        "t1": 10,
        "dt0": 0.1,
        "saveat": SaveAt(ts=jnp.linspace(0, 10, 100)),
        "stepsize_controller": PIDController(rtol=1e-5, atol=1e-5),
    },
)
```

When `solver_args` is provided, all other Diffrax-specific parameters (`time_span`, `n_steps`, `solver`, `rtol`, `atol`, `max_steps`, `event_fn`) are ignored entirely. The solver wraps each call with `jax.vmap` and injects `y0` per trajectory -- do **not** include `y0` in the dictionary.

!!! warning "Baked-in Time Points"
In solver_args mode, the integration time points are determined by the `saveat` entry you provide. Calling `clone(n_steps_factor=10)` will **not** increase the time resolution -- the original `saveat.ts` is used as-is. A warning is logged if `n_steps_factor > 1` in this mode.

Because `solver_args` bypasses all automatic setup, no `ODETerm` wrapping, `PIDController` creation, or `SaveAt` construction is performed. You are responsible for providing a complete and valid set of Diffrax arguments.

### Event Functions

Event functions enable per-trajectory early termination, which is essential for systems where some trajectories diverge to infinity (e.g. the Lorenz system's "broken butterfly" regime). Each trajectory stops independently when the event triggers, while bounded trajectories continue integrating normally.

The event function signature is `(t, y, args) -> scalar Array`. Return a positive value to continue integration, or zero/negative to stop:

```python
import jax.numpy as jnp

def lorenz_stop_event(t, y, args, **kwargs):
    """Stop integration when any state variable exceeds 200 in absolute value."""
    max_val = 200.0
    return max_val - jnp.max(jnp.abs(y))
```

```python
solver = JaxSolver(
    time_span=(0, 1000),
    n_steps=4000,
    device="cuda",
    event_fn=lorenz_stop_event,
)
```

Internally, the event function is wrapped in a `diffrax.Event(cond_fn=event_fn)` and passed to `diffeqsolve`. For more details on handling diverging trajectories, see the [Handling Unbounded Trajectories](../guides/unbounded-trajectories.md) guide.

!!! warning "Post-event state values are `inf`"
When an event triggers early termination, Diffrax fills the remaining saved time points (those after the event) with `inf`. For example, if a trajectory diverges at $t = 50$ but `saveat` requests points up to $t = 1000$, all state values for $t > 50$ will be `inf`. Your feature extraction or classification code must handle this -- checking for `jnp.isinf` in the final state is a reliable way to detect terminated trajectories.

!!! note "Event functions in solver_args mode"
When using `solver_args`, include the event directly in the dictionary (e.g. as a `diffrax.Event` instance) rather than using the `event_fn` parameter.

**Reference:**

> Kidger, P. (2021). _On Neural Differential Equations_. PhD thesis, University of Oxford. [https://docs.kidger.site/diffrax/](https://docs.kidger.site/diffrax/)

---

## TorchDiffEqSolver

A PyTorch-native solver built on [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (Chen, 2018). It supports both adaptive-step and fixed-step methods, runs on CPU and CUDA, and integrates directly with `ODESystem` subclasses (which inherit from `torch.nn.Module`). At large sample sizes on CPU (N = 100,000), `TorchDiffEqSolver` is roughly 2x faster than `JaxSolver` on CPU -- though `JaxSolver` on GPU remains substantially faster overall.

```python
from pybasin.solvers.torchdiffeq_solver import TorchDiffEqSolver

solver = TorchDiffEqSolver(
    time_span=(0, 1000),
    n_steps=5000,
    device="cuda",
    method="dopri5",
    rtol=1e-8,
    atol=1e-6,
)
```

### Constructor Parameters

| Parameter   | Type                  | Default            | Description                                   |
| ----------- | --------------------- | ------------------ | --------------------------------------------- |
| `time_span` | `tuple[float, float]` | `(0, 1000)`        | Integration interval `(t_start, t_end)`       |
| `n_steps`   | `int`                 | `1000`             | Number of evaluation points                   |
| `device`    | `str` or `None`       | `None`             | `"cuda"`, `"cpu"`, or `None` for auto-detect  |
| `method`    | `str`                 | `"dopri5"`         | Integration method (see table below)          |
| `rtol`      | `float`               | `1e-8`             | Relative tolerance for adaptive stepping      |
| `atol`      | `float`               | `1e-6`             | Absolute tolerance for adaptive stepping      |
| `cache_dir` | `str` or `None`       | `".pybasin_cache"` | Cache directory path. `None` disables caching |

### Available Methods

| Method   | Type          | Description                   |
| -------- | ------------- | ----------------------------- |
| `dopri5` | Adaptive-step | Dormand-Prince 5(4) (default) |
| `dopri8` | Adaptive-step | Dormand-Prince 8(5,3)         |
| `bosh3`  | Adaptive-step | Bogacki-Shampine 3(2)         |
| `euler`  | Fixed-step    | Forward Euler                 |
| `rk4`    | Fixed-step    | Classic Runge-Kutta 4         |

Integration runs under `torch.no_grad()`, so no gradient graph is constructed during forward integration. The solver calls `ode_system.to(self.device)` before integrating, which moves the ODE system's `nn.Module` parameters to the solver's device.

**Reference:**

> Chen, R. T. Q. (2018). _torchdiffeq_. [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

---

## TorchOdeSolver

A parallel ODE solver built on [torchode](https://torchode.readthedocs.io/en/latest/) (Lienen & Gunnemann, 2022). Its distinguishing feature is independent step-size control per batch element: each trajectory can advance with its own time step, avoiding the performance penalty that arises when a single stiff trajectory forces small steps for the entire batch.

```python
from pybasin.solvers.torchode_solver import TorchOdeSolver

solver = TorchOdeSolver(
    time_span=(0, 1000),
    n_steps=5000,
    device="cuda",
    method="dopri5",
    rtol=1e-8,
    atol=1e-6,
)
```

### Constructor Parameters

| Parameter   | Type                  | Default            | Description                                   |
| ----------- | --------------------- | ------------------ | --------------------------------------------- |
| `time_span` | `tuple[float, float]` | `(0, 1000)`        | Integration interval `(t_start, t_end)`       |
| `n_steps`   | `int`                 | `1000`             | Number of evaluation points                   |
| `device`    | `str` or `None`       | `None`             | `"cuda"`, `"cpu"`, or `None` for auto-detect  |
| `method`    | `str`                 | `"dopri5"`         | Integration method (see table below)          |
| `rtol`      | `float`               | `1e-8`             | Relative tolerance for adaptive stepping      |
| `atol`      | `float`               | `1e-6`             | Absolute tolerance for adaptive stepping      |
| `cache_dir` | `str` or `None`       | `".pybasin_cache"` | Cache directory path. `None` disables caching |

### Available Methods

| Method   | Type          | Description                   |
| -------- | ------------- | ----------------------------- |
| `dopri5` | Adaptive-step | Dormand-Prince 5(4) (default) |
| `tsit5`  | Adaptive-step | Tsitouras 5(4)                |
| `euler`  | Fixed-step    | Forward Euler                 |
| `heun`   | Fixed-step    | Heun's method                 |

The method string is lowercased internally, so `"Dopri5"` and `"dopri5"` are equivalent. Integration runs under `torch.inference_mode()`. Internally, torchode uses `IntegralController` for adaptive step-size selection and `AutoDiffAdjoint` as the solver wrapper.

!!! warning "Performance at Large N"
Benchmark results show that `TorchOdeSolver` scales poorly at large sample sizes. At N = 100,000 it took roughly 310 seconds on CUDA -- compared to about 11 seconds for `JaxSolver`. Consider `TorchOdeSolver` primarily when per-trajectory step-size independence is important for correctness, not for raw throughput. See the [Solver Comparison benchmark](../benchmarks/solvers.md) for details.

**Reference:**

> Lienen, M., & Gunnemann, S. (2022). torchode: A Parallel ODE Solver for PyTorch. _The Symbiosis of Deep Learning and Differential Equations II_, NeurIPS. [https://openreview.net/forum?id=uiKVKTiUYB0](https://openreview.net/forum?id=uiKVKTiUYB0)

---

## ScipyParallelSolver

A CPU-only solver that delegates integration to [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) and parallelizes across initial conditions using `sklearn.utils.parallel.Parallel` with the loky backend. Each trajectory is solved independently in a separate process. This solver is primarily useful for debugging, for validating results against a well-established reference implementation, and for accessing scipy's implicit solvers (`Radau`, `BDF`) which handle stiff systems.

```python
from pybasin.solvers.scipy_solver import ScipyParallelSolver

solver = ScipyParallelSolver(
    time_span=(0, 1000),
    n_steps=5000,
    n_jobs=-1,           # Use all CPU cores
    method="RK45",
    rtol=1e-8,
    atol=1e-6,
    max_step=None,       # Defaults to (t_end - t_start) / 100
)
```

### Constructor Parameters

| Parameter   | Type                  | Default            | Description                                              |
| ----------- | --------------------- | ------------------ | -------------------------------------------------------- |
| `time_span` | `tuple[float, float]` | `(0, 1000)`        | Integration interval `(t_start, t_end)`                  |
| `n_steps`   | `int`                 | `1000`             | Number of evaluation points                              |
| `device`    | `str` or `None`       | `None`             | Only `"cpu"` is supported (see note below)               |
| `n_jobs`    | `int`                 | `-1`               | Number of parallel workers (`-1` for all CPU cores)      |
| `method`    | `str`                 | `"RK45"`           | `scipy.integrate.solve_ivp` method                       |
| `rtol`      | `float`               | `1e-6`             | Relative tolerance                                       |
| `atol`      | `float`               | `1e-8`             | Absolute tolerance                                       |
| `max_step`  | `float` or `None`     | `None`             | Maximum step size. Defaults to `(t_end - t_start) / 100` |
| `cache_dir` | `str` or `None`       | `".pybasin_cache"` | Cache directory path. `None` disables caching            |

!!! warning "CPU Only"
If you pass `device="cuda"`, the solver logs a warning and silently falls back to CPU. No error is raised. This applies to both the constructor and `clone()`.

### Available Methods

Scipy provides explicit methods (`RK45`, `RK23`, `DOP853`) and implicit methods for stiff problems (`Radau`, `BDF`, `LSODA`). See the [scipy.integrate.solve_ivp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for the full list.

### Parallelization Behavior

When `batch_size > 1` and `n_jobs != 1`, trajectories are distributed across worker processes using the loky backend. For a single trajectory (`batch_size == 1`) or when `n_jobs == 1`, execution is sequential with no multiprocessing overhead. Each worker converts tensors from PyTorch to NumPy, calls `solve_ivp`, and converts results back -- so this solver carries per-trajectory conversion overhead that the GPU-based solvers avoid.

---

## See Also

- [Solver Comparison benchmark](../benchmarks/solvers.md) -- detailed timing data across backends and sample sizes
- [Handling Unbounded Trajectories](../guides/unbounded-trajectories.md) -- event functions and zero-masking strategies for diverging systems
