# Parameter Batching -- Implementation Tasks

Three independent tracks bring parameter batching to pyBasin. Task 1 can land
on its own. Task 2 is the ODE/solver foundation. Task 3 builds on Task 2 to
wire batching into BSE and BSS.

Dependencies: **Task 1 -- none** | **Task 2 -- none** | **Task 3 -- requires Task 2**

---

## Task 1: Unify SweepStudyParams constructor with Grid/Zip ✅ DONE

`GridStudyParams` and `ZipStudyParams` accept `**kwargs` where each key is a
parameter path and each value is a list. `SweepStudyParams` is the odd one out --
it uses positional `name` and `values` arguments. Aligning it removes the
inconsistency.

### Current API

```python
SweepStudyParams(name='ode_system.params["T"]', values=[0.1, 0.2, 0.3])
```

### Target API

```python
SweepStudyParams(**{'ode_system.params["T"]': [0.1, 0.2, 0.3]})
```

One keyword argument, one list of values. The constructor rejects zero or
multiple kwargs with a `ValueError`.

### Implementation

Replace the `__init__` in `SweepStudyParams`:

```python
def __init__(self, **params: list[Any]) -> None:
    if len(params) != 1:
        raise ValueError(
            f"SweepStudyParams takes exactly one parameter, got {len(params)}"
        )
    self.name: str = next(iter(params))
    self.values: list[Any] = list(next(iter(params.values())))
```

`__iter__` and `__len__` stay unchanged -- they already read `self.name`
and `self.values`.

Update the docstring and class-level example to show the `**kwargs` form.

### Call sites to update

**Source:**

- [src/pybasin/study_params.py](src/pybasin/study_params.py) -- class definition + docstring

**Case studies (7 files):**

| File                                                                                                                         | Current `name=` value          |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [case_studies/pendulum/main_pendulum_case2.py](case_studies/pendulum/main_pendulum_case2.py)                                 | `'ode_system.params["T"]'`     |
| [case_studies/rossler_network/main_rossler_network_k_study.py](case_studies/rossler_network/main_rossler_network_k_study.py) | `'ode_system.params["K"]'`     |
| [case_studies/pendulum/main_pendulum_hyperparameters.py](case_studies/pendulum/main_pendulum_hyperparameters.py)             | `"n"`                          |
| [case_studies/lorenz/main_lorenz_sigma.py](case_studies/lorenz/main_lorenz_sigma.py)                                         | `'ode_system.params["sigma"]'` |
| [case_studies/lorenz/main_lorenz_hyperpN.py](case_studies/lorenz/main_lorenz_hyperpN.py)                                     | `"N"`                          |
| [case_studies/lorenz/main_lorenz_hyperpTol.py](case_studies/lorenz/main_lorenz_hyperpTol.py)                                 | `"solver.rtol"`                |
| [case_studies/friction/main_friction_v_study.py](case_studies/friction/main_friction_v_study.py)                             | `'ode_system.params["v_d"]'`   |

**Unit tests:**

| File                                                                                 | Lines                                              |
| ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| [tests/unit/test_study_params.py](tests/unit/test_study_params.py)                   | L36, L46, L53, L62                                 |
| [tests/unit/test_basin_stability_study.py](tests/unit/test_basin_stability_study.py) | L67, L88, L115, L151, L298, L372, L398, L418, L465 |

**Integration tests:**

| File                                                                                                                   | Lines |
| ---------------------------------------------------------------------------------------------------------------------- | ----- |
| [tests/integration/rossler_network/test_rossler_network.py](tests/integration/rossler_network/test_rossler_network.py) | L161  |
| [tests/integration/test_helpers.py](tests/integration/test_helpers.py)                                                 | L496  |

**Docs:**

| File                                                                                 | Lines      |
| ------------------------------------------------------------------------------------ | ---------- |
| [docs/user-guide/basin-stability-study.md](docs/user-guide/basin-stability-study.md) | L132, L380 |

### Validation

```bash
uv run pytest tests/unit/test_study_params.py tests/unit/test_basin_stability_study.py -x
```

---

## Task 2: ODE redesign (`ode(t, y, p)`) + solver `params` argument

Parameters move from `self.params` dict reads inside the ODE to an explicit
`p` array argument. The solver gains an optional `params` kwarg to pass
parameter arrays through to the ODE.

### 2a. ODE base classes

**JaxODESystem** ([src/pybasin/solvers/jax_ode_system.py](src/pybasin/solvers/jax_ode_system.py)):

```python
class JaxODESystem[P]:
    PARAM_KEYS: ClassVar[tuple[str, ...]]

    def __init__(self, params: P) -> None:
        self.params = params            # kept for backward compat / display
        self.default_params: P = params

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        raise NotImplementedError

    def params_to_array(self, params: P | None = None) -> Array:
        p = params if params is not None else self.default_params
        return jnp.array([p[k] for k in self.PARAM_KEYS])
```

- `self.params` stays as an alias for backward compatibility (`get_str`,
  caching, plotter re-creation all read it).
- `PARAM_KEYS` declares which dict entries map to the flat `p` array and
  in what order.
- `args` parameter in the current signature becomes `p`. Subclasses that
  previously ignored `args` now read from `p`.

**ODESystem (torch)** ([src/pybasin/solvers/torch_ode_system.py](src/pybasin/solvers/torch_ode_system.py)):

Same pattern. `ode(self, t, y)` becomes `ode(self, t, y, p)`.
`forward` wraps `ode` and passes `self.params_to_array()` as default `p`.

### 2b. Concrete ODE systems

Each ODE class needs three changes:

1. Add `PARAM_KEYS` class variable.
2. Change `ode` body: replace `self.params["key"]` with `p[..., i]` indexing.
3. For network ODEs (Rossler), store structural state (`_N`, `_edges_i`,
   `_edges_j`) on `self.__init__` and exclude them from `PARAM_KEYS`.

**Example -- DuffingJaxODE** ([case_studies/duffing_oscillator/duffing_jax_ode.py](case_studies/duffing_oscillator/duffing_jax_ode.py)):

```python
class DuffingJaxODE(JaxODESystem[DuffingParams]):
    PARAM_KEYS = ("delta", "k3", "A")

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        delta, k3, amplitude = p[..., 0], p[..., 1], p[..., 2]
        x, x_dot = y[..., 0], y[..., 1]
        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + amplitude * jnp.cos(t)
        return jnp.stack([dx_dt, dx_dot_dt], axis=-1)
```

The `[..., i]` indexing broadcasts over any leading batch dimensions
(scalar `(n_params,)`, per-IC `(B, n_params)`, or flattened `(P*B, n_params)`).

**Files to update (9 ODE classes):**

| File                                                                                                               | Class                  | PARAM_KEYS                              |
| ------------------------------------------------------------------------------------------------------------------ | ---------------------- | --------------------------------------- |
| [case_studies/duffing_oscillator/duffing_jax_ode.py](case_studies/duffing_oscillator/duffing_jax_ode.py)           | `DuffingJaxODE`        | `("delta", "k3", "A")`                  |
| [case_studies/duffing_oscillator/duffing_ode.py](case_studies/duffing_oscillator/duffing_ode.py)                   | `DuffingODE`           | `("delta", "k3", "A")`                  |
| [case_studies/friction/friction_jax_ode.py](case_studies/friction/friction_jax_ode.py)                             | `FrictionJaxODE`       | check params dict                       |
| [case_studies/friction/friction_ode.py](case_studies/friction/friction_ode.py)                                     | `FrictionODE`          | check params dict                       |
| [case_studies/lorenz/lorenz_jax_ode.py](case_studies/lorenz/lorenz_jax_ode.py)                                     | `LorenzJaxODE`         | check params dict                       |
| [case_studies/lorenz/lorenz_ode.py](case_studies/lorenz/lorenz_ode.py)                                             | `LorenzODE`            | check params dict                       |
| [case_studies/pendulum/pendulum_jax_ode.py](case_studies/pendulum/pendulum_jax_ode.py)                             | `PendulumJaxODE`       | check params dict                       |
| [case_studies/pendulum/pendulum_ode.py](case_studies/pendulum/pendulum_ode.py)                                     | `PendulumODE`          | check params dict                       |
| [case_studies/rossler_network/rossler_network_jax_ode.py](case_studies/rossler_network/rossler_network_jax_ode.py) | `RosslerNetworkJaxODE` | float params only; structural on `self` |

### 2c. ODESystemProtocol

[src/pybasin/protocols.py](src/pybasin/protocols.py) -- add `PARAM_KEYS` and
`params_to_array` to the protocol so BSE/solvers can call them generically.

### 2d. Solver changes

The `integrate` signature gains an optional `params`:

```python
def integrate(
    self,
    ode_system: ODESystemProtocol,
    y0: torch.Tensor,
    params: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**JaxSolver** ([src/pybasin/solvers/jax_solver.py](src/pybasin/solvers/jax_solver.py)):

- When `params is None`: call `ode_system.params_to_array()` to get
  `(n_params,)`. Pass as `args` to `diffeqsolve`. vmap over ICs only
  (current behavior, just explicit now).
- When `params` has shape `(P, n_params)`: vmap over both the param axis
  and IC axis. Flatten P\*B internally or nest vmaps.

The existing `ode_wrapper(t, y, args)` already forwards `args` to
`ode_system.ode(t, y, args)`. After the ODE redesign, `args` IS `p` --
no adapter needed.

**TorchDiffEqSolver** ([src/pybasin/solvers/torchdiffeq_solver.py](src/pybasin/solvers/torchdiffeq_solver.py)):

- When `params is None`: call `ode_system.params_to_array()`, capture in
  closure. `torchdiffeq.odeint` calls `f(t, y)` where the closure provides `p`.
- When `params` has shape `(P, n_params)`: flatten `y0` to `(P*B, S)`,
  repeat-interleave `params` to `(P*B, n_params)`, capture `p_flat` in closure.

**Solver base** ([src/pybasin/solvers/base.py](src/pybasin/solvers/base.py)):

- `integrate` and `_integrate` signatures gain `params`.
- Cache key includes the param array when present.

**SolverProtocol** ([src/pybasin/protocols.py](src/pybasin/protocols.py)):

- `integrate` signature gains `params: torch.Tensor | None = None`.

### 2e. TemplateIntegrator

[src/pybasin/template_integrator.py](src/pybasin/template_integrator.py) -- its
`integrate(solver, ode_system)` call must forward default params. Since templates
always use the ODE's own params, pass `params=None` (solver uses
`ode_system.params_to_array()` as fallback).

### Validation

```bash
uv run pytest tests/ -x
```

All existing tests should pass because `params=None` triggers the same
fallback as today. Run case studies to verify end-to-end.

---

## Task 3: BSE pipeline extension + BSS simplification

With `ode(t, y, p)` and `solver.integrate(..., params=)` in place,
BSE gains a batched pipeline and BSS delegates pure-param studies to it.

### 3a. BSE: `_run_basin_stability` + `run_parameter_study`

[src/pybasin/basin_stability_estimator.py](src/pybasin/basin_stability_estimator.py)

Extract the body of `estimate_bs` into `_run_basin_stability(study_params=None)`.
The pipeline:

1. Build `(P, n_params)` array from `StudyParams.to_param_grid(PARAM_KEYS)`
   when `study_params` is given; otherwise `params_to_array()[None, :]` for P=1.
2. Sample B ICs (once, shared across all P).
3. Call `solver.integrate(ode_system, y0, params=param_grid)`.
4. Reshape output into P groups of `(N, B, S)`.
5. Per group: Solution, unbounded detection, orbit data, feature extraction,
   feature filtering, classification, BS computation -> `StudyResult`.

Public methods:

```python
def estimate_bs(self) -> StudyResult:
    return self._run_basin_stability()[0]

def run_parameter_study(self, study_params: StudyParams) -> list[StudyResult]:
    return self._run_basin_stability(study_params)
```

`estimate_bs` keeps its current signature and return type.

### 3b. StudyParams: `to_param_grid`

[src/pybasin/study_params.py](src/pybasin/study_params.py)

Add to `StudyParams` base class:

```python
def to_param_grid(
    self, param_keys: tuple[str, ...]
) -> tuple[Tensor, list[dict[str, Any]]]:
    configs = self.to_list()
    param_grid = stack([
        tensor([rc.study_label[k] for k in param_keys])
        for rc in configs
    ])
    labels = [rc.study_label for rc in configs]
    return param_grid, labels
```

`study_label` already contains short names (produced by `_extract_short_name`).
`param_keys` matches `PARAM_KEYS` from the ODE system, so the column order
aligns with what `ode(t, y, p)` expects.

### 3c. BSS: detect pure-param study and delegate

[src/pybasin/basin_stability_study.py](src/pybasin/basin_stability_study.py)

Add `_is_pure_param_study()` that checks whether every `ParamAssignment.name`
in every `RunConfig` starts with `'ode_system.params['`. If so, all runs vary
only ODE params and can be batched.

```python
def run(self) -> list[StudyResult]:
    if self._is_pure_param_study():
        bse = BasinStabilityEstimator(...)
        self.results = bse.run_parameter_study(self.study_params)
    else:
        # Serial fallback (existing loop, still uses exec for mixed studies)
        ...
    return self.results
```

The serial fallback keeps working as-is for mixed studies (varying sampler,
solver settings, etc.).

### Validation

```bash
uv run pytest tests/ -x
```

Verify batched path produces identical `StudyResult` contents as the serial
path by running a case study both ways and comparing basin stability values.
