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

## Task 2: ODE redesign (`ode(t, y, p)`) + solver `params` argument ✅ DONE

### Summary

All sub-tasks completed in commit `24b793b` (`feat: solver support for parallel parameter batching`).

**2a–2c** (ODE base classes, concrete ODE systems, `ODESystemProtocol`) were already done in prior commits.

**2d. Solver changes** -- implemented across all four backends:

- `SolverProtocol.integrate` and `Solver.integrate` / `Solver._integrate` gain `params: torch.Tensor | None = None`.
- `Solver.integrate` (base) owns the expand logic: when `params` is `(P, n_params)`, it expands `y0` to `(B*P, n_dims)` via `repeat_interleave` and tiles `params` to `(B*P, n_params)` before forwarding to `_integrate`. The output shape is `(t_steps, B*P, n_dims)` in IC-major order (trajectory `ic*P + p` carries `(y0[ic], params[p])`).
- **JaxSolver**: `_integrate_jax_generic` does its own equivalent expand (using `jnp.repeat` / `jnp.tile`) and vmaps `solve_single(y0_i, p_i)` over both arrays together. The `ode_wrapper` already passes `args` straight to `ode(t, y, args)` so no adapter was needed.
- **TorchDiffEqSolver**: base-class expansion is done before `_integrate` is called; `_integrate` stores the pre-expanded tensor as `ode_system._batched_params` so `forward(t, y)` passes it to `ode(t, y, p)`.
- **TorchOdeSolver** and **ScipySolver**: `_integrate` signatures updated; cache keys include `params`.
- `CacheManager` updated to include `params` in the cache key.
- `ODESystemProtocol` gains `params_to_array` stub.
- All 9 concrete ODE classes (Torch + JAX variants for pendulum, friction, Lorenz, Duffing, Rössler network) already use `p[..., i]` indexing and required no further changes.

**2e. TemplateIntegrator** -- already calls `integrate` with `params=None` (default), so no change was needed.

Tests extended in `tests/unit/solvers/test_solver.py` and `tests/unit/solvers/test_jax_solver.py` to cover the batched-params path.

---

## Task 3: BSE pipeline extension + BSS simplification

With `ode(t, y, p)` and `solver.integrate(..., params=)` in place,
BSE gains a batched pipeline and BSS delegates pure-param studies to it.

The key insight from benchmarking: a single flat vmap over `P*B` trajectories
matches the performance of a fixed-param vmap with the same total count.
BSE owns the flatten/reshape logic -- the solver just sees a flat batch.

### 3a. BSE: `_run_basin_stability` + `run_parameter_study`

[src/pybasin/basin_stability_estimator.py](src/pybasin/basin_stability_estimator.py)

Rename `estimate_bs` to `run()` and extract the body into `_run_basin_stability(study_params=None)`.
The pipeline:

1. Build `(P, n_params)` param grid when `study_params` is given; otherwise skip
   (pass `params=None` to the solver and it uses `params_to_array()` as-is).

   Column order is determined by the TypedDict field declaration order of the ODE
   system's params (i.e. `list(ode_system.params.keys())`). For each `RunConfig`:
   - Start from `ode_system.params_to_array()` as the base row.
   - Override the indices for the varied parameters using `rc.study_label`
     (short names) mapped to positions in `param_names`:
     ```python
     param_names = list(ode_system.params.keys())
     for rc in study_params:
         row = ode_system.params_to_array().clone()
         for key, val in rc.study_label.items():
             row[param_names.index(key)] = val
         rows.append(row)
     params_grid = torch.stack(rows)  # (P, n_params)
     ```

2. Sample B ICs (once, shared across all P).
3. Call `solver.integrate(ode_system, y0, params=params_grid)`.
   The solver handles the `(B, P)` → `(B*P,)` expansion internally.
   Output shape: `(t_steps, B*P, n_dims)` in IC-major order.
4. Reshape output `(t_steps, B*P, n_dims)` → slice per-param group of size B.
5. Per group: unbounded detection, orbit data, feature extraction,
   feature filtering, classification, BS computation → `StudyResult`.

Public methods:

```python
def run(self) -> StudyResult:
    return self._run_basin_stability()[0]

def run_parameter_study(self, study_params: StudyParams) -> list[StudyResult]:
    return self._run_basin_stability(study_params)
```

`run()` keeps the same signature and return type as the old `estimate_bs`.

### 3b. BSS: detect pure-param study and delegate

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
