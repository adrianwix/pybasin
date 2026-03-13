# Parameter Batching Architecture

## Problem Statement

`BasinStabilityStudy.run()` iterates serially over P parameter combinations\_. At each
step it creates a fresh `BasinStabilityEstimator`, integrates N initial conditions,
extracts features, classifies, and computes basin stability. The integration step
dominates wall-clock time, and the serial loop leaves hardware (GPU / multi-core CPU)
underutilized.

The experiment in `experiments/solver/experiment_parameter_batching.py` confirms that
both Diffrax (JAX vmap) and torchdiffeq (flattened P\*B batch) can integrate multiple
parameter combinations in a single call. With the `t_eval` save-window logic already
in place, memory consumption stays manageable -- the machine handles 100k trajectories
in one batch when only the steady-state window is stored.

The goal: batch the **integration** of P parameter sets x B initial conditions into
one solver call (or a few sub-batches), then split trajectories back per parameter
for feature extraction, classification, and BS computation.

A secondary goal is to **unify the return type**: ~~`BasinStabilityEstimator.estimate_bs()`
currently returns `dict[str, float]` while the Study wraps that into `StudyResult`.
Both paths should produce `StudyResult` directly.~~ ✅ Done (Track 1).

## Current Pipeline (per parameter combination)

```
Sampling  -->  Integration  -->  Solution  -->  Features  -->  Classification  -->  BS
   B ICs       (B, N, S)        object         (B, F)           (B,) labels       dict
```

Feature extraction and classification are inherently **per-parameter**: different
parameter values yield different attractor landscapes, so features and labels are
only meaningful within a single parameter set. Only the integration step can truly
benefit from cross-parameter batching.

## Architectural Options

> **Context.** Track 1 is complete: `BSE.estimate_bs()` returns `StudyResult`.
> The ODE redesign (Track 2) gives us `ode(t, y, p)` and
> `Solver.integrate(ode_system, y0, params=)`.
>
> **Key constraint.** `BasinStabilityStudy` can vary more than ODE parameters.
> Via `ParamAssignment`, it can swap the sampler (e.g., `CsvSampler` per
> parameter value), change solver settings, or assign entirely different
> initial conditions. Cross-parameter batching (P\*B in one solver call) is
> only valid when all P runs share the same solver, sampler, and ICs -- i.e.,
> **pure ODE-parameter studies**. When the study varies non-ODE things, a
> serial per-parameter loop is the only correct approach.

### Option A -- BSE owns the pipeline, BSS becomes a thin wrapper (recommended)

Extend `BasinStabilityEstimator` with a second public method,
`run_parameter_study()`, for pure ODE-parameter sweeps. Internally, both
`estimate_bs()` and `run_parameter_study()` delegate to a single private
method `_run_basin_stability()` that implements the full pipeline.

**Internal pipeline (`_run_basin_stability`):**

```python
def _run_basin_stability(
    self,
    study_params: StudyParams | None = None,
) -> list[StudyResult]:
    # 1. Build (P, n_params) array from StudyParams + PARAM_KEYS
    if study_params is None:
        param_grid = self.ode_system.params_to_array()[None, :]  # (1, n_params)
        study_labels = [{'baseline': True}]
    else:
        param_grid, study_labels = study_params.to_param_grid(
            self.ode_system.PARAM_KEYS
        )  # (P, n_params), list[dict]

    P = param_grid.shape[0]

    # 2. Sample B ICs
    y0 = self.sampler.sample(self.n)
    B = y0.shape[0]

    # 3. Integrate with params -- solver flattens P*B internally
    t, y = self.solver.integrate(self.ode_system, y0, params=param_grid)
    #   y shape: (N, P*B, S) when P > 1, (N, B, S) when P = 1

    # 4. Reshape into P groups of (N, B, S)
    groups = reshape_to_groups(y, P, B)

    # 5. Per group: solution -> features -> classify -> BS -> StudyResult
    results: list[StudyResult] = []
    for i, y_group in enumerate(groups):
        solution = Solution(y0, t, y_group)
        # ... unbounded detection, orbit data, feature extraction,
        #     feature filtering, classification, BS computation
        #     (reuses existing private methods)
        result = StudyResult(
            study_label=study_labels[i],
            ...
        )
        results.append(result)

    return results
```

**Public methods:**

```python
def estimate_bs(self) -> StudyResult:
    """Single-parameter estimation (P=1). Current public API, unchanged."""
    results = self._run_basin_stability()  # no study_params -> P=1
    return results[0]                      # unwrap single result

def run_parameter_study(
    self,
    study_params: StudyParams,
) -> list[StudyResult]:
    """Pure ODE-parameter sweep. All P sets share solver, sampler, ICs."""
    return self._run_basin_stability(study_params)
```

`estimate_bs()` is just `run_parameter_study` with P=1 and the result
unwrapped. No branching, no conditional return types. The same pipeline
code runs in both cases.

`run_parameter_study` accepts a `StudyParams` object (Sweep, Grid, Zip,
or Custom). The `StudyParams.to_param_grid(PARAM_KEYS)` method converts
the configs into a `(P, n_params)` tensor + `list[dict]` of study labels.
Because only ODE params are varied, keys are short names matching
`PARAM_KEYS` -- no `'ode_system.params["T"]'` paths.

**Simplified StudyParams API for pure param studies:**

The existing `StudyParams` classes use full exec-paths
(`'ode_system.params["T"]'`) because BSS can vary anything. For
`run_parameter_study()` only ODE params change, so keys are short names
that match `PARAM_KEYS`:

```python
# Current BSS API (full paths, can vary anything)
SweepStudyParams(name='ode_system.params["T"]', values=[0.01, 0.06, ...])
GridStudyParams(**{'ode_system.params["K"]': k_vals, 'ode_system.params["sigma"]': s_vals})

# New run_parameter_study API (short keys, ODE params only)
SweepStudyParams(T=[0.01, 0.06, ...])
GridStudyParams(K=k_vals, sigma=s_vals)
ZipStudyParams(T=t_vals, K=k_vals)
```

Internally the params use the same `StudyParams` base class and `RunConfig`
machinery. The key difference is that the names stored in `ParamAssignment`
are short param names (e.g., `"T"`) instead of exec-paths. The new
`to_param_grid(param_keys)` method on `StudyParams` iterates the
`RunConfig`s and builds the `(P, n_params)` array:

```python
def to_param_grid(
    self, param_keys: tuple[str, ...]
) -> tuple[Array, list[dict[str, Any]]]:
    """Convert configs to (P, n_params) array + study labels."""
    configs = self.to_list()
    param_grid = stack([
        array([rc.study_label[k] for k in param_keys])
        for rc in configs
    ])  # (P, n_params)
    labels = [rc.study_label for rc in configs]
    return param_grid, labels
```

For `SweepStudyParams`, the `name` constructor arg becomes the short key
directly. For `GridStudyParams` and `ZipStudyParams`, the `**kwargs` keys
are already short names -- no change to their constructor signature, just
the values passed in.

**BSS change:**

`BasinStabilityStudy` continues to use the existing full-path `StudyParams`
for mixed studies (varying sampler, solver, etc.) via the serial fallback.
For pure param studies, it can delegate to `bse.run_parameter_study()`.

```python
def run(self) -> list[StudyResult]:
    if self._is_pure_param_study():
        bse = BasinStabilityEstimator(...)
        self.results = bse.run_parameter_study(self.study_params)
    else:
        # Serial fallback for mixed studies
        self.results = []
        for run_config in self.study_params:
            bse = self._build_bse(run_config)
            result = bse.estimate_bs()
            result = {**result, "study_label": run_config.study_label}
            self.results.append(result)
    return self.results
```

**Pros:**

- All pipeline logic lives in BSE. No duplication, no extraction needed.
- `estimate_bs()` and `run_parameter_study()` share identical internals.
- Fixed return types: `StudyResult` for single, `list[StudyResult]` for study.
- Correctly handles the constraint that non-ODE variations (sampler, solver)
  cannot be batched -- BSS uses the serial fallback.
- BSE is independently useful for parameter sweeps without BSS.
- Template integration happens once and is reused across all P groups.
- No `exec()`-based mutation in the batched path.
- `StudyParams` reused as-is -- Sweep, Grid, Zip, Custom all work.
- Short param keys (`T=`, `K=`) for `run_parameter_study()` on BSE;
  BSS always uses full paths (`'ode_system.params["T"]'`) to avoid
  name collisions when varying non-ODE things.

**Cons:**

- BSS needs logic to detect pure-param vs mixed studies.
- Serial fallback still uses `exec()` (or its replacement) for mixed studies.
- `_run_basin_stability` has a loop over P groups for the post-integration
  pipeline -- this is inherently serial but fast (feature extraction +
  classification is cheap compared to integration).

## Design Principle: P=1 is not a special case

The distinction between "single parameter" and "batched parameters" should not
exist in the API. Every integration call carries a parameter array. When
`BasinStabilityEstimator` runs standalone, that array has length 1. When
`BasinStabilityStudy` runs, `StudyParams` produces the full array (via grid,
zip, or custom list). The solver sees the same interface either way.

This collapses the design into a single unified pipeline with no branching,
no `integrate_batched` vs `integrate`, and no conditional return types.
The one caveat is that cross-parameter batching only applies to **pure
ODE-parameter studies**; when BSS varies samplers or solver settings, it
falls back to a serial loop (see Option A above).

## Recommendation

Implement Option A. The implementation breaks into two tracks:

### Unified pipeline

```
params: (P, n_ode_params)       -- P=1 for standalone BSE, P>1 for study
y0:     (B, S)                  -- shared ICs, sampled once

Integration  -->  (P, N, B, S)  or equivalently P x (N, B, S) after split
     |
     v  (per parameter set p)
Solution_p  -->  Features_p  -->  Classification_p  -->  StudyResult_p
```

The solver flattens P\*B internally, integrates once, reshapes to P groups.
Everything downstream (Solution, features, classification, BS) runs per group.

- `estimate_bs()` → P=1, returns `StudyResult` (unwraps the single element).
- `run_parameter_study()` → P≥1, returns `list[StudyResult]`.

Both call the same `_run_basin_stability()` internally.

### Track 1 -- Unify return type ✅ DONE

`BSE.estimate_bs()` now returns `StudyResult` directly. BSE was also
refactored with private methods (`_integrate`, `_filter_features`,
`_classify`) that isolate each pipeline stage. `BSS.run()` collects
the `StudyResult` that BSE returns and stamps `study_label` onto it.

### Track 2 -- ODE system redesign and parameter-aware integration

The current ODE systems read parameters from `self.params` (a dict on the
instance). This makes parameter variation awkward: you cannot vmap over a dict
read in JAX, and torchdiffeq needs adapter classes to inject per-sample
parameters. Adding a separate `ode_with_params` method alongside the existing
`ode` would be a patch, not a fix.

The clean solution: like Julia's SciML, **parameters are always a function
argument**. The ODE signature becomes `ode(t, y, p)` where `p` is a flat
array. The instance holds default parameters (as a typed dict for
construction and human readability) and structural state (topology, constants
that never change). The solver always passes `p` explicitly.

#### New ODE base classes

**JAX:**

```python
class JaxODESystem[P]:
    PARAM_KEYS: ClassVar[tuple[str, ...]]   # ordered dict -> array mapping

    def __init__(self, params: P):
        self.default_params: P = params

    @abstractmethod
    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """RHS. p is shape (..., n_params) -- always passed by the solver."""
        ...

    def params_to_array(self, params: P | None = None) -> Array:
        p = params or self.default_params
        return jnp.array([p[k] for k in self.PARAM_KEYS])
```

**PyTorch:**

```python
class ODESystem[P](ABC, nn.Module):
    PARAM_KEYS: ClassVar[tuple[str, ...]]

    def __init__(self, params: P):
        super().__init__()
        self.default_params: P = params

    @abstractmethod
    def ode(self, t: Tensor, y: Tensor, p: Tensor) -> Tensor:
        ...

    def params_to_array(self, params: P | None = None) -> Tensor:
        p = params or self.default_params
        return torch.tensor([p[k] for k in self.PARAM_KEYS])
```

Key properties:

- `PARAM_KEYS` declares which dict keys map to the flat array and in what
  order. This is the contract between the dict world (construction, study
  params, human-readable labels) and the array world (solver, vmap, batching).
- `default_params` replaces `self.params`. The name makes it clear this is
  the fallback, not the only source of parameter values.
- `params_to_array()` converts the dict to a flat array. The solver calls
  this when `params=None` (standalone BSE, P=1).
- There is no separate `ode_with_params` method. There is only `ode(t, y, p)`.

#### Example: Duffing oscillator (JAX)

```python
class DuffingJaxODE(JaxODESystem[DuffingParams]):
    PARAM_KEYS = ("delta", "k3", "A")

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        delta, k3, A = p[..., 0], p[..., 1], p[..., 2]

        x = y[..., 0]
        x_dot = y[..., 1]

        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + A * jnp.cos(t)
        return jnp.stack([dx_dt, dx_dot_dt], axis=-1)
```

The body is nearly identical to today. The only change: `self.params["delta"]`
becomes `p[..., 0]`. The `[..., i]` indexing works at any batch dimension --
scalar `(n_params,)`, per-IC `(B, n_params)`, or flattened `(P*B, n_params)`.

#### Example: Rössler network (structural + study params)

Structural state (topology, node count) stays on `self`. Only the float
parameters that a study might sweep go into `p`:

```python
class RosslerNetworkJaxODE(JaxODESystem[RosslerNetworkParams]):
    PARAM_KEYS = ("a", "b", "c", "K")   # auto-derived from TypedDict

    def __init__(self, params: RosslerNetworkParams, n: int, edges_i: Array, edges_j: Array) -> None:
        super().__init__(params)
        # Structural topology -- constructor args, not in TypedDict
        self._N = n
        self._edges_i = edges_i
        self._edges_j = edges_j

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        a, b, c, k = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

        n = self._N
        x = y[:n]
        y_state = y[n:2*n]
        z = y[2*n:]

        diff = x[self._edges_i] - x[self._edges_j]
        coupling = jnp.zeros_like(x).at[self._edges_i].add(diff)

        dx_dt = -y_state - z - k * coupling
        dy_dt = x + a * y_state
        dz_dt = b + z * (x - c)
        return jnp.concatenate([dx_dt, dy_dt, dz_dt])
```

`PARAM_KEYS` explicitly excludes `edges_i`, `edges_j`, and `N` -- they are
structural topology passed as constructor arguments, not ODE parameters in
the TypedDict.

#### Solver changes

The `integrate` signature gains an optional `params` argument:

```python
def integrate(
    self,
    ode_system: ODESystemProtocol,
    y0: Tensor,                             # (B, S)
    params: Tensor | Array | None = None,   # (P, n_params) or None for P=1
) -> tuple[Tensor, Tensor]:
```

**JaxSolver** -- diffrax already expects `ode(t, y, args)`. The `args`
slot is exactly `p`:

```python
def integrate(self, ode_system, y0, params=None):
    if params is None:
        params = ode_system.params_to_array()   # (n_params,)

    term = ODETerm(ode_system.ode)   # ode(t, y, args) -- args IS p

    if params.ndim == 1:             # P=1
        sol = diffeqsolve(term, ..., y0=y0, args=params)
    else:                            # P>1: vmap over param axis
        solve_one = lambda p: diffeqsolve(term, ..., y0=y0, args=p)
        sol = jax.vmap(solve_one)(params)
```

No adapter, no wrapper. The ODE already has the right signature.

**TorchDiffEqSolver** -- torchdiffeq wants `f(t, y)`, so the solver wraps
to capture `p`:

```python
def integrate(self, ode_system, y0, params=None):
    if params is None:
        params = ode_system.params_to_array()   # (n_params,)

    if params.ndim == 1:             # P=1: broadcast p to all ICs
        def ode_fn(t, y):
            return ode_system.ode(t, y, params)
        return torchdiffeq.odeint(ode_fn, y0, t_span)
    else:                            # P>1: flatten P*B
        P, B = params.shape[0], y0.shape[0]
        y0_flat = y0.repeat(P, 1)                        # (P*B, S)
        p_flat = params.repeat_interleave(B, dim=0)       # (P*B, n_params)

        def ode_fn(t, y):
            return ode_system.ode(t, y, p_flat)
        return torchdiffeq.odeint(ode_fn, y0_flat, t_span)
```

The `[..., i]` indexing in the ODE handles both cases automatically:
`(n_params,)` broadcasts with `(B, S)`, and `(P*B, n_params)` matches
`(P*B, S)`.

#### StudyParams produces the parameter array

`StudyParams.to_param_grid(PARAM_KEYS)` iterates its `RunConfig`s and
stacks the values into a `(P, n_params)` array. The short key names in
`study_label` (e.g., `{"T": 0.5}`) are mapped to column indices by
matching against `PARAM_KEYS`.

For `run_parameter_study()`, the `StudyParams` is created with short
keys directly:

```python
study = SweepStudyParams(T=np.arange(0.01, 0.97, 0.05))
bse.run_parameter_study(study)

study = GridStudyParams(K=k_values, sigma=sigma_values)
bse.run_parameter_study(study)
```

When BSS delegates to `run_parameter_study()`, it passes its own
`StudyParams` (which may use full paths). The `to_param_grid()` method
extracts the short names via `_extract_short_name()` from the assignments.

#### BasinStabilityStudy.run() becomes

1. Check if pure param study (all assignments target ODE params).
2. If yes: create one BSE, call `bse.run_parameter_study(study_params)`.
   Sampling, integration (P\*B batch), and per-group pipeline all happen
   inside BSE.
3. If no (mixed study): serial loop with one `bse.estimate_bs()` per
   `RunConfig`, like today but without `exec()`.

Templates are integrated once (fixed params) and reused across all P groups.

#### Caching

When `params` is provided, the cache key includes the full parameter array.
At P=1 with `params=None` the key is the same as today (derived from
`ode_system.get_str()` which uses `default_params`).

#### Summary of changes vs current design

| Aspect                       | Current                             | New                                   |
| ---------------------------- | ----------------------------------- | ------------------------------------- |
| ODE signature                | `ode(t, y)` (+ `args` for JAX)      | `ode(t, y, p)` always                 |
| Param source in ODE body     | `self.params["delta"]` (dict read)  | `p[..., 0]` (array index)             |
| Param storage                | `self.params` (mutable dict)        | `self.default_params` (fallback)      |
| Dict-to-array mapping        | Not defined                         | `PARAM_KEYS` + `params_to_array()`    |
| JAX vmap over params         | Impossible (can't trace dict reads) | Natural (`args=p`, vmap over p)       |
| PyTorch batched params       | Needs `_BatchedODE` adapter         | Closure captures `p_flat`             |
| Separate method for batching | `ode_with_params` patch             | Gone -- there is only `ode(t, y, p)`  |
| Structural state             | Mixed into `self.params`            | On `self`, excluded from `PARAM_KEYS` |

## Plotter Compatibility Constraint

Both `MatplotlibStudyPlotter` and the interactive `InteractivePlotter` depend on
`BasinStabilityStudy`. Whatever the final batched design looks like, the plotters
must keep working without a rewrite. The contract they rely on is documented below.

### What MatplotlibStudyPlotter consumes

Pure data reader -- it never re-runs estimation. It accesses:

- `bs_study.results: list[StudyResult]` -- iterates, indexes, reads length.
- `bs_study.studied_parameter_names: list[str]` -- for axis labels and grouping.
- `bs_study.output_dir` -- for saving figures.
- Per result: `r["study_label"]`, `r["basin_stability"]`, `r["orbit_data"]`,
  `r["labels"]`. Type checks on study_label values to filter non-numeric params.

No other attribute matters for this plotter.

### What InteractivePlotter consumes

More demanding because it **re-creates BSE instances on demand** for interactive
drill-down into individual parameter values.

**Mode detection:**

- `isinstance(bse, BasinStabilityStudy)` -- hard type check to branch into
  parameter-study mode vs single-BSE mode.

**Study-level reads (same as matplotlib plotter):**

- `bs_study.results`, `bs_study.studied_parameter_names`

**For BSE re-creation** (`_compute_param_bse` and `StudyParameterManagerAIO`):

- `bs_study.study_params` -- iterated to get `RunConfig` objects.
- `bs_study.n`, `bs_study.ode_system`, `bs_study.sampler`, `bs_study.solver`,
  `bs_study.feature_extractor`, `bs_study.estimator`,
  `bs_study.template_integrator` -- all placed into a context dict.
- `run_config.assignments` -- each assignment is `exec()`-ed to mutate the
  context (e.g., `ode_system.params["T"] = 0.5`).
- A new `BasinStabilityEstimator` is constructed from the mutated context and
  `estimate_bs()` is called to get full Solution + features + labels.

**Other AIO components:**

- `ParamOverviewAIO` and `ParamOrbitDiagramAIO` read `bs_study.results` and
  `bs_study.studied_parameter_names` (same read-only pattern).
- `ParamOrbitDiagramAIO` caches `orbit_data.peak_values.cpu().numpy()` per
  result index to avoid repeated GPU transfers.

### Implications for the batched design

1. **`results: list[StudyResult]` is the stable interface.** All plotters
   iterate this list. As long as the batched path populates the same list with
   the same `StudyResult` shape, plotting works unchanged.

2. **`studied_parameter_names` must remain available.** Both plotters use it
   for axis labels and grouping logic.

3. **The interactive plotter's BSE re-creation is the hard constraint.** It
   reads `study_params`, `ode_system`, `solver`, and other components directly
   off the study object, then rebuilds a fresh BSE. Two ways to handle this:
   - **(a) Keep `BasinStabilityStudy` as the public-facing class.** Batching
     happens internally (the `.run()` method calls `integrate_batched` under
     the hood). The study object still carries all components. Plotters see no
     change.
   - **(b) Introduce a `StudyResultSet` data object.** Plotters accept this
     instead of a `BasinStabilityStudy`. The `StudyResultSet` holds `results`,
     `studied_parameter_names`, and the original components needed for
     re-creation. Requires updating `isinstance` checks in `InteractivePlotter`
     and type annotations in `MatplotlibStudyPlotter`. More disruptive.

   Option (a) is strongly preferred -- it preserves backward compatibility
   with zero plotter changes.

4. **`StudyResult` shape is already correct.** The `StudyResult` TypedDict
   (`study_label`, `basin_stability`, `errors`, `n_samples`, `labels`,
   `orbit_data`) is what runs need. Both single-BSE-returned results and
   study-collected results should use this same shape.

## Julia SciML / pyBasin Mapping

This section maps concepts between Julia's SciML (DifferentialEquations.jl) and
pyBasin to clarify what we borrow, what we do differently, and why.

### Where things live

Julia bundles the ODE function, initial conditions, time span, and parameters
into a single `ODEProblem` object. pyBasin distributes these across multiple
classes:

| Concern             | Julia SciML                              | pyBasin (current)                                      |
| ------------------- | ---------------------------------------- | ------------------------------------------------------ |
| ODE right-hand side | `f(du, u, p, t)` (free function)         | `ODESystem.ode(t, y)` (method on class)                |
| Parameters          | `p` argument to `ODEProblem`             | `ode_system.params` (dict on instance)                 |
| Initial conditions  | `u0` argument to `ODEProblem`            | Sampled by `Sampler`, passed to `Solver.integrate(y0)` |
| Time span           | `tspan` argument to `ODEProblem`         | `Solver(t_span=...)` (on the solver)                   |
| Solver tolerances   | kwargs to `solve()` (`abstol`, `reltol`) | `Solver(rtol=..., atol=...)`                           |
| Save points         | `saveat` kwarg to `solve()`              | `Solver(t_eval=...)`                                   |
| Solver algorithm    | Positional arg to `solve()` (`Tsit5()`)  | `JaxSolver(method=Dopri5())`                           |

The only overlap between `ODEProblem` and `ODESystem` is **parameters**.
Everything else is split: ICs come from the `Sampler`, timing lives on the
`Solver`. This is intentional -- basin stability requires sampling many ICs
from a distribution, not specifying one `u0`.

### The ODE function signature

Julia's `f` always receives parameters as an argument:

```julia
function f(du, u, p, t)       # in-place
    du[1] = -p[1] * u[1]      # p is always there
end
```

pyBasin's current `ode` reads parameters from `self`:

```python
def ode(self, t, y):
    alpha = self.params["alpha"]   # implicit state, not an argument
```

This is the root cause of the batching problem. In Julia you can pass a
different `p` per trajectory (via `remake` in an ensemble). In pyBasin you
cannot -- the params are baked into the instance. The Track 2 redesign
(`ode(t, y, p)`) aligns pyBasin with Julia: params become a function
argument, not implicit state.

### Problem construction vs. class construction

Julia creates a problem from independent pieces:

```julia
prob = ODEProblem(f, u0, tspan, p)
```

pyBasin ties the ODE function to its parameters at construction:

```python
ode_system = DuffingODE(params={"delta": 0.08, "k3": 1.0, "A": 0.2})
```

After the redesign, `params` becomes `default_params` -- still set at
construction for human readability and standalone use, but no longer the
only way to supply parameters. The solver can override them by passing a
`p` array to `integrate()`, analogous to Julia's `remake(prob, p=new_p)`.

### Solving

| Julia                                    | pyBasin                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| `sol = solve(prob, Tsit5(); saveat=0.1)` | `t, y = solver.integrate(ode_system, y0)`                                  |
| Returns `ODESolution` (interpolatable)   | Returns raw `(t, y)` tensors, wrapped in `Solution`                        |
| Solver algorithm is a positional arg     | Solver algorithm is a constructor arg on `JaxSolver` / `TorchDiffEqSolver` |

Julia's `solve()` takes the problem + algorithm and returns a rich solution
object with interpolation. pyBasin's `Solver.integrate()` takes the ODE system
and ICs, returns raw tensors. The `Solution` class wraps these with feature
and label bookkeeping -- it is specific to basin stability, not a general ODE
solution type.

### Parameter variation and ensemble simulations

Julia's `EnsembleProblem` runs many trajectories with varied ICs or
parameters, parallelized via a pluggable backend:

```julia
ensemble_prob = EnsembleProblem(prob, prob_func = (prob, i, _) -> remake(prob, p = grid[i]))
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(); trajectories = 1000)
```

Three hooks control the ensemble:

| Hook          | Purpose                                     | pyBasin equivalent                                         |
| ------------- | ------------------------------------------- | ---------------------------------------------------------- |
| `prob_func`   | Modify problem per trajectory (ICs, params) | `StudyParams` → `RunConfig` → `ParamAssignment`            |
| `output_func` | Extract/reduce each solution                | Feature extraction + classification pipeline               |
| `reduction`   | Aggregate batches (with early stopping)     | `BasinStabilityStudy.run()` collecting `list[StudyResult]` |

Key differences:

- **Julia solves trajectories independently**, parallelized by threads /
  processes / GPU. Each trajectory goes through its own ODE solve.
- **pyBasin batches trajectories into one tensor**, integrated in a single
  solver call. All P\*B trajectories share the same time span and solver
  settings, so this is a SIMD-style vectorized solve, not independent solves.
- Julia's `EnsembleGPUArray` recompiles the ODE to GPU threads (still
  independent solves). pyBasin's JAX vmap / torchdiffeq batch operate on
  the entire tensor at once.
- Julia's `prob_func` can change anything (ODE function, time span, solver).
  pyBasin's `StudyParams` only varies ODE parameters (and occasionally
  sampler bounds). This is sufficient for basin stability studies.
- Julia's `reduction` supports early stopping (halt when variance is low
  enough). pyBasin does not have this yet -- all P combinations run to
  completion.

### What pyBasin borrows from Julia

1. **Parameters as a function argument** (Track 2 redesign): `ode(t, y, p)`
   mirrors Julia's `f(du, u, p, t)`. This is the single most important
   alignment -- it makes parameter variation a first-class feature of the
   ODE system rather than a mutation hack.

2. **`remake` semantics**: Julia's `remake(prob, p=new_p)` creates a new
   problem with different params. pyBasin's `params_to_array()` +
   `integrate(..., params=grid)` achieves the same effect without copying
   the ODE system -- the solver just passes a different `p` row.

3. **Separation of structural and study parameters**: Julia's `p` can be
   any type (vector, named tuple). pyBasin's `PARAM_KEYS` explicitly
   declares which dict entries are study parameters (go into `p`) vs.
   structural state (stays on `self`).

### What pyBasin does differently

1. **Batch-first integration**: Julia solves trajectories independently
   and parallelizes across them. pyBasin stacks all trajectories into a
   tensor and solves once. Both valid; pyBasin's approach is better when
   all trajectories share the same time span and settings (always true for
   basin stability).

2. **No general-purpose ensemble**: pyBasin has no equivalent of
   `EnsembleProblem`. `BasinStabilityStudy` is purpose-built for parameter
   sweeps in basin stability. It does not support arbitrary reductions or
   early stopping.

3. **Feature extraction pipeline**: Julia's `output_func` is a simple
   reduction. pyBasin has a full pipeline (Solution → features → classifier
   → labels → BS) that is specific to basin stability analysis.

4. **Caching**: pyBasin caches integration results to disk. Julia's SciML
   ecosystem does not cache by default -- results are recomputed or stored
   by the user.

## Resolved Questions

1. **Templates are fixed across parameter sets.** The `TemplateIntegrator`
   returns training data based on its own initial conditions and ODE params,
   which do not change during the study. Templates can be integrated once and
   reused for all P parameter combinations.

2. **Assume all parameters fit in memory for now.** A separate experiment
   should determine the practical P\*B limit and sub-batching strategy, but
   the initial implementation can skip memory-budget logic.

3. **Integration is always parameter-aware; P=1 is the default.** When
   `params=None` (standalone BSE), the solver uses `ode_system.params` as
   today -- effectively P=1. When `params` is a `(P, n_ode_params)` array
   (study mode), the solver flattens P\*B and integrates in one call. No
   fallback logic needed; the two cases are the same code path.

4. **Feature extraction parallelism is already handled.** `TorchFeatureExtractor`
   manages its own parallelism (CPU multiprocessing or GPU batched CUDA ops).
   No additional `ThreadPoolExecutor` wrapper needed around the per-parameter
   feature extraction loop.

5. **Prefer a clean parameter mapping contract.** A dict like
   `{"T": [0.1, 0.2, ...], "K": [1.0, 1.0, ...]}` is preferable to the
   current string-expression approach. However, the final interface depends
   on implementation details: if the batched solver does not need to know about
   solver hyperparams (rtol, atol, etc.) -- only ODE params -- then the clean
   dict approach works. The current `StudyParams` string expressions
   (`ode_system.params["T"]`) can be parsed to extract parameter names and
   values for the dict.
