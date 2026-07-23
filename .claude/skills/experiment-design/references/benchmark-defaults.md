# Benchmark Defaults

Use these defaults as starting points for pybasin experiments unless the user or existing benchmark already fixes a different setup.

## Common Trajectory Layouts

- GPU-oriented throughput benchmarks: precompute `50_000` trajectories once, then slice subsets such as `5_000`, `10_000`, `20_000`, `50_000`.
- Smaller debugging runs: `128` to `2_048` trajectories.
- Feature-extraction steady-state windows: compare final `2000`, `1000`, `500`, `200` steps by slicing the same saved trajectory tensor.

Exception:

- If the benchmark interprets each batch size as a different sampling density over the same initial-condition region, integrate separately for each batch size and only reuse the resulting full solution across the steady-state window comparisons.

## Common Solver Setup

Typical pendulum-style benchmark:

```python
solver = JaxSolver(
    t_span=(0, 1000),
    t_steps=5000,
    device=device,
    rtol=1e-8,
    atol=1e-6,
    cache_dir=None,
)
```

Interpretation:

- `t_span=(0, 1000)` means integrate over the full time horizon.
- `t_steps=5000` means save 5000 time points over the full saved region.
- A final steady-state window of 2000 steps is a post-solve slice like `y[-2000:]`, not `t_steps=2000`.

## When To Use `t_eval`

Use `t_eval` only when the saved time region itself should be reduced.

Good uses:

- memory reduction when only the final region is ever needed
- production runs where the experiment definition truly saves only a sub-interval

Avoid `t_eval` when:

- comparing several final-window lengths from the same solve
- you can save once and slice many windows cheaply afterward
- the benchmark target is downstream processing rather than solver I/O volume

## Timing Guidance

- Warm up once before repeats.
- Repeat at least 3 to 5 times for notebook-level timing.
- Report median time and optionally min time for best-case throughput.
- For GPU timing, call `torch.cuda.synchronize()` before and after the measured region.
- If those timing controls are fixed for the experiment, keep them as module-level constants and do not mirror them as defaulted helper arguments.
- If the benchmark already computed an expensive trajectory set or other reusable artifact, keep it in notebook state and reuse it in follow-up cells instead of recomputing it.

## Plotting Guidance

- If several figures show `median_ms`, compute one global upper limit from the full results table and reuse it.
- If several figures show `throughput_samples_per_s`, compute one global upper limit from the full results table and reuse it.
- Keep unit labels explicit and stable across figures, for example `median extraction time [ms]` and `throughput [samples/s]`.

## Table Guidance

- Default tables should contain only the columns needed for an informed comparison.
- Prefer one comparison table that directly answers the decision question over multiple overlapping summary tables.
- Do not show the same result twice in different tables unless the second table adds genuinely new information.
- If a diagnostic table exists, keep it separate from the main benchmark comparison and label it as diagnostic.

## Anti-Patterns

Avoid these in benchmark code:

- integrating again for each batch subset when one large solve can be sliced
- integrating again for each steady-state window length when one saved trajectory can be sliced
- recomputing a previously integrated trajectory set in a later notebook cell when the earlier result can be cached and reused
- timing object construction, plotting, or dataframe formatting as part of the core benchmark
- looping over states or trajectories in Python inside the hot path when tensor reductions over `(N, B, S)` are available
- changing feature sets while claiming to compare implementation speed
- helper function signatures that keep unused arguments or defaulted copies of fixed constants
- printing several tables that restate the same benchmark numbers with slightly different labels
