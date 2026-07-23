---
name: experiment-design
description: "Design or review pybasin experiments and benchmarks where measurement structure matters. Use when creating or fixing notebooks or scripts for solver, trajectory, or feature-extraction comparisons and you need to choose `JaxSolver` save semantics (`t_span`, `t_steps`, `t_eval`), decide what should be integrated once and reused versus reintegrated per batch size, keep hot paths batched, reuse expensive benchmark artifacts across notebook cells, or report results in one decision-focused table without duplicated summaries."
---

# Experiment Design

Build experiments so the measurement answers one decision question and does not get polluted by setup noise.

Read [references/benchmark-defaults.md](references/benchmark-defaults.md) for concrete defaults, table rules, caching rules, and anti-patterns.

## Workflow

### Define the measurement target

State exactly what is being timed before writing code.

- If the question is feature extraction speed, do not integrate inside the timed loop.
- If the question is solver speed, do not mix feature extraction or plotting into the timed path.
- If the question compares two implementations of the same operation, keep feature set, trajectory tensor, dtype, device, and steady-state slice identical.

### Structure reuse first

Design the notebook so expensive work is computed once and reused unless the experiment definition truly changes.

- Reuse full integrated trajectories across steady-state slices.
- Reuse cached benchmark artifacts across later notebook cells.
- Reintegrate only when batch size, initial-condition density, solver settings, or model parameters are intentionally different.

### Use solver arguments correctly

Distinguish these arguments carefully:

- `t_span`: full integration interval.
- `t_steps`: number of time points across the full save region, not "steady-state steps".
- `t_eval`: restrict saved time region when the experiment truly needs only part of the trajectory saved.
- `rtol`, `atol`: accuracy controls; keep fixed during implementation comparisons unless tolerance sensitivity is the subject.
- `device`: solver backend target; keep aligned with the experiment goal.
- `cache_dir`: disable caching for fair one-shot timing unless the experiment is explicitly about cached reuse.

Prefer slicing the final `N` saved steps from one solve over changing `t_eval` or reintegrating when only the steady-state window length changes.

### Keep the hot path batched

Avoid Python loops in the timed path when tensor operations can evaluate all trajectories and all states at once.

A loop that only builds labels or metadata outside the timed path is acceptable. A loop that processes each state or trajectory inside the measured extraction path is usually not.

For GPU experiments, synchronize immediately before starting the timer and immediately after the measured call.

### Report only decision-making information

- Default to one compact comparison table.
- Do not print the same benchmark result in multiple overlapping tables.
- Keep diagnostic tables separate from the main decision table.
- Keep comparable plots on shared scales.

## Review Checklist

Check these first:

- Is integration accidentally inside the benchmark loop?
- Is `t_steps` being confused with the final steady-state window length?
- Is `t_eval` being used when simple post-solve slicing would be cleaner?
- Is expensive work being recomputed in later cells instead of reused from notebook state?
- Are batch-size comparisons created by slicing or by reintegrating, and is that choice actually correct for the experiment definition?
- Are implementation comparisons using the same features, same trajectories, same dtype, and same device?
- Is a Python `for` loop doing work that should be one batched tensor operation?
- Is the reported output compact enough to support a decision without repeated information?
- Do helper signatures expose fake configurability or unused arguments?
