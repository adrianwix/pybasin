# PyBasin Roadmap

## Bugs

- [ ] `t_eval` in the solver (e.g. `t_eval=(950.0, 1000.0)`) is intended to save memory by keeping only the steady-state window, but this same truncated trajectory is passed to the interactive and matplotlib plotters — so trajectory plots show only the saved slice rather than the full time series. Plotters need access to the full trajectory, or the solver should re-integrate without `t_eval` when plotting is requested.

## Features

- [ ] Optimize orbit data computation — currently ~33s for pendulum case 2. Peak extraction runs on the full B×P trajectory tensor and is the dominant cost; investigate batched or sparse approaches.
- [ ] Improve logging experience for parameter case studies — steps 5–7 are currently silent during the run and only appear aggregated in the final summary. Consider emitting a compact per-group progress line (e.g. `[3/20] T=0.3 → FP=0.72 LC=0.28`) as each group completes so users get feedback during long sweeps. Also fix double-logging for steps 1–4, which are printed immediately by `StepTimer.step()` and then repeated in the final summary table.
- [ ] Allow classifiers to define ODE parameters per initial condition
  - Currently `KNNClassifier` accepts a single `ode_params` for all templates
  - bSTAB allows defining different parameters for each template initial condition
  - This would enable more flexible template matching across parameter spaces
- [ ] When varying parameters we should not vary initial conditions
- [x] Optimize parameters variation (not hyper-parameters)
- [ ] Look into https://github.com/lmcinnes/umap for feature space visualization
- [x] Using JAX SaveAt and setting diffeqsolve.t0 = 0 we can make JAX return the transient time and save a lot of memory. Need to check if that behaviour applies to other solvers. This could help a lot for parameter sweeps with batch integration, saving 50 points intead of 1000 virtually saves 20x space
- [ ] Introduce a dataset library like https://nolds.readthedocs.io/en/latest/nolds.html#benchmark-dataset-for-hurst-exponent
- [ ] Supervised feature filtering support
- [ ] State space plot for the basin stability should support many dimensions
- [ ] Extent the capabilities of the toolbox to maps and network systems.
- [ ] Batched GPU integration for network studies with fixed topology shape (e.g. Watts-Strogatz rewiring sweeps where N and k are constant). Generalize solver args from flat `(batch, n_params)` to arbitrary pytrees so structural data like edge indices can be vmapped alongside scalar ODE parameters.
- [ ] Can piecewise functions be supported with events? https://github.com/rtqichen/torchdiffeq/blob/master/examples/bouncing_ball.py
