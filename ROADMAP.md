# PyBasin Roadmap

- [ ] Allow classifiers to define ODE parameters per initial condition
  - Currently `KNNClassifier` accepts a single `ode_params` for all templates
  - bSTAB allows defining different parameters for each template initial condition
  - This would enable more flexible template matching across parameter spaces
- [ ] When varying parameters we should not vary initial conditions
- [ ] Optimize parameters variation (not hyper-parameters)
- [x] Improve plotter API. Calling plt.show does not feels right

```python
plotter.plot_templates_trajectories(
  plotted_var=0,
  y_limits=(-1.4, 1.4),
  x_limits=(0, 50),
)
plt.show()  # type: ignore[misc]
```

- [ ] Look into https://github.com/lmcinnes/umap for feature space visualization
- [x] Rename as_parameter_manager to bs_study_parameter_manager and as_bse to bs_study
- [x] Fix Installation guideline, find out how to deploy to pip
- [ ] Using JAX SaveAt and setting diffeqsolve.t0 = 0 we can make JAX return the transient time and save a lot of memory. Need to check if that behaviour applies to other solvers. This could help a lot for parameter sweeps with batch integration, saving 50 points intead of 1000 virtually saves 20x space
- [ ] Review Equinox usage for Diffrax solver
- [ ] Introduce a dataset library like https://nolds.readthedocs.io/en/latest/nolds.html#benchmark-dataset-for-hurst-exponent
- [ ] Supervised feature filtering support
- [ ] State space plot for the basin stability should support many dimensions
- [ ] Extent the capabilities of the toolbox to maps and network systems.
- [ ] Batched GPU integration for network studies with fixed topology shape (e.g. Watts-Strogatz rewiring sweeps where N and k are constant). Generalize solver args from flat `(batch, n_params)` to arbitrary pytrees so structural data like edge indices can be vmapped alongside scalar ODE parameters.
- [ ] Can piecewise functions be supported with events? https://github.com/rtqichen/torchdiffeq/blob/master/examples/bouncing_ball.py
