"""Validates the shared, case-study-agnostic extract/classify graphs
(`extract_features_dim{N}.pte`, `classify_supervised.pte`) against each case study's diffrax
reference -- confirms that selecting the same named features the production KNN classifier
already uses, out of the new generic 10-stat-per-state output, reproduces the reference
labels when applied to the *same* (diffrax) trajectories. See feature_extract_graph.py /
classify_graph.py's module docstrings for why these graphs are shared across case studies
instead of one export per case study.

Unlike Phase 1/2's validation scripts, this doesn't compare against a separate candidate
integrator (`TorchRK4Solver`, since removed): integration is zigode's job in the deployed demo,
already validated on its own terms by zigode's own accuracy/performance tests, so isolating
just the feature-extraction/classification change here is both simpler and the more accurate
thing to check.
"""

import numpy as np
import torch
from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE
from classify_graph import ClassifyGraph
from feature_extract_graph import FEATURE_NAMES, FeatureExtractGraph

from pybasin.feature_extractors.tstorch_feature_extractor import TorchFeatureExtractorAdapter
from pybasin.predictors import KNeighborsClassifier
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler
from pybasin.solution import Solution
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator

N = 1000
MIN_AGREEMENT = 0.99


def column_indices(selection: dict[int, list[str]]) -> list[int]:
    """Column indices into FeatureExtractGraph's `(n, state_dim * 10)` output for a
    `{state_idx: [feature_name, ...]}` selection -- what a UI "choose features" control
    produces client-side."""
    return [
        state_idx * len(FEATURE_NAMES) + FEATURE_NAMES.index(name)
        for state_idx, names in selection.items()
        for name in names
    ]


def steady_window(y: torch.Tensor, t: torch.Tensor, time_steady: float) -> torch.Tensor:
    """Slices `y` (n, state_dim, t_steps) to the steady-state window `t >= time_steady`,
    matching `FeatureExtractor.filter_time`'s convention -- `FeatureExtractGraph` itself takes
    an already-windowed trajectory, it doesn't filter time internally."""
    start_idx = int(torch.where(t >= time_steady)[0][0])
    return y[..., start_idx:]


def check(
    name,  # type: ignore[no-untyped-def]
    ref_kwargs,
    ref_time_steady,
    jax_ode_cls,
    params,
    sampler,
    ref_feature_spec,
    selection,
    template_y0,
    labels,
):
    y0 = sampler.sample(N)
    state_dim = y0.shape[1]

    ref_solver = JaxSolver(cache_dir=None, device="cpu", **ref_kwargs)
    t_ref, y_ref = ref_solver.integrate(jax_ode_cls(params), y0)
    fe_ref = TorchFeatureExtractorAdapter(
        time_steady=ref_time_steady,
        features=None,
        normalize=False,
        features_per_state=ref_feature_spec,
    )
    feat_ref = fe_ref.extract_features(Solution(initial_condition=y0, time=t_ref, y=y_ref))
    ti_ref = TemplateIntegrator(template_y0=template_y0, labels=labels, ode_params=params)
    ti_ref.integrate(ref_solver, jax_ode_cls(params))
    train_x_ref, train_y = ti_ref.get_training_data(fe_ref)
    knn_ref = KNeighborsClassifier(n_neighbors=1)
    knn_ref.fit(train_x_ref, train_y)
    pred_ref = knn_ref.predict(feat_ref)

    extract = FeatureExtractGraph(state_dim=state_dim)
    cols = column_indices(selection)
    y_windowed = steady_window(y_ref, t_ref, ref_time_steady)
    with torch.no_grad():
        features = extract(y_windowed)[:, cols]

    template_y0_t = torch.tensor(template_y0, dtype=torch.float32)
    t_tmpl, y_tmpl = ref_solver.integrate(jax_ode_cls(params), template_y0_t)
    y_tmpl_windowed = steady_window(y_tmpl, t_tmpl, ref_time_steady)
    with torch.no_grad():
        reference_features = extract(y_tmpl_windowed)[:, cols]

    classify = ClassifyGraph(unsupervised=False)
    with torch.no_grad():
        pred_idx = classify(features, reference_features)
    pred = np.array([labels[i] for i in pred_idx.tolist()])

    agree = (pred_ref == pred).mean()
    print(f"{name}: n={N} agreement={agree * 100:.1f}%")
    assert agree >= MIN_AGREEMENT, f"{name}: agreement {agree:.3f} below {MIN_AGREEMENT}"


def main() -> None:
    duffing_params = {"delta": 0.08, "k3": 1, "A": 0.2}
    check(
        "Duffing",
        {
            "t_span": (0, 1000),
            "t_steps": 5000,
            "t_eval": (900.0, 1000.0),
            "rtol": 1e-8,
            "atol": 1e-6,
        },
        900.0,
        DuffingJaxODE,
        duffing_params,
        UniformRandomSampler(BoxRegion([-1, -0.5], [1, 1]), device="cpu"),
        {0: {"maximum": None, "standard_deviation": None}},
        {0: ["maximum", "standard_deviation"]},
        [[-0.21, 0.02], [1.05, 0.77], [-0.67, 0.02], [-0.46, 0.30], [-0.43, 0.12]],
        ["y1", "y2", "y3", "y4", "y5"],
    )

    pendulum_params = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    check(
        "Pendulum",
        {"t_span": (0, 1000), "t_steps": 1000},
        950.0,
        PendulumJaxODE,
        pendulum_params,
        UniformRandomSampler(
            BoxRegion(
                [
                    -np.pi + np.arcsin(pendulum_params["T"] / pendulum_params["K"]),
                    -10.0,
                ],
                [
                    np.pi + np.arcsin(pendulum_params["T"] / pendulum_params["K"]),
                    10.0,
                ],
            ),
            device="cpu",
        ),
        {1: {"log_delta": None}},
        {1: ["log_delta"]},
        [[0.4, 0.0], [2.7, 0.0]],
        ["FP", "LC"],
    )


if __name__ == "__main__":
    main()
