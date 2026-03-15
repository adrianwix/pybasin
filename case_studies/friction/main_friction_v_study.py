import numpy as np

from case_studies.friction.setup_friction_system import setup_friction_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution

FRICTION_V_D_VALUES: list[float] = [float(v) for v in np.linspace(0.8, 2.225, 20)]


def main():
    props = setup_friction_system()

    study_params = SweepStudyParams(
        **{'ode_system.params["v_d"]': FRICTION_V_D_VALUES},
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert estimator is not None, "estimator is required for BasinStabilityStudy"

    bse = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        output_dir="results_friction_vd_study",
    )

    print("Estimating Basin Stability...")
    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_friction_v_study.py", main)

    plotter = MatplotlibStudyPlotter(bse)

    plotter.plot_parameter_stability()
    plotter.plot_orbit_diagram([1])
    plotter.show()

    bse.save()
