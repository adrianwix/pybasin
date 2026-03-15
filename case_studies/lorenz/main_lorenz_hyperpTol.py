"""
Lorenz Hyperparameter Study: Tolerance Variation

This script computes the sensitivity of basin stability values against
the choice of the ODE integration tolerance (rtol).

This is the Python equivalent of the MATLAB script:
bSTAB-M/case_lorenz/main_lorenz_hyperpTol.m

The study varies the relative tolerance (rtol) from 1e-3 to 1e-8 to observe
how the basin stability estimates change with different integration accuracies.
"""

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution

LORENZ_RTOL_VALUES: list[float] = [1.0e-03, 1.0e-04, 1.0e-05, 1.0e-06, 1.0e-07, 1.0e-08]


def main():
    """
    Main function to run the Lorenz hyperparameter tolerance study.

    This study investigates how basin stability estimates vary with
    different ODE integration tolerances (rtol).
    """
    print("=" * 80)
    print("Lorenz System: Hyperparameter Study - Integration Tolerance")
    print("=" * 80)

    # Use the standard setup function to configure the Lorenz system
    props = setup_lorenz_system()

    # Override the sample size for this specific study
    # Matches MATLAB: props.roi.N = 20000
    n = 20000

    # Define the study parameters
    # Varies relative tolerance from 1e-3 to 1e-8
    # Matches MATLAB: props.ap_study.ap_values = [1.0e-03, ..., 1.0e-08]
    study_params = SweepStudyParams(
        **{"solver.rtol": LORENZ_RTOL_VALUES},
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None, "solver is required for BasinStabilityStudy"
    assert feature_extractor is not None, "feature_extractor is required for BasinStabilityStudy"
    assert estimator is not None, "estimator is required for BasinStabilityStudy"

    # Initialize the Basin Stability Study
    bse = BasinStabilityStudy(
        n=n,
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        output_dir="results_hyperpTol",
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_lorenz_hyperpTol.py", main)

    plotter = MatplotlibStudyPlotter(bse)

    plotter.plot_parameter_stability()
    plotter.show()

    bse.save()
