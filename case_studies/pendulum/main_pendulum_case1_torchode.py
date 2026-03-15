from case_studies.pendulum.setup_pendulum_system_torchode import (
    setup_pendulum_system_torchode,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter import InteractivePlotter
from pybasin.utils import time_execution


def main():
    props = setup_pendulum_system_torchode()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        output_dir="results_case1_torchode",
        feature_selector=None,
    )

    result = bse.run()
    print("Basin Stability:", result["basin_stability"])

    # bse.save()
    # bse.save_to_excel()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_pendulum_case1_torchode.py", main)
    plotter = InteractivePlotter(bse, state_labels={0: "θ", 1: "ω"})
    plotter.run(port=8050)
