from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.basin_stability_estimator import BasinStabilityEstimator


def main():
    props = setup_pendulum_system()

    bse = BasinStabilityEstimator(
        ode_system=props["ode_system"],
        sampler=props["sampler"],
    )

    bse.run()

    return bse


if __name__ == "__main__":
    main()
