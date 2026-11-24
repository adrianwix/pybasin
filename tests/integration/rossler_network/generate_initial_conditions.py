"""Generate the shared initial-condition CSV for the Rossler network tests."""

from pathlib import Path

import numpy as np
from case_studies.rossler_network.rossler_network_topology import N_NODES
from case_studies.rossler_network.setup_rossler_network_system import (
    setup_rossler_network_system,
)

OUTPUT_PATH = Path(__file__).parent / "ground_truths" / "initial_conditions.csv"
SEED = 42


def main() -> None:
    """Sample the paper's region once and save the rows used by every device."""
    setup = setup_rossler_network_system(device="cpu")
    samples = setup["sampler"].sample(setup["n"], seed=SEED).cpu().numpy()
    columns = [
        *(f"x_{index}" for index in range(N_NODES)),
        *(f"y_{index}" for index in range(N_NODES)),
        *(f"z_{index}" for index in range(N_NODES)),
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT_PATH,
        samples,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


if __name__ == "__main__":
    main()
