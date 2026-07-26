# pyright: basic

"""Test per-state feature configuration with TsfreshFeatureExtractor.

This demonstrates how to apply different tsfresh feature sets to different
state variables based on domain knowledge about the system.
"""

import sys
import time
from pathlib import Path
from typing import cast

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

from pybasin.feature_extractors.tsfresh_feature_extractor import TsfreshFeatureExtractor
from pybasin.sampler import GridSampler
from pybasin.solution import Solution
from pybasin.solvers import TorchOdeSolver


def test_per_state_configuration():
    """Test different feature extraction strategies per state variable."""

    print("=" * 80)
    print("Testing Per-State Feature Configuration")
    print("=" * 80)

    n_test = 1000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Setting up on device: {device}")

    # Pendulum parameters
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumODE(params)
    print(f"   ✓ Created ODE system with params: {params}")

    # Sampler
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device=device,
    )

    # Solver
    solver = TorchOdeSolver(
        time_span=(0, 1000),
        n_steps=500,
        device=device,
        method="dopri5",
        rtol=1e-8,
        atol=1e-6,
    )

    # Generate and solve
    print(f"\n2. Generating and solving {n_test} trajectories")
    initial_conditions = sampler.sample(n_test)
    time_arr, y_arr = solver.integrate(ode_system, initial_conditions)
    solution = Solution(
        initial_condition=initial_conditions,
        time=time_arr,
        y=y_arr,
        model_params=cast(dict[str, float], params),
    )
    print(f"   ✓ Solution shape: {solution.y.shape}")

    # Test 1: Uniform features (baseline)
    print("\n" + "=" * 80)
    print("Test 1: UNIFORM features (minimal for both states)")
    print("-" * 80)

    extractor_uniform = TsfreshFeatureExtractor(
        time_steady=950.0,
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=-1,
        normalize=True,
    )

    t_start = time.perf_counter()
    features_uniform = extractor_uniform.extract_features(solution)
    t_elapsed = time.perf_counter() - t_start

    print(f"   ✓ Extraction time: {t_elapsed:.4f}s")
    print(f"   ✓ Features shape: {features_uniform.shape}")
    print(f"   ✓ Features per trajectory: {features_uniform.shape[1]}")
    print(f"   ✓ Feature range: [{features_uniform.min():.4f}, {features_uniform.max():.4f}]")

    # Test 2: Per-state configuration
    print("\n" + "=" * 80)
    print("Test 2: PER-STATE configuration")
    print("   State 0 (θ position): Minimal features (basic stats)")
    print("   State 1 (θ̇ velocity): Comprehensive features (full spectral analysis)")
    print("-" * 80)

    extractor_per_state = TsfreshFeatureExtractor(
        time_steady=950.0,
        kind_to_fc_parameters={
            0: MinimalFCParameters(),  # Position: basic statistics sufficient
            1: ComprehensiveFCParameters(),  # Velocity: needs spectral analysis for periodicity
        },
        n_jobs=-1,
        normalize=True,
    )

    t_start = time.perf_counter()
    features_per_state = extractor_per_state.extract_features(solution)
    t_elapsed = time.perf_counter() - t_start

    print(f"   ✓ Extraction time: {t_elapsed:.4f}s")
    print(f"   ✓ Features shape: {features_per_state.shape}")
    print(f"   ✓ Features per trajectory: {features_per_state.shape[1]}")
    print(f"   ✓ Feature range: [{features_per_state.min():.4f}, {features_per_state.max():.4f}]")

    # Test 3: Custom specific features
    print("\n" + "=" * 80)
    print("Test 3: CUSTOM SPECIFIC features (hand-picked extractors)")
    print("   Using: mean, variance, maximum, minimum, median")
    print("-" * 80)

    # Define specific features using tsfresh's feature name format
    # See: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
    custom_fc_parameters = {
        "mean": None,
        "variance": None,
        "maximum": None,
        "minimum": None,
        "median": None,
        "standard_deviation": None,
        "skewness": None,
        "kurtosis": None,
    }

    extractor_custom = TsfreshFeatureExtractor(
        time_steady=950.0,
        default_fc_parameters=custom_fc_parameters,
        n_jobs=-1,
        normalize=True,
    )

    t_start = time.perf_counter()
    features_custom = extractor_custom.extract_features(solution)
    t_elapsed = time.perf_counter() - t_start

    print(f"   ✓ Extraction time: {t_elapsed:.4f}s")
    print(f"   ✓ Features shape: {features_custom.shape}")
    print(f"   ✓ Features per trajectory: {features_custom.shape[1]}")
    print(f"   ✓ Feature range: [{features_custom.min():.4f}, {features_custom.max():.4f}]")

    # Test 4: Custom per-state specific features
    print("\n" + "=" * 80)
    print("Test 4: CUSTOM PER-STATE features (different hand-picked extractors per state)")
    print("   State 0 (θ position): mean, max, min (basic stats for equilibrium detection)")
    print("   State 1 (θ̇ velocity): variance, skewness, autocorrelation (dynamics analysis)")
    print("-" * 80)

    # Position state: simple statistics to identify equilibrium points
    position_features = {
        "mean": None,
        "maximum": None,
        "minimum": None,
        "median": None,
    }

    # Velocity state: features for dynamics/oscillation detection
    velocity_features = {
        "mean": None,
        "variance": None,
        "standard_deviation": None,
        "skewness": None,
        "kurtosis": None,
        "abs_energy": None,
        "root_mean_square": None,
        "autocorrelation": [{"lag": 10}],  # Parameterized feature
    }

    extractor_custom_per_state = TsfreshFeatureExtractor(
        time_steady=950.0,
        kind_to_fc_parameters={
            0: position_features,
            1: velocity_features,
        },
        n_jobs=-1,
        normalize=True,
    )

    t_start = time.perf_counter()
    features_custom_per_state = extractor_custom_per_state.extract_features(solution)
    t_elapsed = time.perf_counter() - t_start

    print(f"   ✓ Extraction time: {t_elapsed:.4f}s")
    print(f"   ✓ Features shape: {features_custom_per_state.shape}")
    print(f"   ✓ Features per trajectory: {features_custom_per_state.shape[1]}")
    print(
        f"   ✓ Feature range: [{features_custom_per_state.min():.4f}, {features_custom_per_state.max():.4f}]"
    )

    # Test 5: Comprehensive for both (comparison)
    print("\n" + "=" * 80)
    print("Test 5: COMPREHENSIVE features (for both states)")
    print("-" * 80)

    extractor_comprehensive = TsfreshFeatureExtractor(
        time_steady=950.0,
        default_fc_parameters=ComprehensiveFCParameters(),
        n_jobs=-1,
        normalize=True,
    )

    t_start = time.perf_counter()
    features_comprehensive = extractor_comprehensive.extract_features(solution)
    t_elapsed = time.perf_counter() - t_start

    print(f"   ✓ Extraction time: {t_elapsed:.4f}s")
    print(f"   ✓ Features shape: {features_comprehensive.shape}")
    print(f"   ✓ Features per trajectory: {features_comprehensive.shape[1]}")
    print(
        f"   ✓ Feature range: [{features_comprehensive.min():.4f}, {features_comprehensive.max():.4f}]"
    )

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("-" * 80)
    print(f"   Uniform (minimal):        {features_uniform.shape[1]:5d} features")
    print(f"   Per-state (mixed):        {features_per_state.shape[1]:5d} features")
    print(f"   Custom (specific):        {features_custom.shape[1]:5d} features")
    print(f"   Custom per-state:         {features_custom_per_state.shape[1]:5d} features")
    print(f"   Comprehensive (both):     {features_comprehensive.shape[1]:5d} features")
    print()
    print("   Insight: Feature configuration strategies:")
    print("   • MinimalFCParameters(): Fast, ~20 features per state")
    print("   • ComprehensiveFCParameters(): Full extraction, ~800 features per state")
    print("   • Custom dict: Hand-pick specific features for domain knowledge")
    print("   • kind_to_fc_parameters: Different features per state variable")
    print("   • Parameterized features: e.g., autocorrelation with specific lag")

    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_per_state_configuration()
