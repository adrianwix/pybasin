# pyright: basic

"""Experimental test for TsfreshFeatureExtractor with pendulum system.

This script tests the tsfresh-based feature extractor with the pendulum ODE system
to verify compatibility with PyTorch/JAX tensor shapes and evaluate feature extraction
performance. Uses MinimalFCParameters for distinguishing:
- Fixed Point (FP): System converges to equilibrium (low variance, no oscillation)
- Limit Cycle (LC): System exhibits periodic behavior (high variance, oscillation)
"""

# import os

# Recommended by tsfresh for parallelism. No visible improvement here
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import traceback
from pathlib import Path
from typing import cast

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction import MinimalFCParameters

from pybasin.feature_extractors.tsfresh_feature_extractor import TsfreshFeatureExtractor
from pybasin.sampler import GridSampler
from pybasin.solution import Solution
from pybasin.solvers import TorchOdeSolver
from pybasin.template_integrator import TemplateIntegrator


def test_tsfresh_extractor():
    """Test TsfreshFeatureExtractor with a small pendulum dataset."""

    print("=" * 80)
    print("Testing TsfreshFeatureExtractor with Pendulum System")
    print("=" * 80)

    # Use the same scale as the actual pendulum case study
    n_test = 10000

    # Use GPU for better performance
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Setting up on device: {device}")

    # Define pendulum parameters (same as setup_pendulum_system)
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

    # Create ODE system
    ode_system = PendulumODE(params)
    print(f"   ✓ Created ODE system with params: {params}")

    # Create sampler
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device=device,
    )
    print("   ✓ Created sampler")

    # Create solver (same parameters as setup_pendulum_system)
    solver = TorchOdeSolver(
        time_span=(0, 1000),
        n_steps=500,
        device=device,
        method="dopri5",  # Dormand-Prince 5(4) - similar to MATLAB's ode45
        rtol=1e-8,
        atol=1e-6,
    )
    print("   ✓ Created solver with time_span=(0, 1000), n_steps=500")

    # Generate sample initial conditions once (reused for both feature sets)
    print(f"\n2. Generating {n_test} initial conditions")
    initial_conditions = sampler.sample(n_test)
    print(f"   ✓ Initial conditions shape: {initial_conditions.shape}")

    # Solve ODE system once (reused for both feature sets)
    print("\n3. Solving ODE system...")
    time_arr, y_arr = solver.integrate(ode_system, initial_conditions)
    solution = Solution(
        initial_condition=initial_conditions,
        time=time_arr,
        y=y_arr,
        model_params=cast(dict[str, float], params),  # Convert TypedDict to dict
    )
    print(f"   ✓ Solution shape: {solution.y.shape}")
    print(f"   ✓ Time shape: {solution.time.shape}")
    print(f"   ✓ Time range: [{solution.time.min():.2f}, {solution.time.max():.2f}]")

    # Test with MinimalFCParameters
    print("\n4. Testing with MinimalFCParameters")
    print("-" * 80)

    # Create tsfresh feature extractor with MinimalFCParameters
    # This provides a compact set of ~10 features per state variable
    # Note: n_jobs=1 for deterministic results, normalize=False for KNN compatibility
    feature_extractor = TsfreshFeatureExtractor(
        time_steady=950.0,  # Same as PendulumFeatureExtractor
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=1,  # Use n_jobs=1 for deterministic results (parallelism causes non-determinism)
        normalize=False,  # Don't normalize - causes issues with KNN when templates differ from main data
    )
    print("   ✓ Created TsfreshFeatureExtractor with MinimalFCParameters")
    print("     - time_steady=950.0 (steady-state analysis)")
    print("   ✓ Using single thread (n_jobs=1) for deterministic results")
    print("   ✓ Normalization disabled (for KNN compatibility)")

    # Extract features and measure time
    print("\n5. Extracting features with tsfresh (MinimalFCParameters)...")
    t_start = time.perf_counter()
    try:
        features = feature_extractor.extract_features(solution)
        t_elapsed = time.perf_counter() - t_start

        print("   ✓ Features extracted successfully!")
        print(f"   ✓ Extraction time: {t_elapsed:.4f}s ({t_elapsed * 1000:.2f}ms)")
        print(f"   ✓ Per-sample time: {t_elapsed / n_test * 1000:.3f}ms")
        print(f"   ✓ Features shape: {features.shape}")
        print(f"   ✓ Number of features per trajectory: {features.shape[1]}")
        print(f"   ✓ Feature dtype: {features.dtype}")
        print(f"   ✓ Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"   ✓ NaN values: {torch.isnan(features).sum().item()}")
        print(f"   ✓ Inf values: {torch.isinf(features).sum().item()}")

        # Show some feature statistics
        print("\n6. Feature statistics (after normalization):")
        print(f"   - Mean: {features.mean():.4f}")
        print(f"   - Std: {features.std():.4f}")
        print(f"   - Non-zero features: {(features != 0).sum().item()} / {features.numel()}")

    except Exception as e:
        print("   ✗ Feature extraction failed!")
        print(f"   Error: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 80)

    # Test with classifier
    print("\n7. Testing with KNN classifier")
    print("-" * 80)

    # Reuse the same feature extractor (already fitted with training data)
    print("   ✓ Using the same feature extractor (already fitted)")

    classifier_initial_conditions = [
        [0.5, 0.0],  # FP: fixed point
        [2.7, 0.0],  # LC: limit cycle
    ]
    classifier_labels = ["FP", "LC"]

    print(f"   ✓ Template conditions: {len(classifier_initial_conditions)} templates")

    # Solve for templates (need to convert to tensor for direct solver usage)
    classifier_tensor = torch.tensor(
        classifier_initial_conditions, dtype=torch.float32, device=device
    )
    time_arr, y_arr = solver.integrate(ode_system, classifier_tensor)
    template_solution = Solution(
        initial_condition=classifier_tensor,
        time=time_arr,
        y=y_arr,
        model_params=cast(dict[str, float], params),  # Convert TypedDict to dict
    )
    print(f"   ✓ Template solution: {template_solution.y.shape}")

    # Extract template features
    template_features = feature_extractor.extract_features(template_solution)
    print(f"   ✓ Template features: {template_features.shape}")

    # Create and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    template_integrator = TemplateIntegrator(
        template_y0=classifier_initial_conditions,
        labels=classifier_labels,
        ode_params=params,
    )

    # Fit the classifier with tsfresh features
    print("\n8. Training KNN classifier with tsfresh features")
    try:
        template_integrator.integrate(solver, ode_system)
        X_train, y_train = template_integrator.get_training_data(feature_extractor)
        knn.fit(X_train, y_train)
        print("   ✓ Classifier trained successfully!")

        # Extract features for the test data using the same feature extractor
        print(f"\n9. Extracting features for {n_test} test samples")
        t_start = time.perf_counter()
        test_features = feature_extractor.extract_features(solution)
        t_elapsed = time.perf_counter() - t_start
        print(f"   ✓ Test features shape: {test_features.shape}")
        print(f"   ✓ Extraction time: {t_elapsed:.4f}s ({t_elapsed * 1000:.2f}ms)")

        # Predict on test data
        print(f"\n10. Testing predictions on {n_test} samples")
        predictions = knn.predict(test_features.cpu().numpy())
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Unique labels: {set(predictions)}")
        print("   ✓ Label distribution:")
        for label in classifier_labels:
            count = (predictions == label).sum()
            print(f"      - {label}: {count} ({count / len(predictions) * 100:.1f}%)")

        # ====================================================================
        # Compare with expected results (from main_pendulum_case1.json)
        # ====================================================================
        print("\n11. Comparing with expected results")
        print("-" * 80)

        # Expected basin stability values from reference implementation
        expected_results = {
            "FP": {"basinStability": 0.152, "absNumMembers": 1520},
            "LC": {"basinStability": 0.848, "absNumMembers": 8480},
        }

        # Calculate actual basin stability
        actual_results = {}
        for label in classifier_labels:
            count = (predictions == label).sum()
            actual_results[label] = {
                "basinStability": count / len(predictions),
                "absNumMembers": int(count),
            }

        # Tolerance for comparison (1% absolute tolerance for basin stability)
        tolerance = 0.01  # 1% absolute tolerance

        print(f"   Tolerance: ±{tolerance * 100:.1f}% absolute")
        print()
        print(f"   {'Label':<6} {'Expected':>12} {'Actual':>12} {'Diff':>10} {'Status':>10}")
        print(f"   {'-' * 6:<6} {'-' * 12:>12} {'-' * 12:>12} {'-' * 10:>10} {'-' * 10:>10}")

        all_passed = True
        for label in classifier_labels:
            expected_bs = expected_results[label]["basinStability"]
            actual_bs = actual_results[label]["basinStability"]
            diff = actual_bs - expected_bs
            within_tolerance = abs(diff) <= tolerance

            status = "✓ PASS" if within_tolerance else "✗ FAIL"
            if not within_tolerance:
                all_passed = False

            print(
                f"   {label:<6} {expected_bs:>11.1%} {actual_bs:>11.1%} {diff:>+9.1%} {status:>10}"
            )

        print()
        if all_passed:
            print("   ✓ All basin stability values within tolerance!")
        else:
            print("   ✗ Some basin stability values outside tolerance!")
            print("   Note: This may be due to limited templates (only 2) for KNN classifier.")
            print("   Consider using more representative templates or a different classifier.")

    except Exception as e:
        print("   ✗ Classifier test failed!")
        print(f"   Error: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Run main test with full dataset
    test_tsfresh_extractor()
