# pyright: basic

"""
Benchmark nolds feature extractor on all case studies.

This script measures how long it takes to extract Lyapunov exponent and
correlation dimension features using the NoldsFeatureExtractor.
"""

import time
import warnings

import torch
from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from case_studies.friction.setup_friction_system import setup_friction_system
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system

from pybasin.feature_extractors.nolds_feature_extractor import NoldsFeatureExtractor
from pybasin.solution import Solution

warnings.filterwarnings(
    "ignore", message="autocorrelation declined too slowly", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="RANSAC did not reach consensus", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="R\\^2 score is not well-defined", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="signal has very low mean frequency", category=RuntimeWarning
)

N_SAMPLES = 1000

SYSTEMS = [
    ("pendulum", setup_pendulum_system),
    ("lorenz", setup_lorenz_system),
    ("friction", setup_friction_system),
    ("duffing", setup_duffing_oscillator_system),
]


def benchmark_extractor(
    extractor: NoldsFeatureExtractor,
    solution: Solution,
) -> tuple[float, torch.Tensor]:
    """Benchmark the feature extractor."""
    start = time.perf_counter()
    features = extractor.extract_features(solution)
    elapsed = time.perf_counter() - start
    return elapsed, features


def benchmark_system(system_name: str, setup_fn) -> dict[str, float]:
    """Benchmark the nolds extractor on a single system."""
    print(f"\n{'=' * 60}")
    print(f"System: {system_name}")
    print(f"{'=' * 60}")

    props = setup_fn()
    sampler = props["sampler"]
    solver = props.get("solver")
    ode_system = props["ode_system"]
    feature_extractor = props.get("feature_extractor")

    print(f"  Sampling {N_SAMPLES} initial conditions...")
    y0 = sampler.sample(N_SAMPLES)
    print(f"  Initial conditions shape: {y0.shape}")

    print("  Integrating ODE...")
    t_start = time.perf_counter()
    t, y = solver.integrate(ode_system, y0)
    integration_time = time.perf_counter() - t_start
    print(f"  Integration time: {integration_time:.3f}s")

    solution = Solution(initial_condition=y0, time=t, y=y)
    print(f"  Solution shape: {solution.y.shape}")

    time_steady = feature_extractor.time_steady

    results: dict[str, float] = {"integration": integration_time}

    lyap_extractor = NoldsFeatureExtractor(
        time_steady=time_steady,
        features={"lyap_r": None},
    )
    print("  Extracting Lyapunov features...")
    lyap_time, lyap_features = benchmark_extractor(lyap_extractor, solution)
    print(f"    Time: {lyap_time:.3f}s ({lyap_time / N_SAMPLES * 1000:.2f}ms/sample)")
    print(f"    Features shape: {lyap_features.shape}")
    print(f"    Feature range: [{lyap_features.min():.4f}, {lyap_features.max():.4f}]")
    results["lyapunov"] = lyap_time

    corr_extractor = NoldsFeatureExtractor(
        time_steady=time_steady,
        features={"corr_dim": None},
    )
    print("  Extracting Correlation Dimension features...")
    corr_time, corr_features = benchmark_extractor(corr_extractor, solution)
    print(f"    Time: {corr_time:.3f}s ({corr_time / N_SAMPLES * 1000:.2f}ms/sample)")
    print(f"    Features shape: {corr_features.shape}")
    print(f"    Feature range: [{corr_features.min():.4f}, {corr_features.max():.4f}]")
    results["correlation_dimension"] = corr_time

    all_extractor = NoldsFeatureExtractor(time_steady=time_steady)
    print("  Extracting all features (combined)...")
    all_time, all_features = benchmark_extractor(all_extractor, solution)
    print(f"    Time: {all_time:.3f}s ({all_time / N_SAMPLES * 1000:.2f}ms/sample)")
    print(f"    Features shape: {all_features.shape}")
    print(f"    Feature names: {all_extractor.feature_names[:4]}...")
    results["all_combined"] = all_time

    custom_extractor = NoldsFeatureExtractor(
        time_steady=time_steady,
        features={
            "lyap_r": [
                {"emb_dim": 5},
                {"emb_dim": 10},
                {"emb_dim": 15},
            ],
        },
    )
    print("  Extracting custom multi-param features...")
    custom_time, custom_features = benchmark_extractor(custom_extractor, solution)
    print(f"    Time: {custom_time:.3f}s ({custom_time / N_SAMPLES * 1000:.2f}ms/sample)")
    print(f"    Features shape: {custom_features.shape}")
    print(f"    Feature names: {custom_extractor.feature_names[:3]}")
    results["custom"] = custom_time

    results["total_extraction"] = lyap_time + corr_time

    return results


def main():
    """Run benchmarks on all systems."""
    print("=" * 60)
    print("Nolds Feature Extractor Benchmark")
    print(f"N samples: {N_SAMPLES}")
    print("=" * 60)

    all_results: dict[str, dict[str, float]] = {}

    for system_name, setup_fn in SYSTEMS:
        results = benchmark_system(system_name, setup_fn)
        all_results[system_name] = results

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"{'System':<15} {'Integration':>12} {'Lyapunov':>12} {'CorrDim':>12} "
        f"{'Combined':>12} {'Custom':>12}"
    )
    print("-" * 90)

    total_integration = 0.0
    total_lyap = 0.0
    total_corr = 0.0
    total_combined = 0.0
    total_custom = 0.0

    for system_name, results in all_results.items():
        print(
            f"{system_name:<15} "
            f"{results['integration']:>11.3f}s "
            f"{results['lyapunov']:>11.3f}s "
            f"{results['correlation_dimension']:>11.3f}s "
            f"{results['all_combined']:>11.3f}s "
            f"{results['custom']:>11.3f}s"
        )
        total_integration += results["integration"]
        total_lyap += results["lyapunov"]
        total_corr += results["correlation_dimension"]
        total_combined += results["all_combined"]
        total_custom += results["custom"]

    print("-" * 90)
    print(
        f"{'TOTAL':<15} "
        f"{total_integration:>11.3f}s "
        f"{total_lyap:>11.3f}s "
        f"{total_corr:>11.3f}s "
        f"{total_combined:>11.3f}s "
        f"{total_custom:>11.3f}s"
    )

    total_samples = N_SAMPLES * len(SYSTEMS)
    print("\n" + "=" * 90)
    print("PER-SAMPLE AVERAGES")
    print("=" * 90)
    print(f"  Lyapunov:              {total_lyap / total_samples * 1000:.2f} ms/sample")
    print(f"  Correlation Dimension: {total_corr / total_samples * 1000:.2f} ms/sample")
    print(f"  Combined (default):    {total_combined / total_samples * 1000:.2f} ms/sample")
    print(f"  Custom (3x lyap_r):    {total_custom / total_samples * 1000:.2f} ms/sample")


if __name__ == "__main__":
    main()
