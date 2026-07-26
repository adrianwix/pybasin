# pyright: basic
"""
Pure torchdiffeq implementation of Lorenz basin stability computation with masking.

Replicates the MATLAB implementation but uses LorenzODE's built-in masking for unbounded trajectories.
No event functions needed - masking is handled in the ODE itself when max(|y|) > 200.

Expected results should match main_lorenz.json:
- butterfly1: ~8.9% basin stability
- butterfly2: ~8.7% basin stability
- unbounded: ~82.3% basin stability
"""

import json
import time

import torch
from case_studies.lorenz.lorenz_ode import LorenzODE, LorenzParams
from torchdiffeq import odeint


def solve_batch_trajectories(
    y0_batch: torch.Tensor,
    ode_system: LorenzODE,
    t_span: tuple[float, float] = (0.0, 1000.0),
    n_steps: int = 500,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    method: str = "dopri5",
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve batch of Lorenz trajectories using regular odeint with masking.

    Args:
        y0_batch: initial conditions [batch, 3]
        ode_system: LorenzODE instance with masking
        t_span: integration time span (default matches MATLAB: 0 to 1000)
        n_steps: number of evaluation points
        rtol: relative tolerance
        atol: absolute tolerance
        method: integration method (default 'dopri5')
        device: device to run on

    Returns:
        tuple of (t_eval, y_solution) with shapes ([n_steps], [n_steps, batch, 3])
    """
    t_eval = torch.linspace(t_span[0], t_span[1], n_steps, dtype=torch.float32, device=device)

    # Use regular odeint - masking is handled in ODE
    y_solution = odeint(
        ode_system,
        y0_batch,
        t_eval,
        atol=atol,
        rtol=rtol,
        method=method,
    )

    return t_eval, y_solution


def extract_features_batch(
    t_eval: torch.Tensor,
    y_solution: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Extract features from batch of solutions for classification.

    Replicates LorenzFeatureExtractor logic:
    - If unbounded (max(|y|) > threshold): return [0, 0]
    - Otherwise: classify by mean x-coordinate in tail
      - mean(x) > 0: butterfly1 [1, 0]
      - mean(x) < 0: butterfly2 [0, 1]

    Args:
        t_eval: time evaluation points [n_steps]
        y_solution: full solution [n_steps, batch, 3]
        y0_batch: initial conditions [batch, 3] (unused but kept for API compatibility)
        ode_system: LorenzODE instance
        t_span_end: expected end time (for tail computation)
        device: device to run on

    Returns:
        feature vectors [batch, 2] as one-hot encodings
    """
    batch_size = y_solution.shape[1]
    features = torch.zeros((batch_size, 2), device=device)

    # Check which trajectories are unbounded by looking at max values
    max_vals = torch.max(torch.abs(y_solution), dim=2)[0]  # [n_steps, batch]
    max_val_per_traj = torch.max(max_vals, dim=0)[0]  # [batch]
    unbounded_mask = max_val_per_traj >= 200.0  # LorenzODE uses 200 as threshold

    # For bounded solutions, classify by mean x-coordinate in tail
    bounded_mask = ~unbounded_mask
    n_bounded = bounded_mask.sum().item()

    if n_bounded > 0:
        # Extract tail: last 10% of time points
        n_tail = max(1, len(t_eval) // 10)
        y_tail = y_solution[-n_tail:, bounded_mask, :]  # [n_tail, n_bounded, 3]

        # Mean of x-coordinate (first component) across time
        x_mean = torch.mean(y_tail[:, :, 0], dim=0)  # [n_bounded]

        # Classify based on sign of x_mean
        # butterfly1: x_mean > 0 -> [1, 0]
        # butterfly2: x_mean < 0 -> [0, 1]
        bounded_features = torch.zeros((n_bounded, 2), device=device)
        bounded_features[x_mean > 0, 0] = 1.0  # butterfly1
        bounded_features[x_mean <= 0, 1] = 1.0  # butterfly2

        # Assign features to bounded trajectories
        features[bounded_mask] = bounded_features

    # Unbounded trajectories get [0, 0] (already initialized)

    return features


def generate_uniform_samples(
    n_samples: int, min_limits: list[float], max_limits: list[float], device: str = "cpu"
) -> torch.Tensor:
    """
    Generate uniform random samples in the specified domain.

    Matches MATLAB setup:
    - minLimits = [-10, -20, 0]
    - maxLimits = [10, 20, 0]

    Args:
        n_samples: number of samples
        min_limits: minimum bounds [3]
        max_limits: maximum bounds [3]
        device: device to create tensors on

    Returns:
        tensor of initial conditions [n_samples, 3]
    """
    min_tensor = torch.tensor(min_limits, device=device)
    max_tensor = torch.tensor(max_limits, device=device)

    # Generate uniform samples
    samples = torch.rand(n_samples, 3, device=device)
    samples = samples * (max_tensor - min_tensor) + min_tensor

    return samples


def compute_template_features(
    template_ics: torch.Tensor, ode_system: LorenzODE, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute template features from known initial conditions.

    Matches MATLAB templates:
    - butterfly1: [0.8, -3.0, 0.0] -> expect [1, 0]
    - butterfly2: [-0.8, 3.0, 0.0] -> expect [0, 1]
    - unbounded: [10.0, 50.0, 0.0] -> expect [0, 0]

    Args:
        template_ics: tensor of template initial conditions [3, 3]
        ode_system: LorenzODE instance
        device: device to run on

    Returns:
        tensor of template features [3, 2]
    """
    t_eval, y_solution = solve_batch_trajectories(template_ics, ode_system, device=device)
    features = extract_features_batch(t_eval, y_solution, device=device)
    return features


def main():
    """Main experiment: compute Lorenz basin stability with torchdiffeq."""
    print("=" * 70)
    print("TorchDiffEq Lorenz Basin Stability Computation with Masking")
    print("=" * 70)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Parameters matching MATLAB setup
    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}
    ode_system = LorenzODE(params)
    n_samples = 100

    # Domain for sampling (matches MATLAB roi.minLimits/maxLimits)
    min_limits = [-10.0, -20.0, 0.0]
    max_limits = [10.0, 20.0, 0.0]

    # Template initial conditions (matches MATLAB setup_lorenz.m)
    template_ics = torch.tensor(
        [
            [0.8, -3.0, 0.0],  # butterfly1
            [-0.8, 3.0, 0.0],  # butterfly2
            [10.0, 50.0, 0.0],  # unbounded
        ],
        device=device,
    )
    labels = ["butterfly1", "butterfly2", "unbounded"]

    print(f"\nSystem parameters: sigma={params['sigma']}, r={params['r']}, b={params['b']}")
    print(f"Number of samples: {n_samples}")
    print(
        f"Sampling domain: x∈{min_limits[0], max_limits[0]}, "
        f"y∈{min_limits[1], max_limits[1]}, z∈{min_limits[2], max_limits[2]}"
    )

    # Compute template features
    print("\nComputing template features...")
    templates = compute_template_features(template_ics, ode_system, device=device)
    print("Template features:")
    for label, template_ic, template_feat in zip(labels, template_ics, templates, strict=True):
        print(
            f"  {label:<15} : IC={template_ic.cpu().numpy()} -> features={template_feat.cpu().numpy()}"
        )

    # Generate random initial conditions
    print(f"\nGenerating {n_samples} random initial conditions...")
    torch.manual_seed(42)
    initial_conditions = generate_uniform_samples(n_samples, min_limits, max_limits, device=device)

    # Compute basin stability
    print("\nComputing basin stability...")
    start_time = time.time()

    # Process all trajectories at once on GPU
    t_eval, y_solution = solve_batch_trajectories(initial_conditions, ode_system, device=device)

    print(f"  Integration complete in {time.time() - start_time:.1f}s")

    # Extract features for all trajectories
    features = extract_features_batch(t_eval, y_solution, device=device)

    # Find closest template for each trajectory
    distances = torch.cdist(features, templates)
    label_indices = torch.argmin(distances, dim=1)

    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.1f}s ({n_samples / total_time:.1f} traj/s)")

    # Compute statistics
    results = []
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for i, label in enumerate(labels):
        count = torch.sum(label_indices == i).item()
        basin_stability = float(count) / n_samples
        std_error = (basin_stability * (1 - basin_stability) / n_samples) ** 0.5

        results.append({
            "label": label,
            "basinStability": float(basin_stability) * 100,
            "absNumMembers": int(count),
            "standardError": float(std_error) * 100,
        })

        print(f"\n{label}:")
        print(f"  Basin stability: {basin_stability * 100:.3f}%")
        print(f"  Absolute members: {int(count)}/{n_samples}")
        print(f"  Standard error: {std_error * 100:.4f}%")

    # Add NaN entry for compatibility with expected format
    results.append({
        "label": "NaN",
        "basinStability": 0.0,
        "absNumMembers": 0,
        "standardError": 0.0,
    })

    # Save results
    output_file = "experiments/torchdiffeq_lorenz_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")

    # Compare with expected results
    print("\n" + "=" * 70)
    print("COMPARISON WITH EXPECTED RESULTS")
    print("=" * 70)

    expected_file = "tests/integration/lorenz/main_lorenz.json"
    try:
        with open(expected_file) as f:
            expected = json.load(f)

        print(f"\nComparing with {expected_file}:")
        print(f"{'Label':<15} {'Expected BS%':<15} {'Computed BS%':<15} {'Difference%':<15}")
        print("-" * 60)

        for exp_result in expected:
            label = exp_result["label"]
            if label == "NaN":
                continue

            computed_result = next((r for r in results if r["label"] == label), None)
            if computed_result:
                exp_bs = exp_result["basinStability"] * 100  # Convert to percentage
                comp_bs = computed_result["basinStability"]  # Already in percentage
                diff = abs(comp_bs - exp_bs)
                print(f"{label:<15} {exp_bs:<15.3f} {comp_bs:<15.3f} {diff:<15.3f}")

        print("\nNote: Some difference is expected due to randomness in sampling.")
        print("Basin stability values should be within ~1-2% of expected values.")

    except FileNotFoundError:
        print(f"\nExpected results file not found: {expected_file}")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
