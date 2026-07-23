# pyright: basic

"""
Pure JAX + Diffrax implementation of Lorenz basin stability computation.

Replicates the MATLAB implementation with event-based stopping for unbounded trajectories.
Uses Diffrax Event API to detect when max(|y|) > 200 (similar to lorenzStopFcn.m).

Expected results should match main_lorenz.json:
- butterfly1: ~8.9% basin stability
- butterfly2: ~8.7% basin stability
- unbounded: ~82.3% basin stability
"""

import json
import time
from typing import NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp


class LorenzParams(NamedTuple):
    """Parameters for the broken butterfly Lorenz system."""

    sigma: float = 0.12
    r: float = 0.0
    b: float = -0.6


def lorenz_vector_field(t, y, args):
    """
    Lorenz system vector field (broken butterfly parameterization).

    Corresponds to ode_lorenz.m:
    dydt = [sigma*(y(2)-y(1));
            r*y(1)-y(1)*y(3)-y(2);
            y(1)*y(2) - b*y(3)];

    Args:
        t: time (unused but required by Diffrax)
        y: state vector [x, y, z]
        args: LorenzParams tuple

    Returns:
        derivative vector dydt
    """
    params = args
    x, y_coord, z = y

    dx = params.sigma * (y_coord - x)
    dy = params.r * x - x * z - y_coord
    dz = x * y_coord - params.b * z

    return jnp.array([dx, dy, dz])


def lorenz_stop_event(max_val: float = 200.0):
    """
    Create event function to stop integration when amplitude exceeds threshold.

    Replicates lorenzStopFcn.m:
    value = (abs(max(y)) - maxval);
    isterminal = 1;  % halts the integration

    Args:
        max_val: threshold for stopping (default 200.0)

    Returns:
        Event condition function for use with diffrax.Event
    """

    def cond_fn(t, y, args, **kwargs):
        """Returns positive when under threshold, negative when over (triggers stop)."""
        # Match MATLAB: abs(max(y)) - maxval
        # This computes the max of all components, then takes abs
        max_y = jnp.max(y)
        max_abs_y = jnp.abs(max_y)
        # Return value crosses zero when max_abs_y crosses max_val
        # Negative return -> event triggered
        return max_val - max_abs_y

    return cond_fn


def solve_single_trajectory(y0, params, t_span=(0.0, 1000.0), dt0=0.04, n_steps=1000):
    """
    Solve single Lorenz trajectory with event-based stopping.

    Args:
        y0: initial condition [x, y, z]
        params: LorenzParams
        t_span: integration time span (default matches MATLAB: 0 to 1000)
        dt0: initial step size (fs=25 in MATLAB -> dt=0.04)
        n_steps: number of time points to save (default 1000)

    Returns:
        solution from diffrax.diffeqsolve
    """
    term = dfx.ODETerm(lorenz_vector_field)
    solver = dfx.Tsit5()  # Similar to MATLAB's ode45

    # Event to stop when amplitude > 200
    # No root_finder means it stops at the end of the step where condition triggers
    event = dfx.Event(cond_fn=lorenz_stop_event(max_val=200.0))

    # Stepsize controller matching MATLAB tolerances
    stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-6)

    # Save at discrete time points (like pendulum benchmark)
    # This is much more memory-efficient than dense=True
    # We save the final time and the full trajectory for post-processing
    t_eval = jnp.linspace(t_span[0], t_span[1], n_steps)
    saveat = dfx.SaveAt(t1=True, ts=t_eval)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt0,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=10000,
    )

    return sol


def extract_features_single(sol, t_steady=900.0, t_span_end=1000.0):
    """
    Extract features from a single solution for classification.

    Replicates LorenzFeatureExtractor logic:
    - If unbounded (event triggered): return [0, 0]
    - Otherwise: classify by mean x-coordinate in tail
      - mean(x) > 0: butterfly1 [1, 0]
      - mean(x) < 0: butterfly2 [0, 1]

    Args:
        sol: Diffrax solution object
        t_steady: time after which to consider steady state
        t_span_end: expected end time (for detecting early termination)

    Returns:
        feature vector [2] as one-hot encoding
    """
    final_state = sol.ys[-1]
    unbounded = jnp.any(jnp.isinf(final_state))

    # For bounded solutions, extract tail from the last 10% of REQUESTED time points
    # Since t_eval goes from 0 to 1000 with n_steps points, last 10% = last n_steps//10 points
    # These correspond to t > 900 for t_span=(0, 1000)
    n_saved = sol.ys.shape[0] - 1  # Exclude the t1=True point at the end
    n_tail = max(1, n_saved // 10)  # Last 10% of the trajectory points (excluding final t1)

    # Extract tail: use points from -n_tail-1 to -1 (excluding the final t1=True point)
    y_tail = sol.ys[-(n_tail + 1) : -1]  # shape: (n_tail, 3)

    # Mean of x-coordinate (first component)
    x_mean = jnp.mean(y_tail[:, 0])

    # Classify based on sign of x_mean
    # butterfly1: x_mean > 0 -> [1, 0]
    # butterfly2: x_mean < 0 -> [0, 1]
    is_butterfly1 = x_mean > 0

    # Use jnp.where for JAX-compatible branching
    # If unbounded: [0, 0], else classify by x_mean
    bounded_features = jnp.where(is_butterfly1, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    features = jnp.where(unbounded, jnp.array([0.0, 0.0]), bounded_features)

    return features


def generate_uniform_samples(n_samples, min_limits, max_limits, key):
    """
    Generate uniform random samples in the specified domain.

    Matches MATLAB setup:
    - minLimits = [-10, -20, 0]
    - maxLimits = [10, 20, 0]

    Args:
        n_samples: number of samples
        min_limits: minimum bounds [3]
        max_limits: maximum bounds [3]
        key: JAX random key

    Returns:
        array of initial conditions [n_samples, 3]
    """
    min_arr = jnp.array(min_limits)
    max_arr = jnp.array(max_limits)

    # Generate uniform samples
    samples = jax.random.uniform(key, shape=(n_samples, 3), minval=min_arr, maxval=max_arr)

    return samples


def compute_template_features(template_ics, params):
    """
    Compute template features from known initial conditions.

    Matches MATLAB templates:
    - butterfly1: [0.8, -3.0, 0.0] -> expect [1, 0]
    - butterfly2: [-0.8, 3.0, 0.0] -> expect [0, 1]
    - unbounded: [10.0, 50.0, 0.0] -> expect [0, 0]

    Args:
        template_ics: array of template initial conditions [3, 3]
        params: LorenzParams

    Returns:
        array of template features [3, 2]
    """
    templates = []
    for y0 in template_ics:
        sol = solve_single_trajectory(y0, params)
        features = extract_features_single(sol)
        templates.append(features)

    return jnp.stack(templates)


def main():
    """Main experiment: compute Lorenz basin stability with JAX + Diffrax."""
    print("=" * 70)
    print("JAX + Diffrax Lorenz Basin Stability Computation")
    print("=" * 70)

    # Parameters matching MATLAB setup
    params = LorenzParams(sigma=0.12, r=0.0, b=-0.6)
    n_samples = 20_000  # Matches MATLAB N=20000

    # Domain for sampling (matches MATLAB roi.minLimits/maxLimits)
    min_limits = [-10.0, -20.0, 0.0]
    max_limits = [10.0, 20.0, 0.0]

    # Template initial conditions (matches MATLAB setup_lorenz.m)
    template_ics = jnp.array([
        [0.8, -3.0, 0.0],  # butterfly1
        [-0.8, 3.0, 0.0],  # butterfly2
        [10.0, 50.0, 0.0],  # unbounded
    ])
    labels = ["butterfly1", "butterfly2", "unbounded"]

    print(f"\nSystem parameters: sigma={params.sigma}, r={params.r}, b={params.b}")
    print(f"Number of samples: {n_samples}")
    print(
        f"Sampling domain: x∈{min_limits[0], max_limits[0]}, "
        f"y∈{min_limits[1], max_limits[1]}, z∈{min_limits[2], max_limits[2]}"
    )

    # Compute template features
    print("\nComputing template features...")
    templates = compute_template_features(template_ics, params)
    print("Template features:")
    for label, template_ic, template_feat in zip(labels, template_ics, templates, strict=True):
        print(f"  {label:<15} : IC={template_ic} -> features={template_feat}")

    # Generate random initial conditions
    print(f"\nGenerating {n_samples} random initial conditions...")
    key = jax.random.PRNGKey(42)
    initial_conditions = generate_uniform_samples(n_samples, min_limits, max_limits, key)

    # Compute basin stability
    print("\nComputing basin stability (this may take a while)...")
    print("Processing trajectories with event-based stopping (GPU-accelerated with vmap)...")

    # Process function
    def process_trajectory(y0):
        sol = solve_single_trajectory(y0, params)
        features = extract_features_single(sol)
        label_idx = jnp.argmin(jnp.linalg.norm(templates - features, axis=1))
        return label_idx

    # JIT compile the batched function for better performance
    print("Compiling batched solver (this may take a moment)...")
    batched_process = jax.jit(jax.vmap(process_trajectory))

    # Warm-up compilation with a small batch
    _ = batched_process(initial_conditions[:10]).block_until_ready()
    print("Compilation complete!")

    # Process ALL samples in a SINGLE batch to test memory efficiency
    print(f"\nProcessing ALL {n_samples} trajectories in ONE BATCH (fully vectorized)...")

    all_label_indices = []
    start_time = time.time()

    # Process everything at once - no chunking!
    all_labels = batched_process(initial_conditions).block_until_ready()
    all_label_indices.append(all_labels)

    total_time = time.time() - start_time
    print(
        f"  Complete! Processed all {n_samples} trajectories in {total_time:.1f}s ({n_samples / total_time:.1f} traj/s)"
    )

    # Combine results
    label_indices = jnp.concatenate(all_label_indices)
    print(f"  Complete! Processed all {n_samples} trajectories")

    # Compute statistics
    results = []
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for i, label in enumerate(labels):
        count = jnp.sum(label_indices == i)
        basin_stability = float(count) / n_samples
        std_error = jnp.sqrt(basin_stability * (1 - basin_stability) / n_samples)

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
    output_file = "experiments/jax_lorenz_results.json"
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
