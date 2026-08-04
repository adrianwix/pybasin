"""Tests for the `settings` presets consumed by `TorchFeatureExtractor`."""

from __future__ import annotations

import math

import torch
from tstorch.settings import DYNAMICAL_SYSTEM_FEATURES_CONFIG
from tstorch.torch_feature_extractor import TorchFeatureExtractor


def test_dynamical_system_fc_parameters_distinguishes_fixed_point_from_limit_cycle() -> None:
    """`DYNAMICAL_SYSTEM_FEATURES_CONFIG` should yield 7 features per state and produce
    values consistent with its stated purpose: separating a fixed point (near-zero
    variance/amplitude) from a limit cycle (oscillating, non-zero variance/amplitude).
    """
    batch_size, n_states, n_steps = 4, 2, 500
    t = torch.linspace(0, 20 * math.pi, n_steps)

    y = torch.empty(batch_size, n_states, n_steps)
    y[:, 0, :] = 1.0  # state 0: fixed point (constant trajectory)
    y[:, 1, :] = 2.0 * torch.sin(t)  # state 1: limit cycle (periodic oscillation)

    extractor = TorchFeatureExtractor(
        features=DYNAMICAL_SYSTEM_FEATURES_CONFIG,
        use_compile=False,
    )
    features = extractor.extract_features(y)
    names = extractor.feature_names

    assert features.shape == (batch_size, n_states * 7)
    assert len(names) == n_states * 7

    for state_idx in range(n_states):
        prefix = f"state_{state_idx}__"
        state_names = [name for name in names if name.startswith(prefix)]
        base_names = [name[len(prefix) :].split("__")[0] for name in state_names]
        assert sorted(base_names) == sorted(
            ["variance", "amplitude", "mean", "linear_trend", "spectral_frequency_ratio"]
            + ["autocorrelation_periodicity"] * 2
        )

    def value_of(state_idx: int, base_name: str) -> torch.Tensor:
        prefix = f"state_{state_idx}__{base_name}"
        (col,) = [i for i, name in enumerate(names) if name.startswith(prefix)]
        return features[:, col]

    fp_variance = value_of(0, "variance")
    lc_variance = value_of(1, "variance")
    fp_amplitude = value_of(0, "amplitude")
    lc_amplitude = value_of(1, "amplitude")

    assert torch.all(fp_variance < 1e-6)
    assert torch.all(lc_variance > fp_variance)
    assert torch.all(fp_amplitude < 1e-6)
    assert torch.all(lc_amplitude > 1.0)
