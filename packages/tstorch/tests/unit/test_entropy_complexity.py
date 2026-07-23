"""Accuracy, naming, compile, and export tests for entropy/complexity features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pytest
import torch
from torch import Tensor
from tstorch._ops.lempel_ziv import lempel_ziv_phrase_count
from tstorch.calculators.torch_features_entropy_complexity import (
    ApproximateEntropy,
    ApproximateEntropyParams,
    BinnedEntropy,
    BinnedEntropyParams,
    CidCe,
    CidCeParams,
    FourierEntropy,
    FourierEntropyParams,
    LempelZivComplexity,
    LempelZivComplexityParams,
    PermutationEntropy,
    PermutationEntropyParams,
    SampleEntropy,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)


class _Opcheck(Protocol):
    def __call__(
        self,
        op: object,
        args: tuple[Tensor, ...],
        kwargs: dict[str, object] | None = None,
        *,
        test_utils: str | Sequence[str] = ...,
        raise_exception: bool = True,
    ) -> dict[str, str]: ...


_opcheck: _Opcheck = vars(torch.library)["opcheck"]

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["entropy_complexity"]}

PERMUTATION_ENTROPY_PARAMS: list[PermutationEntropyParams] = [
    {"tau": 1, "dimension": 2},
    {"tau": 1, "dimension": 3},
    {"tau": 1, "dimension": 4},
    {"tau": 2, "dimension": 2},
    {"tau": 2, "dimension": 3},
    {"tau": 2, "dimension": 4},
    {"tau": 3, "dimension": 2},
    {"tau": 3, "dimension": 3},
]
BINNED_ENTROPY_PARAMS: list[BinnedEntropyParams] = [
    {"max_bins": bins} for bins in [2, 3, 5, 10, 16, 32, 64, 100]
]
FOURIER_ENTROPY_PARAMS: list[FourierEntropyParams] = [
    {"bins": bins} for bins in [2, 3, 4, 5, 8, 10, 32, 100]
]
LEMPEL_ZIV_COMPLEXITY_PARAMS: list[LempelZivComplexityParams] = [
    {"bins": bins} for bins in [1, 2, 3, 4, 5, 8, 16, 32]
]
CID_CE_PARAMS: list[CidCeParams] = [{"normalize": True}, {"normalize": False}]
APPROXIMATE_ENTROPY_PARAMS: list[ApproximateEntropyParams] = [
    {"m": 1, "r": 0.1},
    {"m": 2, "r": 0.1},
    {"m": 3, "r": 0.1},
    {"m": 2, "r": 0.0},
    {"m": 2, "r": 0.3},
    {"m": 2, "r": 0.5},
    {"m": 2, "r": 0.9},
    {"m": 2, "r": 1.2},
]


def _reference_lempel_ziv_phrase_count(symbols: Sequence[int]) -> int:
    phrases: set[tuple[int, ...]] = set()
    position = 0
    candidate_length = 1
    while position + candidate_length <= len(symbols):
        phrase = tuple(symbols[position : position + candidate_length])
        if phrase in phrases:
            candidate_length += 1
        else:
            phrases.add(phrase)
            position += candidate_length
            candidate_length = 1
    return len(phrases)


def test_permutation_entropy_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        PermutationEntropy,
        FEATURES["permutation_entropy"],
        canonical_trajectories,
        PERMUTATION_ENTROPY_PARAMS,
    )


def test_permutation_entropy_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "permutation_entropy",
        PermutationEntropy,
        PERMUTATION_ENTROPY_PARAMS[:2],
        y_bst,
    )


def test_permutation_entropy_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "permutation_entropy",
        PermutationEntropy,
        PERMUTATION_ENTROPY_PARAMS[:2],
        y_bst,
    )


def test_permutation_entropy_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "permutation_entropy", PermutationEntropy, PERMUTATION_ENTROPY_PARAMS
    )


def test_binned_entropy_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        BinnedEntropy,
        FEATURES["binned_entropy"],
        canonical_trajectories,
        BINNED_ENTROPY_PARAMS,
    )


def test_binned_entropy_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "binned_entropy", BinnedEntropy, BINNED_ENTROPY_PARAMS[:2], y_bst
    )


def test_binned_entropy_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "binned_entropy", BinnedEntropy, BINNED_ENTROPY_PARAMS[:2], y_bst
    )


def test_binned_entropy_has_valid_names() -> None:
    assert_feature_names_are_valid("binned_entropy", BinnedEntropy, BINNED_ENTROPY_PARAMS)


def test_fourier_entropy_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        FourierEntropy,
        FEATURES["fourier_entropy"],
        canonical_trajectories,
        FOURIER_ENTROPY_PARAMS,
    )


def test_fourier_entropy_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "fourier_entropy", FourierEntropy, FOURIER_ENTROPY_PARAMS[:2], y_bst
    )


def test_fourier_entropy_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "fourier_entropy", FourierEntropy, FOURIER_ENTROPY_PARAMS[:2], y_bst
    )


def test_fourier_entropy_has_valid_names() -> None:
    assert_feature_names_are_valid("fourier_entropy", FourierEntropy, FOURIER_ENTROPY_PARAMS)


def test_lempel_ziv_complexity_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        LempelZivComplexity,
        FEATURES["lempel_ziv_complexity"],
        canonical_trajectories,
        LEMPEL_ZIV_COMPLEXITY_PARAMS,
    )


def test_lempel_ziv_complexity_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "lempel_ziv_complexity",
        LempelZivComplexity,
        LEMPEL_ZIV_COMPLEXITY_PARAMS[:2],
        y_bst,
    )


def test_lempel_ziv_complexity_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "lempel_ziv_complexity",
        LempelZivComplexity,
        LEMPEL_ZIV_COMPLEXITY_PARAMS[:2],
        y_bst,
    )


def test_lempel_ziv_complexity_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "lempel_ziv_complexity",
        LempelZivComplexity,
        LEMPEL_ZIV_COMPLEXITY_PARAMS,
    )


def test_lempel_ziv_complexity_matches_reference_parse() -> None:
    symbols = torch.tensor(
        [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        dtype=torch.float32,
    ).view(1, 1, -1)
    result = LempelZivComplexity([{"bins": 2}])(symbols)
    torch.testing.assert_close(result, torch.tensor([[[8 / 16]]]))


@pytest.mark.parametrize(
    ("rows", "sequence_length", "alphabet_size"),
    [(1, 1, 2), (4, 7, 3), (8, 31, 5), (3, 128, 16)],
)
def test_lempel_ziv_complexity_operator_matches_reference(
    rows: int,
    sequence_length: int,
    alphabet_size: int,
) -> None:
    generator = torch.Generator().manual_seed(rows * 10_000 + sequence_length * 100 + alphabet_size)
    symbols = torch.randint(
        alphabet_size,
        (rows, sequence_length),
        dtype=torch.int64,
        generator=generator,
    )
    expected = torch.tensor(
        [_reference_lempel_ziv_phrase_count([int(symbol) for symbol in row]) for row in symbols],
        dtype=torch.int64,
    )
    torch.testing.assert_close(lempel_ziv_phrase_count(symbols), expected)
    if torch.cuda.is_available():
        actual = lempel_ziv_phrase_count(symbols.cuda()).cpu()
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lempel_ziv_complexity_matches_reference_parse_on_cuda() -> None:
    symbols = torch.tensor(
        [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        dtype=torch.float32,
        device="cuda",
    ).view(1, 1, -1)
    result = LempelZivComplexity([{"bins": 2}])(symbols)
    torch.testing.assert_close(result.cpu(), torch.tensor([[[8 / 16]]]))


def test_lempel_ziv_complexity_operator_registration() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        symbols = torch.tensor(
            [[1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0]],
            dtype=torch.int64,
            device=device,
        )
        _opcheck(lempel_ziv_phrase_count, (symbols,))


def test_cid_ce_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(CidCe, FEATURES["cid_ce"], canonical_trajectories, CID_CE_PARAMS)


def test_cid_ce_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("cid_ce", CidCe, CID_CE_PARAMS, y_bst)


def test_cid_ce_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("cid_ce", CidCe, CID_CE_PARAMS, y_bst)


def test_cid_ce_has_valid_names() -> None:
    assert_feature_names_are_valid("cid_ce", CidCe, CID_CE_PARAMS)


def test_approximate_entropy_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        ApproximateEntropy,
        FEATURES["approximate_entropy"],
        canonical_trajectories,
        APPROXIMATE_ENTROPY_PARAMS,
    )


def test_approximate_entropy_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "approximate_entropy",
        ApproximateEntropy,
        APPROXIMATE_ENTROPY_PARAMS[:2],
        y_bst,
    )


def test_approximate_entropy_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "approximate_entropy",
        ApproximateEntropy,
        APPROXIMATE_ENTROPY_PARAMS[:2],
        y_bst,
    )


def test_approximate_entropy_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "approximate_entropy", ApproximateEntropy, APPROXIMATE_ENTROPY_PARAMS
    )


def test_sample_entropy_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        SampleEntropy, FEATURES["sample_entropy"], canonical_trajectories
    )


def test_sample_entropy_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("sample_entropy", SampleEntropy, None, y_bst)


def test_sample_entropy_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("sample_entropy", SampleEntropy, None, y_bst)


def test_sample_entropy_has_valid_names() -> None:
    assert_feature_names_are_valid("sample_entropy", SampleEntropy)
