"""``SUPPORTED_FEATURES``: the ``feature_name -> Feature class`` mapping.

Every extractor-eligible feature, parameter-free or parametrized, is registered here by
importing its class directly from the calculator file that defines it - no factory/adapter
layer in between. Both :class:`tstorch.torch_feature_extractor.TorchFeatureExtractor` and
:class:`tstorch.dask.DaskFeatureExtractor` resolve every feature name through this same
mapping, so requesting anything not listed here raises identically for both.
"""

from typing import Any

from tstorch.calculators.torch_features_advanced import (
    C3,
    BenfordCorrelation,
    EnergyRatioByChunks,
    TimeReversalAsymmetryStatistic,
)
from tstorch.calculators.torch_features_autocorrelation import (
    AggAutocorrelation,
    Autocorrelation,
    AutocorrelationPeriodicity,
    PartialAutocorrelation,
)
from tstorch.calculators.torch_features_boolean import (
    HasDuplicate,
    HasDuplicateMax,
    HasDuplicateMin,
    HasVarianceLargerThanStandardDeviation,
    LargeStandardDeviation,
)
from tstorch.calculators.torch_features_change import (
    AbsoluteSumOfChanges,
    ChangeQuantiles,
    MeanAbsChange,
    MeanChange,
    MeanSecondDerivativeCentral,
)
from tstorch.calculators.torch_features_count import (
    CountAbove,
    CountAboveMean,
    CountBelow,
    CountBelowMean,
    CountInRange,
    CountValue,
)
from tstorch.calculators.torch_features_entropy_complexity import (
    ApproximateEntropy,
    BinnedEntropy,
    CidCe,
    FourierEntropy,
    LempelZivComplexity,
    PermutationEntropy,
    SampleEntropy,
)
from tstorch.calculators.torch_features_frequency import (
    CwtCoefficients,
    FftAggregated,
    FftCoefficient,
    SpectralFrequencyRatio,
    SpktWelchDensity,
)
from tstorch.calculators.torch_features_location import (
    FirstLocationOfMaximum,
    FirstLocationOfMinimum,
    IndexMassQuantile,
    LastLocationOfMaximum,
    LastLocationOfMinimum,
)
from tstorch.calculators.torch_features_pattern import (
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    NumberCrossingM,
    NumberCwtPeaks,
    NumberPeaks,
)
from tstorch.calculators.torch_features_reocurrance import (
    PercentageOfReoccurringDatapointsToAllDatapoints,
    PercentageOfReoccurringValuesToAllValues,
    RatioValueNumberToTimeSeriesLength,
    SumOfReoccurringDataPoints,
    SumOfReoccurringValues,
)
from tstorch.calculators.torch_features_statistical import (
    AbsEnergy,
    AbsoluteMaximum,
    Amplitude,
    Delta,
    Kurtosis,
    Length,
    LogDelta,
    Maximum,
    Mean,
    MeanNAbsoluteMax,
    Median,
    Minimum,
    Quantile,
    RatioBeyondRSigma,
    RootMeanSquare,
    Skewness,
    StandardDeviation,
    SumValues,
    SymmetryLooking,
    Variance,
    VariationCoefficient,
)
from tstorch.calculators.torch_features_trend import (
    AggLinearTrend,
    ArCoefficient,
    AugmentedDickeyFuller,
    LinearTrend,
    LinearTrendTimewise,
)
from tstorch.feature_factory import Feature, ParameterlessFeature

SUPPORTED_FEATURES: dict[str, type[Feature[Any]] | type[ParameterlessFeature]] = {
    "benford_correlation": BenfordCorrelation,
    "c3": C3,
    "energy_ratio_by_chunks": EnergyRatioByChunks,
    "time_reversal_asymmetry_statistic": TimeReversalAsymmetryStatistic,
    "autocorrelation": Autocorrelation,
    "partial_autocorrelation": PartialAutocorrelation,
    "agg_autocorrelation": AggAutocorrelation,
    "autocorrelation_periodicity": AutocorrelationPeriodicity,
    "has_duplicate": HasDuplicate,
    "has_duplicate_max": HasDuplicateMax,
    "has_duplicate_min": HasDuplicateMin,
    "has_variance_larger_than_standard_deviation": HasVarianceLargerThanStandardDeviation,
    "large_standard_deviation": LargeStandardDeviation,
    "absolute_sum_of_changes": AbsoluteSumOfChanges,
    "mean_abs_change": MeanAbsChange,
    "mean_change": MeanChange,
    "mean_second_derivative_central": MeanSecondDerivativeCentral,
    "change_quantiles": ChangeQuantiles,
    "count_above": CountAbove,
    "count_above_mean": CountAboveMean,
    "count_below": CountBelow,
    "count_below_mean": CountBelowMean,
    "count_in_range": CountInRange,
    "count_value": CountValue,
    "permutation_entropy": PermutationEntropy,
    "binned_entropy": BinnedEntropy,
    "fourier_entropy": FourierEntropy,
    "lempel_ziv_complexity": LempelZivComplexity,
    "cid_ce": CidCe,
    "approximate_entropy": ApproximateEntropy,
    "sample_entropy": SampleEntropy,
    "fft_coefficient": FftCoefficient,
    "fft_aggregated": FftAggregated,
    "spkt_welch_density": SpktWelchDensity,
    "cwt_coefficients": CwtCoefficients,
    "spectral_frequency_ratio": SpectralFrequencyRatio,
    "first_location_of_maximum": FirstLocationOfMaximum,
    "first_location_of_minimum": FirstLocationOfMinimum,
    "last_location_of_maximum": LastLocationOfMaximum,
    "last_location_of_minimum": LastLocationOfMinimum,
    "index_mass_quantile": IndexMassQuantile,
    "longest_strike_above_mean": LongestStrikeAboveMean,
    "longest_strike_below_mean": LongestStrikeBelowMean,
    "number_crossing_m": NumberCrossingM,
    "number_cwt_peaks": NumberCwtPeaks,
    "number_peaks": NumberPeaks,
    "percentage_of_reoccurring_datapoints_to_all_datapoints": (
        PercentageOfReoccurringDatapointsToAllDatapoints
    ),
    "percentage_of_reoccurring_values_to_all_values": (PercentageOfReoccurringValuesToAllValues),
    "sum_of_reoccurring_data_points": SumOfReoccurringDataPoints,
    "sum_of_reoccurring_values": SumOfReoccurringValues,
    "ratio_value_number_to_time_series_length": RatioValueNumberToTimeSeriesLength,
    "sum_values": SumValues,
    "median": Median,
    "mean": Mean,
    "length": Length,
    "standard_deviation": StandardDeviation,
    "variance": Variance,
    "root_mean_square": RootMeanSquare,
    "maximum": Maximum,
    "absolute_maximum": AbsoluteMaximum,
    "minimum": Minimum,
    "delta": Delta,
    "log_delta": LogDelta,
    "abs_energy": AbsEnergy,
    "kurtosis": Kurtosis,
    "skewness": Skewness,
    "quantile": Quantile,
    "variation_coefficient": VariationCoefficient,
    "mean_n_absolute_max": MeanNAbsoluteMax,
    "ratio_beyond_r_sigma": RatioBeyondRSigma,
    "symmetry_looking": SymmetryLooking,
    "amplitude": Amplitude,
    "linear_trend": LinearTrend,
    "linear_trend_timewise": LinearTrendTimewise,
    "agg_linear_trend": AggLinearTrend,
    "ar_coefficient": ArCoefficient,
    "augmented_dickey_fuller": AugmentedDickeyFuller,
}
