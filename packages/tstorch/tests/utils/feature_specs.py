"""Benchmark feature specifications grouped by calculator domain."""

from __future__ import annotations

from .benchmark_utils import FeatureSpec

# TorchFeatureExtractor only supports the calculators listed in
# `SUPPORTED_FEATURES` (packages/tstorch/src/tstorch/torch_feature_extractor.py). Every spec below
# for a feature outside that set is marked `supports_batched_extractor=False` so the extractor-path
# benchmark skips it cleanly instead of failing on the extractor's fail-fast "unsupported feature"
# error - direct-calculator and tsfresh benchmarks for these features are unaffected. Flip a spec
# back to the default (True) as its feature is migrated.

DOMAIN_FEATURES: dict[str, list[FeatureSpec]] = {
    "statistical": [
        FeatureSpec("sum_values", "statistical", tsfresh_name="sum_values"),
        FeatureSpec("median", "statistical"),
        FeatureSpec("mean", "statistical"),
        FeatureSpec("length", "statistical"),
        FeatureSpec("standard_deviation", "statistical"),
        FeatureSpec("variance", "statistical"),
        FeatureSpec("root_mean_square", "statistical"),
        FeatureSpec("maximum", "statistical"),
        FeatureSpec("absolute_maximum", "statistical"),
        FeatureSpec("minimum", "statistical"),
        FeatureSpec("delta", "statistical", supports_tsfresh=False),
        FeatureSpec("log_delta", "statistical", supports_tsfresh=False),
        FeatureSpec("abs_energy", "statistical"),
        FeatureSpec("kurtosis", "statistical"),
        FeatureSpec("skewness", "statistical"),
        FeatureSpec(
            "quantile",
            "statistical",
            kwargs={"q": 0.5},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                'parameter_list for feature_calculators.quantile (fctype="simple"); '
                "feature_calculators.quantile calls np.quantile(x, q) independently per q."
            ),
        ),
        FeatureSpec("variation_coefficient", "statistical"),
        FeatureSpec(
            "mean_n_absolute_max",
            "statistical",
            kwargs={"number_of_maxima": 3},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.mean_n_absolute_max "
                '(fctype="simple"); feature_calculators.mean_n_absolute_max independently '
                "re-sorts the absolute values per number_of_maxima."
            ),
        ),
        FeatureSpec(
            "ratio_beyond_r_sigma",
            "statistical",
            kwargs={"r": 1.0},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.ratio_beyond_r_sigma "
                '(fctype="simple"); feature_calculators.ratio_beyond_r_sigma independently '
                "recomputes mean/std per r."
            ),
        ),
        FeatureSpec(
            "symmetry_looking",
            "statistical",
            kwargs={"r": 0.1},
            tsfresh_param=[{"r": 0.1}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.symmetry_looking is a combiner; its body computes "
                "mean_median_difference and max_min_difference once, then compares each "
                "requested r's threshold against that shared pair."
            ),
        ),
        FeatureSpec("amplitude", "statistical", supports_tsfresh=False),
    ],
    "change": [
        FeatureSpec("absolute_sum_of_changes", "change"),
        FeatureSpec("mean_abs_change", "change"),
        FeatureSpec("mean_change", "change"),
        FeatureSpec("mean_second_derivative_central", "change"),
        FeatureSpec(
            "change_quantiles",
            "change",
            kwargs={"ql": 0.2, "qh": 0.8, "isabs": False, "f_agg": "mean"},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.change_quantiles "
                '(fctype="simple"); feature_calculators.change_quantiles independently '
                "recomputes qcut and first differences per argument."
            ),
        ),
    ],
    "count": [
        FeatureSpec(
            "count_above",
            "count",
            kwargs={"t": 0.0},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.count_above "
                '(fctype="simple"); feature_calculators.count_above performs an '
                "independent threshold comparison and reduction per argument."
            ),
        ),
        FeatureSpec("count_above_mean", "count"),
        FeatureSpec(
            "count_below",
            "count",
            kwargs={"t": 0.0},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.count_below "
                '(fctype="simple"); feature_calculators.count_below performs an '
                "independent threshold comparison and reduction per argument."
            ),
        ),
        FeatureSpec("count_below_mean", "count"),
        FeatureSpec(
            "count_in_range",
            "count",
            kwargs={"min_val": -1.0, "max_val": 1.0},
            tsfresh_name="range_count",
            tsfresh_kwargs={"min": -1.0, "max": 1.0},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.range_count "
                '(fctype="simple"); feature_calculators.range_count performs two '
                "independent bound comparisons and a reduction per argument."
            ),
        ),
        FeatureSpec(
            "count_value",
            "count",
            kwargs={"value": 0.0},
            tsfresh_name="value_count",
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.value_count "
                '(fctype="simple"); feature_calculators.value_count performs an '
                "independent equality or NaN comparison per argument."
            ),
        ),
    ],
    "boolean": [
        FeatureSpec("has_duplicate", "boolean"),
        FeatureSpec("has_duplicate_max", "boolean"),
        FeatureSpec("has_duplicate_min", "boolean"),
        FeatureSpec(
            "has_variance_larger_than_standard_deviation",
            "boolean",
            tsfresh_name="variance_larger_than_standard_deviation",
        ),
        FeatureSpec(
            "large_standard_deviation",
            "boolean",
            kwargs={"r": 0.25},
            tsfresh_name="large_standard_deviation",
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops over "
                "parameter_list for feature_calculators.large_standard_deviation "
                '(fctype="simple"); feature_calculators.large_standard_deviation '
                "independently recomputes std/max/min per r."
            ),
        ),
    ],
    "location": [
        FeatureSpec("first_location_of_maximum", "location"),
        FeatureSpec("first_location_of_minimum", "location"),
        FeatureSpec("last_location_of_maximum", "location"),
        FeatureSpec("last_location_of_minimum", "location"),
        FeatureSpec(
            "index_mass_quantile",
            "location",
            kwargs={"q": 0.5},
            tsfresh_param=[{"q": 0.5}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.index_mass_quantile is a combiner; its body computes "
                "abs_x, total mass, and cumulative mass once, then finds the crossing for "
                "each requested q."
            ),
        ),
    ],
    "pattern": [
        FeatureSpec("longest_strike_above_mean", "pattern"),
        FeatureSpec("longest_strike_below_mean", "pattern"),
        FeatureSpec(
            "number_crossing_m",
            "pattern",
            kwargs={"m": 0.0},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk "
                "loops once per params dict because number_crossing_m is fctype='simple'"
            ),
        ),
        FeatureSpec(
            "number_peaks",
            "pattern",
            kwargs={"n": 3},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk "
                "loops once per params dict because number_peaks is fctype='simple'"
            ),
        ),
        FeatureSpec(
            "number_cwt_peaks",
            "pattern",
            kwargs={"n": 5},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk "
                "loops once per params dict because number_cwt_peaks is fctype='simple'"
            ),
        ),
    ],
    "autocorrelation": [
        FeatureSpec(
            "autocorrelation",
            "autocorrelation",
            kwargs={"lag": 1},
            # No tsfresh_param: unlike agg_autocorrelation below, tsfresh's own
            # `autocorrelation` is fctype="simple" (one real call per lag, no `param=list`
            # convention to mirror) - leaving tsfresh_param unset means execute_tsfresh_direct
            # calls tsfresh with plain `lag=1` (its real calling convention) while
            # execute_tstorch_direct/extractor fall back to `[kwargs]` = `[{"lag": 1}]` for
            # the tstorch side (Autocorrelation(params)(x), same shape every migrated
            # Feature class takes).
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk loops "
                "over parameter_list for feature_calculators.autocorrelation "
                '(fctype="simple"); feature_calculators.autocorrelation recomputes '
                "mean/variance inside each per-lag call."
            ),
        ),
        FeatureSpec(
            "partial_autocorrelation",
            "autocorrelation",
            kwargs={"lag": 1},
            tsfresh_param=[{"lag": 1}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.partial_autocorrelation is a combiner; its body "
                "computes max_demanded_lag once, calls statsmodels pacf once up to that "
                "lag, then indexes pacf_coeffs for each requested lag."
            ),
        ),
        FeatureSpec(
            "agg_autocorrelation",
            "autocorrelation",
            kwargs={"maxlag": 10, "f_agg": "mean"},
            tsfresh_param=[{"maxlag": 10, "f_agg": "mean"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.agg_autocorrelation is a combiner; its body "
                "computes max_maxlag once, calls acf once up to that lag, then slices the "
                "shared ACF vector for each requested (maxlag, f_agg)."
            ),
        ),
        FeatureSpec(
            "autocorrelation_periodicity",
            "autocorrelation",
            supports_tsfresh=False,
        ),
    ],
    "entropy_complexity": [
        FeatureSpec(
            "permutation_entropy",
            "entropy_complexity",
            kwargs={"tau": 1, "dimension": 3},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "feature_calculators.permutation_entropy is a simple calculator; the "
                "tsfresh dispatcher invokes it independently for each parameter dict."
            ),
        ),
        FeatureSpec(
            "binned_entropy",
            "entropy_complexity",
            kwargs={"max_bins": 10},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "feature_calculators.binned_entropy is a simple calculator; the tsfresh "
                "dispatcher invokes it independently for each parameter dict."
            ),
        ),
        FeatureSpec(
            "fourier_entropy",
            "entropy_complexity",
            kwargs={"bins": 10},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "feature_calculators.fourier_entropy is a simple calculator; the tsfresh "
                "dispatcher invokes it independently for each parameter dict."
            ),
        ),
        FeatureSpec(
            "lempel_ziv_complexity",
            "entropy_complexity",
            kwargs={"bins": 2},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "feature_calculators.lempel_ziv_complexity is a simple calculator; the "
                "tsfresh dispatcher invokes it independently for each parameter dict."
            ),
        ),
        FeatureSpec(
            "cid_ce",
            "entropy_complexity",
            kwargs={"normalize": True},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "feature_calculators.cid_ce is a simple calculator; the tsfresh dispatcher "
                "invokes it independently for each parameter dict."
            ),
        ),
        FeatureSpec(
            "approximate_entropy",
            "entropy_complexity",
            kwargs={"m": 2, "r": 0.3},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "feature_calculators.approximate_entropy is a simple calculator; the "
                "tsfresh dispatcher invokes it independently for each parameter dict."
            ),
        ),
        FeatureSpec(
            "sample_entropy",
            "entropy_complexity",
        ),
    ],
    "frequency": [
        FeatureSpec(
            "fft_coefficient",
            "frequency",
            kwargs={"coeff": 1, "attr": "abs"},
            tsfresh_param=[{"coeff": 1, "attr": "abs"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.fft_coefficient is a combiner; its body computes "
                "np.fft.rfft once, then selects each requested coefficient and attribute."
            ),
        ),
        FeatureSpec(
            "fft_aggregated",
            "frequency",
            kwargs={"aggtype": "centroid"},
            tsfresh_param=[{"aggtype": "centroid"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.fft_aggregated is a combiner; its body computes the "
                "absolute FFT once, then evaluates requested moment statistics from it."
            ),
        ),
        FeatureSpec(
            "spkt_welch_density",
            "frequency",
            kwargs={"coeff": 2},
            tsfresh_param=[{"coeff": 2}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.spkt_welch_density is a combiner; its body computes "
                "scipy.signal.welch once, then selects all requested coefficients."
            ),
        ),
        FeatureSpec(
            "cwt_coefficients",
            "frequency",
            kwargs={"widths": (2,), "coeff": 1, "w": 2},
            tsfresh_param=[{"widths": (2,), "coeff": 1, "w": 2}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.cwt_coefficients is a combiner; its body caches one "
                "pywt.cwt result per distinct widths tuple and selects requested columns."
            ),
        ),
        FeatureSpec(
            "spectral_frequency_ratio",
            "frequency",
            supports_tsfresh=False,
        ),
    ],
    "trend": [
        FeatureSpec(
            "linear_trend",
            "trend",
            kwargs={"attr": "slope"},
            tsfresh_param=[{"attr": "slope"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.linear_trend is a combiner; its body calls linregress "
                "once, then selects every requested result attribute."
            ),
        ),
        FeatureSpec(
            "linear_trend_timewise",
            "trend",
            kwargs={"attr": "slope"},
            tsfresh_param=[{"attr": "slope"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.linear_trend_timewise is a combiner; its body calls "
                "linregress once, then selects every requested result attribute."
            ),
        ),
        FeatureSpec(
            "agg_linear_trend",
            "trend",
            kwargs={"attr": "slope", "chunk_size": 5, "f_agg": "mean"},
            tsfresh_param=[{"attr": "slope", "chunk_len": 5, "f_agg": "mean"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.agg_linear_trend is a combiner; its body caches one "
                "chunk aggregation and linregress result per (f_agg, chunk_len) pair."
            ),
        ),
        FeatureSpec(
            "ar_coefficient",
            "trend",
            kwargs={"k": 3, "coeff": 1},
            tsfresh_param=[{"k": 3, "coeff": 1}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.ar_coefficient is a combiner; its body fits one AutoReg "
                "model per distinct k and selects all requested coefficients."
            ),
        ),
        FeatureSpec(
            "augmented_dickey_fuller",
            "trend",
            kwargs={"attr": "teststat"},
            tsfresh_param=[{"attr": "teststat"}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "feature_calculators.augmented_dickey_fuller is a combiner; its cached "
                "compute_adf call runs once per distinct autolag value."
            ),
        ),
    ],
    "reocurrance": [
        FeatureSpec(
            "percentage_of_reoccurring_datapoints_to_all_datapoints",
            "reocurrance",
        ),
        FeatureSpec(
            "percentage_of_reoccurring_values_to_all_values",
            "reocurrance",
        ),
        FeatureSpec("sum_of_reoccurring_data_points", "reocurrance"),
        FeatureSpec("sum_of_reoccurring_values", "reocurrance"),
        FeatureSpec(
            "ratio_value_number_to_time_series_length",
            "reocurrance",
        ),
    ],
    "advanced": [
        FeatureSpec("benford_correlation", "advanced"),
        FeatureSpec(
            "c3",
            "advanced",
            kwargs={"lag": 1},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk: "
                "simple calculators are called once per parameter dict"
            ),
        ),
        FeatureSpec(
            "energy_ratio_by_chunks",
            "advanced",
            kwargs={"num_segments": 10, "segment_focus": 0},
            tsfresh_param=[{"num_segments": 10, "segment_focus": 0}],
            tsfresh_argument_batching="calculator_gets_param_list_and_shares_work",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.feature_calculators.energy_ratio_by_chunks: "
                "the combiner computes full_series_energy once, then loops over params"
            ),
        ),
        FeatureSpec(
            "time_reversal_asymmetry_statistic",
            "advanced",
            kwargs={"lag": 1},
            tsfresh_argument_batching="proxy_loops_per_argument",
            tsfresh_argument_batching_source=(
                "tsfresh.feature_extraction.extraction._do_extraction_on_chunk: "
                "simple calculators are called once per parameter dict"
            ),
        ),
    ],
    "dynamical": [
        FeatureSpec(
            "lyapunov_r",
            "dynamical",
            supports_tsfresh=False,
            supports_batched_extractor=False,
        ),
        FeatureSpec(
            "lyapunov_e",
            "dynamical",
            kwargs={"emb_dim": 3, "matrix_dim": 2, "min_nb": 2},
            supports_tsfresh=False,
            supports_batched_extractor=False,
        ),
        FeatureSpec(
            "correlation_dimension",
            "dynamical",
            kwargs={"emb_dim": 2, "lag": 1, "n_rvals": 8},
            supports_tsfresh=False,
            supports_batched_extractor=False,
        ),
        FeatureSpec(
            "friedrich_coefficients",
            "dynamical",
            kwargs={"m": 3, "r": 30.0, "coeff": 1},
            tsfresh_param=[{"m": 3, "r": 30.0, "coeff": 1}],
            supports_batched_extractor=False,
        ),
        FeatureSpec(
            "max_langevin_fixed_point",
            "dynamical",
            kwargs={"m": 3, "r": 30.0},
            supports_batched_extractor=False,
        ),
    ],
}
