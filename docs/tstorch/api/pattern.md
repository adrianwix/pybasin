# Pattern

{{ api_object_heading("Longest Strike Above Mean", "tstorch.calculators.torch_features_pattern.LongestStrikeAboveMean") }}

::: tstorch.calculators.torch_features_pattern.LongestStrikeAboveMean
    options:
      docstring_style: google
      show_root_heading: false
      heading_level: 3
      members:
        - names
        - __call__

{{ api_object_heading("Longest Strike Below Mean", "tstorch.calculators.torch_features_pattern.LongestStrikeBelowMean") }}

::: tstorch.calculators.torch_features_pattern.LongestStrikeBelowMean
    options:
      docstring_style: google
      show_root_heading: false
      heading_level: 3
      members:
        - names
        - __call__

{{ api_object_heading("Number Crossing M", "tstorch.calculators.torch_features_pattern.NumberCrossingM") }}

::: tstorch.calculators.torch_features_pattern.NumberCrossingM
    options:
      docstring_style: google
      show_root_heading: false
      heading_level: 3
      members:
        - names
        - __call__

{{ typed_dict_schema("tstorch.calculators.torch_features_pattern.NumberCrossingMParams") }}

{{ api_object_heading("Number CWT Peaks", "tstorch.calculators.torch_features_pattern.NumberCwtPeaks") }}

::: tstorch.calculators.torch_features_pattern.NumberCwtPeaks
    options:
      docstring_style: google
      show_root_heading: false
      heading_level: 3
      members:
        - names
        - __call__

{{ typed_dict_schema("tstorch.calculators.torch_features_pattern.NumberCwtPeaksParams") }}

### Numerical accuracy

The current [pattern benchmark results](../benchmarks/pattern.md#number_cwt_peaks) record
the absolute count differences for every tested `n` on the canonical trajectory suite.
This regression uses an absolute tolerance of three peaks and no relative tolerance. A
separate validation sweep compared 792 random and sinusoidal outputs across lengths from 21
through 1000 without a count mismatch.

Strict maxima remain sensitive to convolution roundoff. For a length-31 all-ones series and
`n=5`, tstorch returns zero while SciPy 1.17 returns one because two adjacent CWT coefficients
are equal in PyTorch but differ by one floating-point step in SciPy. The class documentation
states this boundary as part of the feature contract.

{{ api_object_heading("Number Peaks", "tstorch.calculators.torch_features_pattern.NumberPeaks") }}

::: tstorch.calculators.torch_features_pattern.NumberPeaks
    options:
      docstring_style: google
      show_root_heading: false
      heading_level: 3
      members:
        - names
        - __call__

{{ typed_dict_schema("tstorch.calculators.torch_features_pattern.NumberPeaksParams") }}
