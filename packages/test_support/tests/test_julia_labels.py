import numpy as np
from test_support import julia_labels_to_pybasin


def test_shifts_labels_to_zero_based_and_keeps_out_of_range_sentinel() -> None:
    labels = julia_labels_to_pybasin(np.array([1, 2, -1, 3]))

    np.testing.assert_array_equal(labels, [0, 1, -1, 2])
