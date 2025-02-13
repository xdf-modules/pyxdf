import itertools

import numpy as np
import pytest
from pyxdf.pyxdf import _find_segment_indices


@pytest.mark.parametrize(
    "b_breaks, expected",
    [
        (np.array([]), [(0, 0)]),
        (np.array([False]), [(0, 1)]),
        (np.array([True]), [(0, 0), (1, 1)]),
        (np.array([False, False]), [(0, 2)]),
        (np.array([True, False]), [(0, 0), (1, 2)]),
        (np.array([False, True]), [(0, 1), (2, 2)]),
        (np.array([True, True]), [(0, 0), (1, 1), (2, 2)]),
    ],
)
def test_segments(b_breaks, expected):
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == expected


@pytest.mark.parametrize(
    "size, n_breaks, breaks, expected",
    [
        (
            size,
            n_breaks,
            breaks,
            list(zip(np.concat([[0], breaks + 1]), np.concat([breaks, [size]]))),
        )
        for size, n_breaks, breaks_list in [
            (
                size,
                n_breaks,
                list(itertools.combinations(range(0, size), n_breaks)),
            )
            # All sequences of length 3 to 5, continuing from the tests above.
            for size in range(3, 6)
            # All numbers of breaks between 0 and size
            for n_breaks in range(0, size + 1)
        ]
        for breaks in [np.array(breaks, dtype=int) for breaks in breaks_list]
    ],
)
def test_segments_exhaustive(size, n_breaks, breaks, expected):
    b_breaks = np.full(size, False)
    b_breaks[breaks] = True
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == expected
