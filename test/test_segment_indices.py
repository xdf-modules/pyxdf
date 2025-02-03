import itertools

import numpy as np
import pytest
from pyxdf.pyxdf import _find_segment_indices


def test_segments_size_0_nbreaks_0():
    b_breaks = np.array([])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 0)]


def test_segments_size_1_nbreaks_0():
    b_breaks = np.array([False])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 1)]


def test_segments_size_1_nbreaks_1():
    b_breaks = np.array([True])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 0), (1, 1)]


def test_segments_size_2_nbreaks_0():
    b_breaks = np.array([False, False])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 2)]


def test_segments_size_2_nbreaks_1a():
    b_breaks = np.array([True, False])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 0), (1, 2)]


def test_segments_size_2_nbreaks_1b():
    b_breaks = np.array([False, True])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 1), (2, 2)]


def test_segments_size_2_nbreaks_2():
    b_breaks = np.array([True, True])
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == [(0, 0), (1, 1), (2, 2)]


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
            # All sequences of length 0 to 5
            for size in range(0, 6)
            # All numbers of breaks between 0 and size
            for n_breaks in range(0, size + 1)
        ]
        for breaks in [np.array(breaks, dtype=int) for breaks in breaks_list]
    ],
)
def test_segments_all(size, n_breaks, breaks, expected):
    b_breaks = np.full(size, False)
    b_breaks[breaks] = True
    segments = _find_segment_indices(b_breaks)[0]
    assert segments == expected
