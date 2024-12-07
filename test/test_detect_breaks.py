import numpy as np
import pytest
from pyxdf.pyxdf import _detect_breaks

from mock_data_stream import MockStreamData

# Monotonic timeseries data


@pytest.mark.parametrize("seconds", (2, 1))
def test_detect_no_breaks_seconds(seconds):
    time_stamps = list(range(-5, 5))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > seconds -> 0
    b_breaks = _detect_breaks(stream, threshold_seconds=seconds, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 0


@pytest.mark.parametrize("samples", (2, 1))
def test_detect_no_breaks_samples(samples):
    time_stamps = list(range(-5, 5))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > samples * tdiff -> 0
    b_breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=samples)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 0


def test_detect_no_breaks_equal():
    time_stamps = [-2, -2, -1, 0, 0, 1, 2, 2]
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff < 0 or diff > 1 -> 0
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 0


def test_detect_breaks_seconds():
    time_stamps = list(range(-5, 5, 2))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > 1 -> 4
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


def test_detect_breaks_samples():
    time_stamps = list(range(-5, 5, 2))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > 1 -> 4
    breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=1)
    assert breaks.size == len(time_stamps) - 1


def test_detect_breaks_gap_in_negative():
    time_stamps = [-4, 1, 2, 3, 4]
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > 1 -> 1
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 1
    assert breaks[0] == 0
    time_stamps = [-4, -2, -1, 0, 1, 2, 3, 4]
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > 1 -> 1
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 1
    assert breaks[0] == 0
    # if diff > 0.1 -> 7
    b_breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


def test_detect_breaks_gap_in_positive():
    time_stamps = [1, 3, 4, 5, 6]
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff > 1 -> 1
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 1
    assert breaks[0] == 0
    # if diff > 0.1 -> 4
    b_breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


# Non-monotonic timeseries data: segment at negative time-intervals or positive
# time-intervals exceeding threshold


def test_detect_breaks_reverse():
    time_stamps = list(reversed(range(-5, 5)))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff < 0 -> 9
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


def test_detect_breaks_non_monotonic_gap_in_negative():
    time_stamps = [-4, -7, -6, -5, -3, -2, -1, 0]
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff < 0 or diff > 1 -> 2
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 2
    assert breaks[0] == 0
    assert breaks[1] == 3
    # if diff < 0 or diff > 2 -> 1
    b_breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 1
    assert breaks[0] == 0
    # if diff < 0 or diff > 0.1 -> 7
    b_breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


def test_detect_breaks_non_monotonic_gap_in_positive():
    time_stamps = [4, 1, 2, 3, 5, 6, 7, 8]
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff < 0 or diff > 1 -> 2
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 2
    assert breaks[0] == 0
    assert breaks[1] == 3
    # if diff < 0 or diff > 2 -> 1
    b_breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 1
    assert breaks[0] == 0
    # if diff < 0 or diff > 0.1 -> 7
    b_breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1
