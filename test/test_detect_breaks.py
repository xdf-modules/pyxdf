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


@pytest.mark.parametrize("threshold_seconds", [0.1, 1, 2])
@pytest.mark.parametrize("offset", [-10, -1, 0, 10])
def test_detect_breaks_monotonic(offset, threshold_seconds):
    time_stamps = [0] + [2, 3, 4, 5] + [8, 9, 10]
    time_stamps = np.array(time_stamps) + offset
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    b_breaks = _detect_breaks(
        stream,
        threshold_seconds=threshold_seconds,
        threshold_samples=0,
    )
    breaks = np.where(b_breaks)[0]
    if threshold_seconds == 0.1:
        # if diff > 0.1 -> 8
        assert breaks.size == len(time_stamps) - 1
    elif threshold_seconds == 1:
        # if diff > 1 -> 2
        assert breaks.size == 2
        assert breaks[0] == 0
        assert breaks[1] == 4
    elif threshold_seconds == 2:
        # if diff > 2 -> 1
        assert breaks.size == 1
        assert breaks[0] == 4


# Non-monotonic timeseries data: segment at negative time-intervals or positive
# time-intervals exceeding threshold


def test_detect_breaks_reverse():
    time_stamps = list(reversed(range(-5, 5)))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if diff < 0 -> 9
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


@pytest.mark.parametrize("threshold_seconds", [0.1, 1, 2])
@pytest.mark.parametrize("offset", [-10, -1, 0, 10])
def test_detect_breaks_non_monotonic(offset, threshold_seconds):
    time_stamps = [3] + [0, 1, 2] + [4, 5, 6] + [5]
    time_stamps = np.array(time_stamps) + offset
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    b_breaks = _detect_breaks(
        stream,
        threshold_seconds=threshold_seconds,
        threshold_samples=0,
    )
    breaks = np.where(b_breaks)[0]
    if threshold_seconds == 0.1:
        # if diff < 0 or diff > 0.1 -> 8
        assert breaks.size == len(time_stamps) - 1
    elif threshold_seconds == 1:
        # if diff < 0 or diff > 1 -> 3
        assert breaks.size == 3
        assert breaks[0] == 0
        assert breaks[1] == 3
        assert breaks[2] == 6
    elif threshold_seconds == 2:
        # if diff < 0 or diff > 2 -> 2
        assert breaks.size == 2
        assert breaks[0] == 0
        assert breaks[1] == 6
