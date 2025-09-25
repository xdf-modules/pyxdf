import numpy as np
import pytest
from pyxdf.pyxdf import _detect_breaks

from mock_data_stream import MockStreamData

# Segment at absolute time-intervals exceeding a given threshold. Sequences of
# identical absolute intervals should be segmented the same regardless of
# whether they are positive or negative.

# Monotonic timeseries data


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("seconds", (2, 1))
def test_detect_no_breaks_seconds(seconds, reverse):
    time_stamps = list(range(-5, 5))
    if reverse:
        time_stamps = list(reversed(time_stamps))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if abs(diff) > seconds -> 0
    b_breaks = _detect_breaks(stream, threshold_seconds=seconds, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 0


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("samples", (2, 1))
def test_detect_no_breaks_samples(samples, reverse):
    time_stamps = list(range(-5, 5))
    if reverse:
        time_stamps = list(reversed(time_stamps))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if abs(diff) > samples * tdiff -> 0
    b_breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=samples)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 0


@pytest.mark.parametrize("reverse", (False, True))
def test_detect_no_breaks_equal(reverse):
    time_stamps = [-2, -2, -1, 0, 0, 1, 2, 2]
    if reverse:
        time_stamps = list(reversed(time_stamps))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if abs(diff) > 1 -> 0
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == 0


@pytest.mark.parametrize("reverse", (False, True))
def test_detect_breaks_seconds(reverse):
    time_stamps = list(range(-5, 5, 2))
    if reverse:
        time_stamps = list(reversed(time_stamps))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if abs(diff) > 1 -> 4
    b_breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    breaks = np.where(b_breaks)[0]
    assert breaks.size == len(time_stamps) - 1


@pytest.mark.parametrize("reverse", (False, True))
def test_detect_breaks_samples(reverse):
    time_stamps = list(range(-5, 5, 2))
    if reverse:
        time_stamps = list(reversed(time_stamps))
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    # if abs(diff) > 1 -> 4
    breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=1)
    assert breaks.size == len(time_stamps) - 1


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("threshold_seconds", [0.1, 1, 2])
@pytest.mark.parametrize("offset", [-10, -1, 0, 10])
def test_detect_breaks_monotonic(offset, threshold_seconds, reverse):
    time_stamps = [0] + [2, 3, 4, 5] + [8]
    if reverse:
        time_stamps = list(reversed(time_stamps))
    time_stamps = np.array(time_stamps) + offset
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    b_breaks = _detect_breaks(
        stream,
        threshold_seconds=threshold_seconds,
        threshold_samples=0,
    )
    breaks = np.where(b_breaks)[0]
    if threshold_seconds == 0.1:
        # if abs(diff) > 0.1 -> 5
        assert breaks.size == len(time_stamps) - 1
    elif threshold_seconds == 1:
        # if abs(diff) > 1 -> 2
        assert breaks.size == 2
        assert breaks[0] == 0
        assert breaks[1] == 4
    elif threshold_seconds == 2:
        # if abs(diff) > 2 -> 1
        assert breaks.size == 1
        if reverse:
            assert breaks[0] == 0
        else:
            assert breaks[0] == 4


# Non-monotonic timeseries data


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("threshold_seconds", [0.1, 1, 2])
@pytest.mark.parametrize("offset", [-10, -1, 0, 10])
def test_detect_breaks_non_monotonic(offset, threshold_seconds, reverse):
    time_stamps = [3] + [0, 1, 2] + [4, 5, 6] + [5]
    if reverse:
        time_stamps = list(reversed(time_stamps))
    time_stamps = np.array(time_stamps) + offset
    stream = MockStreamData(time_stamps=time_stamps, tdiff=1)
    b_breaks = _detect_breaks(
        stream,
        threshold_seconds=threshold_seconds,
        threshold_samples=0,
    )
    breaks = np.where(b_breaks)[0]
    if threshold_seconds == 0.1:
        # if abs(diff) > 0.1 -> 7
        assert breaks.size == len(time_stamps) - 1
    elif threshold_seconds == 1:
        # if abs(diff) > 1 -> 2
        assert breaks.size == 2
        if reverse:
            assert breaks[0] == 3
            assert breaks[1] == 6
        else:
            assert breaks[0] == 0
            assert breaks[1] == 3
    elif threshold_seconds == 2:
        # if abs(diff) > 2 -> 1
        assert breaks.size == 1
        if reverse:
            assert breaks[0] == 6
        else:
            assert breaks[0] == 0
