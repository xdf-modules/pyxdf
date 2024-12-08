import pytest
from pyxdf.pyxdf import _detect_breaks


class MockStreamData:
    def __init__(self, time_stamps, tdiff):
        self.time_stamps = time_stamps
        self.tdiff = tdiff


def test_detect_no_breaks():
    timestamps = list(range(-5, 5))
    stream = MockStreamData(timestamps, 1)
    # if diff > 2 and larger 500 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=500)
    assert breaks.size == 0
    # if diff > 0.1 and larger 1 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=1)
    assert breaks.size == 0


def test_detect_breaks_reverse():
    timestamps = list(reversed(range(-5, 5)))
    stream = MockStreamData(timestamps, 1)
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1


def test_detect_breaks_gap_in_negative():
    timestamps = [-4, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 1
    timestamps = [-4, -2, -1, 0, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 1


def test_detect_breaks_gap_in_positive():
    timestamps = [1, 3, 4, 5, 6]
    stream = MockStreamData(timestamps, 1)
    # if diff > 1 and larger 1 * tdiff -> 1 -> 1
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 1
    # if diff > 0.1 and larger 0 * tdiff ->
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1
