import numpy as np
from pyxdf.pyxdf import _detect_breaks, _ensure_sorted


class MockStreamData:
    def __init__(self, time_stamps, tdiff, fmt="float32"):
        self.time_stamps = np.array(time_stamps)
        self.tdiff = tdiff
        self.fmt = fmt
        if fmt == "string":
            self.time_series = [str(x) for x in time_stamps]
        else:
            self.time_series = np.array(time_stamps, dtype=fmt)


# Monotonic timeseries data.


def test_detect_no_breaks_seconds():
    timestamps = list(range(-5, 5))
    stream = MockStreamData(timestamps, 1)
    # if diff > 2 and larger 0 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=0)
    assert breaks.size == 0
    # if diff > 1 and larger 0 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    assert breaks.size == 0


def test_detect_no_breaks_samples():
    timestamps = list(range(-5, 5))
    stream = MockStreamData(timestamps, 1)
    # if diff > 0 and larger 2 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=2)
    assert breaks.size == 0
    # if diff > 0 and larger 1 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=1)
    assert breaks.size == 0


def test_detect_breaks_seconds():
    timestamps = list(range(-5, 5, 2))
    stream = MockStreamData(timestamps, 1)
    # if diff > 1 and larger 0 * tdiff -> 4
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1


def test_detect_breaks_samples():
    timestamps = list(range(-5, 5, 2))
    stream = MockStreamData(timestamps, 1)
    # if diff > 0 and larger 1 * tdiff -> 4
    breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=1)
    assert breaks.size == len(timestamps) - 1


def test_detect_breaks_gap_in_negative():
    timestamps = [-4, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    # if diff > 1 and larger 1 * tdiff -> 1
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 1
    timestamps = [-4, -2, -1, 0, 1, 2, 3, 4]
    stream = MockStreamData(timestamps, 1)
    # if diff > 1 and larger 1 * tdiff -> 1
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 1
    # if diff > 0.1 and larger 0 * tdiff -> 7
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1


def test_detect_breaks_gap_in_positive():
    timestamps = [1, 3, 4, 5, 6]
    stream = MockStreamData(timestamps, 1)
    # if diff > 1 and larger 1 * tdiff -> 1
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 1
    # if diff > 0.1 and larger 0 * tdiff -> 4
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1


# Non-monotonic timeseries data.


def test_detect_breaks_reverse():
    timestamps = list(reversed(range(-5, 5)))
    stream = MockStreamData(timestamps, 1)
    stream = _ensure_sorted(1, stream)
    # Timeseries should now also be sorted.
    assert np.all(stream.time_series == sorted(timestamps))
    # if diff > 1 and larger 0 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=0)
    assert breaks.size == 0


def test_detect_breaks_non_monotonic_num():
    timestamps = [-4, -5, -3, -2, 0, 0, 1, 2]
    stream = MockStreamData(timestamps, 1)
    stream = _ensure_sorted(1, stream)
    # Timeseries data should now also be sorted.
    assert np.all(stream.time_series == sorted(timestamps))
    # if diff > 1 and larger 1 * tdiff -> 1
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 4
    # if diff > 2 and larger 2 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=2)
    assert breaks.size == 0
    # if diff > 0.1 and larger 0 * tdiff -> 6
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 2
    assert list(breaks) == [1, 2, 3, 4, 6, 7]


def test_detect_breaks_non_monotonic_str():
    timestamps = [-4, -5, -3, -2, 0, 0, 1, 2]
    stream = MockStreamData(timestamps, 1, "string")
    stream = _ensure_sorted(1, stream)
    # Timeseries data should now also be sorted.
    assert np.all(stream.time_series == [str(x) for x in sorted(timestamps)])
    # if diff > 1 and larger 1 * tdiff -> 1
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert breaks.size == 1
    assert breaks[0] == 4
    # if diff > 2 and larger 2 * tdiff -> 0
    breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=2)
    assert breaks.size == 0
    # if diff > 0.1 and larger 0 * tdiff -> 6
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 2
    assert list(breaks) == [1, 2, 3, 4, 6, 7]
