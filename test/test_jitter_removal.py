from pyxdf.pyxdf import _detect_breaks


class MockStreamData:
    def __init__(self, time_stamps, tdiff):
        self.time_stamps = time_stamps
        self.tdiff = tdiff


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
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
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


def test_detect_breaks_reverse():
    timestamps = list(reversed(range(-5, 5)))
    stream = MockStreamData(timestamps, 1)
    # if diff <= 0 -> 9
    breaks = _detect_breaks(stream, threshold_seconds=0, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1


def test_detect_breaks_gaps_non_monotonic():
    timestamps = [-4, 1, -3, -2, -1, 1, 5, 1, 2]
    stream = MockStreamData(timestamps, 1)
    # if diff <= 0 or diff > 1 and larger 1 * tdiff -> 5
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert list(breaks) == [1, 2, 5, 6, 7]
    # if diff <= 0 or diff > 2 and larger 1 * tdiff -> 4
    breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=1)
    assert list(breaks) == [1, 2, 6, 7]
    # if diff <= 0 or diff > 0.1 and larger 0 * tdiff -> 8
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1


def test_detect_breaks_strict_non_monotonic():
    timestamps = [-4, -5, -3, -2, -1, 0, 0, 1, 2]
    stream = MockStreamData(timestamps, 1)
    # if diff <= 0 or diff > 1 and larger 1 * tdiff -> 3
    breaks = _detect_breaks(stream, threshold_seconds=1, threshold_samples=1)
    assert list(breaks) == [1, 2, 6]
    # if diff <= 0 or diff > 2 and larger 2 * tdiff -> 2
    breaks = _detect_breaks(stream, threshold_seconds=2, threshold_samples=2)
    assert list(breaks) == [1, 6]
    # if diff <= 0 or diff > 0.1 and larger 0 * tdiff -> 8
    breaks = _detect_breaks(stream, threshold_seconds=0.1, threshold_samples=0)
    assert breaks.size == len(timestamps) - 1
