import pytest
from pyxdf.pyxdf import _check_monotonicity, _monotonic_increasing

from mock_data_stream import MockStreamData

# Monotonic data.


def test_empty_list():
    a = []
    monotonicity = _monotonic_increasing(a)
    assert monotonicity.result is True
    assert monotonicity.n == 0
    assert monotonicity.dec_count == 0
    assert monotonicity.eq_count == 0


def test_single_item_list():
    a = [0]
    monotonicity = _monotonic_increasing(a)
    assert monotonicity.result is True
    assert monotonicity.n == 0
    assert monotonicity.dec_count == 0
    assert monotonicity.eq_count == 0


def test_strict_increasing():
    a = list(range(-4, 5))
    monotonicity = _monotonic_increasing(a)
    assert monotonicity.result is True
    assert monotonicity.n == len(a) - 1
    assert monotonicity.dec_count == 0
    assert monotonicity.eq_count == 0


@pytest.mark.parametrize(
    "a, expected_dec_count, expected_eq_count",
    (
        ([-4] + list(range(-4, 5)), 0, 1),
        (list(range(-4, 5)) + [4], 0, 1),
        ([-4, -3, -2, -1, -1, 0, 1, 2, 3, 4], 0, 1),
        ([-4, -3, -2, -1, 0, 1, 1, 2, 3, 4], 0, 1),
        ([-4, -3, -2, -2, -1, 0, 1, 2, 2, 3, 4], 0, 2),
    ),
)
def test_non_decreasing(a, expected_dec_count, expected_eq_count):
    monotonicity = _monotonic_increasing(a)
    assert monotonicity.result is True
    assert monotonicity.n == len(a) - 1
    assert monotonicity.dec_count == expected_dec_count
    assert monotonicity.eq_count == expected_eq_count


# Non-monotonic data.


@pytest.mark.parametrize(
    "a, expected_dec_count, expected_eq_count",
    (
        ([-4, -2, -3, -1, 0, 1, 2, 3, 4], 1, 0),
        ([-4, -3, -2, -1, 0, 1, 3, 2, 4], 1, 0),
        ([-4, -3, -2, -1, 0, -1, 2, 3, 4], 1, 0),
        ([-4, -4, -3, -2, -1, 0, -1, 2, 3, 4], 1, 1),
        ([-4, -3, -2, -1, 0, -1, 2, 3, 3, 4], 1, 1),
        ([-4, -2, -3, -1, 0, 1, 3, 2, 4], 2, 0),
        ([-4, -2, -3, -1, 0, 0, 1, 3, 2, 4], 2, 1),
        ([-4, -4, -2, -3, -1, 0, 1, 3, 2, 4, 4], 2, 2),
    ),
)
def test_decreasing(a, expected_dec_count, expected_eq_count):
    monotonicity = _monotonic_increasing(a)
    assert monotonicity.result is False
    assert monotonicity.n == len(a) - 1
    assert monotonicity.dec_count == expected_dec_count
    assert monotonicity.eq_count == expected_eq_count


# Monotonic streams: test both time-stamps and clock-times.


def test_strict_increasing_stream():
    time_stamps = list(range(-4, 5))
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=time_stamps,
        )
    }
    monotonicity = _check_monotonicity(streams)
    assert monotonicity["time_stamps"][1] is True
    assert monotonicity["clock_times"][1] is True


@pytest.mark.parametrize(
    "time_stamps",
    (
        [-4] + list(range(-4, 5)),
        list(range(-4, 5)) + [4],
        [-4, -3, -2, -1, -1, 0, 1, 2, 3, 4],
        [-4, -3, -2, -1, 0, 1, 1, 2, 3, 4],
        [-4, -3, -2, -2, -1, 0, 1, 2, 2, 3, 4],
    ),
)
def test_non_decreasing_stream(time_stamps):
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=time_stamps,
        )
    }
    monotonicity = _check_monotonicity(streams)
    assert monotonicity["time_stamps"][1] is True
    assert monotonicity["clock_times"][1] is True


# Non-monotonic streams.


@pytest.mark.parametrize(
    "time_stamps",
    (
        [-4, -2, -3, -1, 0, 1, 2, 3, 4],
        [-4, -3, -2, -1, 0, 1, 3, 2, 4],
        [-4, -3, -2, -1, 0, -1, 2, 3, 4],
        [-4, -4, -3, -2, -1, 0, -1, 2, 3, 4],
        [-4, -3, -2, -1, 0, -1, 2, 3, 3, 4],
        [-4, -2, -3, -1, 0, 1, 3, 2, 4],
        [-4, -2, -3, -1, 0, 0, 1, 3, 2, 4],
        [-4, -4, -2, -3, -1, 0, 1, 3, 2, 4, 4],
    ),
)
def test_decreasing_stream(time_stamps):
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            # Reversed non-monotonic data will still be non-monotonic, but
            # with different statistics.
            clock_times=list(reversed(time_stamps)),
        )
    }
    monotonicity = _check_monotonicity(streams)
    assert monotonicity["time_stamps"][1] is False
    assert monotonicity["clock_times"][1] is False


# Mixed monotonic/non-monotonic time-stamps and clock-times.


@pytest.mark.parametrize(
    "time_stamps, clock_times, expected_ts, expected_ct",
    (
        (list(range(-4, 5)), list(reversed(range(-4, 5))), True, False),
        (list(reversed(range(-4, 5))), list(range(-4, 5)), False, True),
    ),
)
def test_mixed_strict_increasing_stream(
    time_stamps,
    clock_times,
    expected_ts,
    expected_ct,
):
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
        )
    }
    monotonicity = _check_monotonicity(streams)
    assert monotonicity["time_stamps"][1] is expected_ts
    assert monotonicity["clock_times"][1] is expected_ct


# Multiple mixed monotonic/non-monotonic time-stamps and clock-times.


def test_multiple_strict_increasing_stream():
    time_stamps = list(range(-4, 5))
    streams = {
        1: MockStreamData(
            stream_id=1,
            time_stamps=time_stamps,
            clock_times=time_stamps,
        ),
        2: MockStreamData(
            stream_id=2,
            time_stamps=time_stamps,
            clock_times=list(reversed(time_stamps)),
        ),
        3: MockStreamData(
            stream_id=3,
            time_stamps=list(reversed(time_stamps)),
            clock_times=time_stamps,
        ),
        4: MockStreamData(
            stream_id=4,
            time_stamps=list(reversed(time_stamps)),
            clock_times=list(reversed(time_stamps)),
        ),
    }
    monotonicity = _check_monotonicity(streams)
    assert all(monotonicity["time_stamps"].values()) is False
    assert all(monotonicity["clock_times"].values()) is False
    assert monotonicity["time_stamps"][1] is True
    assert monotonicity["clock_times"][1] is True
    assert monotonicity["time_stamps"][2] is True
    assert monotonicity["clock_times"][2] is False
    assert monotonicity["time_stamps"][3] is False
    assert monotonicity["clock_times"][3] is True
    assert monotonicity["time_stamps"][4] is False
    assert monotonicity["clock_times"][4] is False
