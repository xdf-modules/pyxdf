import numpy as np
import pytest
from pyxdf.pyxdf import _clock_sync

from mock_data_stream import MockStreamData


# No clock resets


@pytest.mark.parametrize("n_clock_offsets", list(range(0, 5)))
@pytest.mark.parametrize("handle_clock_resets", [True, False])
def test_sync_empty_stream(n_clock_offsets, handle_clock_resets):
    time_stamps = []
    clock_tdiff = 5
    duration = n_clock_offsets * clock_tdiff
    clock_times = list(range(0, duration, clock_tdiff))
    clock_values = [0] * n_clock_offsets
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        handle_clock_resets=handle_clock_resets,
    )
    np.testing.assert_equal(streams[1].time_stamps, time_stamps)
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


@pytest.mark.parametrize("n_time_stamps", list(range(0, 5)))
@pytest.mark.parametrize("handle_clock_resets", [True, False])
def test_sync_empty_offsets(n_time_stamps, handle_clock_resets):
    tdiff = 1
    time_stamps = list(range(0, n_time_stamps, tdiff))
    clock_times = []
    clock_values = []
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        handle_clock_resets=handle_clock_resets,
    )
    np.testing.assert_equal(streams[1].time_stamps, time_stamps)
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


@pytest.mark.parametrize("n_clock_offsets", list(range(2, 5)))
@pytest.mark.parametrize("clock_value", list(range(-10, 11, 5)))
@pytest.mark.parametrize("handle_clock_resets", [True, False])
def test_sync_no_resets(n_clock_offsets, clock_value, handle_clock_resets):
    tdiff = 1
    clock_tdiff = 5
    duration = (n_clock_offsets - 1) * clock_tdiff
    time_stamps = np.arange(0, duration, tdiff)
    # Clock offsets cover the full duration of time-stamps
    clock_times = list(range(0, duration + tdiff, clock_tdiff))
    clock_values = [clock_value] * len(clock_times)
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        handle_clock_resets=handle_clock_resets,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        time_stamps + clock_value,
        atol=1e-14,
    )
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


@pytest.mark.parametrize("n_clock_offsets", list(range(2, 4)))
@pytest.mark.parametrize("clock_value", list(range(-10, 11, 5)))
@pytest.mark.parametrize("handle_clock_resets", [True, False])
def test_sync_no_resets_bounds(n_clock_offsets, clock_value, handle_clock_resets):
    tdiff = 1
    clock_tdiff = 5
    duration = n_clock_offsets * clock_tdiff
    # Time-stamps extend 5 seconds before and after clock offsets
    time_stamps = np.arange(-clock_tdiff, duration, tdiff)
    clock_times = list(range(0, duration, clock_tdiff))
    clock_values = [clock_value] * len(clock_times)
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        handle_clock_resets=handle_clock_resets,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        time_stamps + clock_value,
        atol=1e-14,
    )
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


@pytest.mark.parametrize("n_clock_offsets", list(range(2, 4)))
@pytest.mark.parametrize("clock_value", list(range(-10, 11, 5)))
@pytest.mark.parametrize("handle_clock_resets", [True, False])
def test_sync_no_resets_bounds_jitter(
    n_clock_offsets, clock_value, handle_clock_resets
):
    tdiff = 1
    clock_tdiff = 5
    duration = n_clock_offsets * clock_tdiff
    # Time-stamps extend 5 seconds before and after clock offsets
    time_stamps = np.hstack(
        [
            np.arange(-clock_tdiff, 0, tdiff),
            np.zeros(duration),
            np.arange(1, clock_tdiff + tdiff, tdiff),
        ]
    )
    jit_std = 0.0001
    rng = np.random.default_rng(9)
    clock_times = np.arange(0, duration, clock_tdiff)
    clock_times = (
        clock_times + rng.standard_normal(len(clock_times)) * jit_std
    ).tolist()
    clock_values = clock_value + rng.standard_normal(n_clock_offsets) * jit_std
    clock_values = clock_values.tolist()
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        handle_clock_resets=handle_clock_resets,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        time_stamps + clock_value,
        atol=1e-3,
    )
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


# Clock resets


def test_sync_clock_jumps_forward_break_at_reset():
    expected = [0, 1, 2, 3, 4] + [5, 6, 7, 8, 9]
    # Time-stamps within clock regions
    time_stamps = [1, 2, 3, 4, 5] + [17, 18, 19, 20, 21]
    clock_times = [1, 6, 17, 22]
    clock_values = [-1, -1, -12, -12]
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )


def test_sync_clock_jumps_forward_break_between_reset():
    expected = [3, 4, 5, 6, 7, 8, 9, 10] + [9, 10, 11, 12, 13, 14, 15, 16]
    # Time-stamps between clock regions
    time_stamps = [4, 5, 6, 7, 8, 9, 10, 11] + [12, 13, 14, 15, 16, 17, 18, 19]
    clock_times = [1, 6, 17, 22]
    clock_values = [-1, -1, -3, -3]
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )


def test_sync_clock_jumps_backward_break_at_reset():
    expected = [5, 6, 7, 8, 9] + [10, 11, 12, 13, 14]
    # Time-stamps within clock regions
    time_stamps = [17, 18, 19, 20, 21] + [1, 2, 3, 4, 5]
    clock_times = [17, 22, 1, 6]
    clock_values = [-12, -12, 9, 9]
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )


def test_sync_clock_jumps_backward_break_between_reset():
    expected = [5, 6, 7, 8, 9, 10, 11, 12] + [10, 11, 12, 13, 14, 15, 16, 17]
    # Time-stamps between clock regions
    time_stamps = [17, 18, 19, 20, 21, 22, 23, 24] + [1, 2, 3, 4, 5, 6, 7, 8]
    clock_times = [17, 22, 1, 6]
    clock_values = [-12, -12, 9, 9]
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "clock_offsets",
    [
        ((0, 1), (3, 4)),
        ((0, 3), (5, 8)),
        ((0, 4), (6, 10), (12, 16)),
    ],
)
@pytest.mark.parametrize("tdiff", [1, 1 / 10, 1 / 100])
@pytest.mark.parametrize("clock_tdiff", [5, 10])
def test_sync_clock_jumps_forward_tdiffs(clock_offsets, tdiff, clock_tdiff):
    offsets_per_range = [(end - start) + 1 for start, end in clock_offsets]
    clock_reset_times = [
        (
            start * clock_tdiff,
            (end + 1) * clock_tdiff,
        )
        for start, end in clock_offsets
    ]
    clock_times = [
        t
        for start, end in clock_reset_times
        for t in range(
            start,
            end,
            clock_tdiff,
        )
    ]
    clock_values = np.repeat(
        [-start for start, _ in clock_reset_times], offsets_per_range
    ).tolist()
    time_stamps = np.hstack(
        [np.arange(start, stop, tdiff) for start, stop in clock_reset_times]
    )
    expected = np.hstack(
        [np.arange(0, clock_tdiff * offsets, tdiff) for offsets in offsets_per_range]
    )
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        reset_threshold_seconds=clock_tdiff - 1,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


@pytest.mark.parametrize(
    "clock_offsets",
    [
        ((3, 4), (0, 1)),
        ((5, 8), (0, 3)),
        ((12, 16), (6, 10), (0, 4)),
    ],
)
@pytest.mark.parametrize("tdiff", [1, 1 / 10, 1 / 100])
@pytest.mark.parametrize("clock_tdiff", [5, 10])
def test_sync_clock_jumps_backward_tdiffs(clock_offsets, tdiff, clock_tdiff):
    offsets_per_range = [(end - start) + 1 for start, end in clock_offsets]
    clock_reset_times = [
        (
            start * clock_tdiff,
            (end + 1) * clock_tdiff,
        )
        for start, end in clock_offsets
    ]
    clock_times = [
        t
        for start, end in clock_reset_times
        for t in range(
            start,
            end,
            clock_tdiff,
        )
    ]
    clock_values = np.repeat(
        [-start for start, _ in clock_reset_times], offsets_per_range
    ).tolist()
    time_stamps = np.hstack(
        [np.arange(start, stop, tdiff) for start, stop in clock_reset_times]
    )
    expected = np.hstack(
        [np.arange(0, clock_tdiff * offsets, tdiff) for offsets in offsets_per_range]
    )
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)


@pytest.mark.parametrize(
    "clock_offsets",
    [
        ((3, 4), (0, 1), (3, 4)),
        ((5, 8), (0, 3), (5, 8)),
        ((12, 16), (6, 10), (0, 4), (6, 10), (12, 16)),
    ],
)
@pytest.mark.parametrize("tdiff", [1, 1 / 10, 1 / 100])
@pytest.mark.parametrize("clock_tdiff", [5, 10])
def test_sync_clock_jumps_forward_backward_tdiffs(clock_offsets, tdiff, clock_tdiff):
    offsets_per_range = [(end - start) + 1 for start, end in clock_offsets]
    clock_reset_times = [
        (
            start * clock_tdiff,
            (end + 1) * clock_tdiff,
        )
        for start, end in clock_offsets
    ]
    clock_times = [
        t
        for start, end in clock_reset_times
        for t in range(
            start,
            end,
            clock_tdiff,
        )
    ]
    clock_values = np.repeat(
        [-start for start, _ in clock_reset_times], offsets_per_range
    ).tolist()
    time_stamps = np.hstack(
        [np.arange(start, stop, tdiff) for start, stop in clock_reset_times]
    )
    expected = np.hstack(
        [np.arange(0, clock_tdiff * offsets, tdiff) for offsets in offsets_per_range]
    )
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    _clock_sync(
        streams,
        reset_threshold_seconds=clock_tdiff - 1,
    )
    np.testing.assert_allclose(
        streams[1].time_stamps,
        expected,
        atol=1e-13,
    )
    np.testing.assert_equal(streams[1].time_series[:, 0], time_stamps)
    np.testing.assert_equal(streams[1].clock_times, clock_times)
    np.testing.assert_equal(streams[1].clock_values, clock_values)
