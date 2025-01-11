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

