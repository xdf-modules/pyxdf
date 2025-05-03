import numpy as np
import pytest
from mock_data_stream import MockStreamData

from pyxdf.pyxdf import _jitter_removal, _sort_stream_data

# Single-segment timeseries data


@pytest.mark.parametrize(
    "t_start, t_end",
    [
        (1, 11),
        (-10, 0),
        (-10, 11),
    ],
)
def test_jitter_removal(t_start, t_end):
    srate = 1
    tdiff = 1
    time_stamps = list(range(t_start, t_end, tdiff))
    streams = {1: MockStreamData(time_stamps=time_stamps, srate=srate, tdiff=tdiff)}
    _jitter_removal(streams, threshold_seconds=1, threshold_samples=1)
    stream = streams[1]
    assert stream.segments == [(0, len(time_stamps) - 1)]
    np.testing.assert_allclose(stream.time_stamps, time_stamps, atol=1e-14)
    np.testing.assert_equal(stream.time_series[:, 0], time_stamps)
    np.testing.assert_allclose(stream.effective_srate, srate, rtol=1e-15)


@pytest.mark.parametrize(
    "t_start, t_end",
    [
        (1, 1001),
        (-1000, 0),
        (-1000, 1001),
    ],
)
def test_jitter_removal_with_jitter(t_start, t_end):
    srate = 500
    tdiff = 0.002
    time_stamps_orig = np.arange(t_start, t_end, tdiff)
    # Add jitter
    jitter = tdiff * 0.001
    rng = np.random.default_rng(9)
    time_stamps = time_stamps_orig + rng.standard_normal(len(time_stamps_orig)) * jitter
    streams = {
        1: MockStreamData(
            time_stamps=time_stamps,
            srate=srate,
            tdiff=tdiff,
        )
    }
    _jitter_removal(streams, threshold_seconds=1, threshold_samples=500)
    stream = streams[1]
    assert stream.segments == [(0, len(time_stamps) - 1)]
    np.testing.assert_allclose(
        stream.time_stamps,
        time_stamps_orig,
        rtol=1e-05,
        atol=1e-09,
    )
    np.testing.assert_equal(stream.time_series[:, 0], time_stamps)
    np.testing.assert_allclose(stream.effective_srate, srate, rtol=1e-10)


# Multi-segment timeseries data


@pytest.mark.parametrize(
    "segments, t_starts",
    [
        ([(0, 9), (10, 19)], [0, 11]),
        ([(0, 9), (10, 19)], [-10, 1]),
        ([(0, 19), (20, 39)], [-10, 11]),
    ],
)
def test_jitter_removal_two_segments(segments, t_starts):
    srate = 1
    tdiff = 1
    time_stamps = [
        np.arange(start_time, start_time + (seg_end - seg_start) + 1, tdiff)
        for (seg_start, seg_end), start_time in zip(segments, t_starts)
    ]
    time_stamps = np.hstack(time_stamps)
    streams = {1: MockStreamData(time_stamps=time_stamps, srate=srate, tdiff=tdiff)}
    _jitter_removal(streams, threshold_seconds=1, threshold_samples=1)
    stream = streams[1]
    assert stream.segments == segments
    np.testing.assert_allclose(stream.time_stamps, time_stamps, atol=1e-14)
    np.testing.assert_equal(stream.time_series[:, 0], time_stamps)
    np.testing.assert_allclose(stream.effective_srate, srate, rtol=1e-15)


# Non-monotonic timeseries data - handled by either sorting time-stamps or both
# time-stamps and samples together


@pytest.mark.parametrize("reorder_timeseries", [False, True])
@pytest.mark.parametrize(
    "segments, t_starts",
    [
        ([(0, 9), (10, 19)], [11, 0]),
        ([(0, 9), (10, 19)], [1, -10]),
        ([(0, 19), (20, 39)], [11, -10]),
    ],
)
def test_jitter_removal_two_segments_non_monotonic_sorted(
    segments, t_starts, reorder_timeseries
):
    srate = 1
    tdiff = 1
    time_stamps = [
        np.arange(start_time, start_time + (seg_end - seg_start) + 1, tdiff)
        for (seg_start, seg_end), start_time in zip(segments, t_starts)
    ]
    time_stamps = np.hstack(time_stamps)
    streams = {
        1: _sort_stream_data(
            1,
            MockStreamData(time_stamps=time_stamps, srate=srate, tdiff=tdiff),
            reorder_timeseries,
        )
    }
    _jitter_removal(streams, threshold_seconds=1, threshold_samples=1)
    stream = streams[1]
    assert stream.segments == segments
    np.testing.assert_allclose(stream.time_stamps, sorted(time_stamps), atol=1e-14)
    if reorder_timeseries:
        np.testing.assert_equal(stream.time_series[:, 0], sorted(time_stamps))
    else:
        np.testing.assert_equal(stream.time_series[:, 0], time_stamps)
    np.testing.assert_allclose(stream.effective_srate, srate, rtol=1e-15)


@pytest.mark.parametrize("reorder_timeseries", [False, True])
def test_jitter_removal_glitch_sorted(reorder_timeseries):
    srate = 1
    tdiff = 1
    time_stamps = [1, 2, 3, 4] + [6] + [5, 7, 8, 9, 10]
    streams = {
        1: _sort_stream_data(
            1,
            MockStreamData(time_stamps=time_stamps, srate=srate, tdiff=tdiff),
            reorder_timeseries,
        )
    }
    _jitter_removal(streams, threshold_seconds=1, threshold_samples=1)
    stream = streams[1]
    assert stream.segments == [(0, len(time_stamps) - 1)]
    np.testing.assert_allclose(stream.time_stamps, sorted(time_stamps), rtol=1e-14)
    if reorder_timeseries:
        np.testing.assert_equal(stream.time_series[:, 0], sorted(time_stamps))
    else:
        np.testing.assert_equal(stream.time_series[:, 0], time_stamps)
    np.testing.assert_allclose(stream.effective_srate, srate, rtol=1e-15)
