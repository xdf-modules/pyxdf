from contextlib import nullcontext

import numpy as np
import pytest

from mock_data_stream import MockStreamData


@pytest.mark.parametrize(
    "srate, tdiff, expected_srate, expected_tdiff, context",
    [
        (None, None, None, None, nullcontext()),
        (1, 1, 1, 1, nullcontext()),
        (2, 1 / 2, 2, 1 / 2, nullcontext()),
        (3, None, 3, 1 / 3, nullcontext()),
        (None, 1 / 4, 4, 1 / 4, nullcontext()),
        (
            1,
            2,
            None,
            None,
            pytest.raises(ValueError, match="tdiff not equal to 1/srate"),
        ),
    ],
)
def test_srate_tdiff(srate, tdiff, expected_srate, expected_tdiff, context):
    with context:
        stream = MockStreamData(
            srate=srate,
            tdiff=tdiff,
        )
        assert stream.srate == expected_srate
        assert stream.tdiff == expected_tdiff


@pytest.mark.parametrize(
    "time_series, expected, nchans, context",
    [
        ([], [], 1, nullcontext()),
        ([["a"]], [["a"]], 1, nullcontext()),
        ([["a"], ["b"]], [["a"], ["b"]], 1, nullcontext()),
        ([["a", "b"], ["a", "b"]], [["a", "b"], ["a", "b"]], 2, nullcontext()),
        (
            ["a", "a"],
            None,
            None,
            pytest.raises(
                ValueError, match="All string samples must be lists of strings"
            ),
        ),
        (
            [["a"], ["a", "b"]],
            None,
            None,
            pytest.raises(
                ValueError, match="All samples must have the same number of channels"
            ),
        ),
        (
            [["a"], [1]],
            None,
            None,
            pytest.raises(ValueError, match="All string sample values must be strings"),
        ),
    ],
)
def test_time_series_str(time_series, expected, nchans, context):
    with context:
        stream = MockStreamData(
            time_series=time_series,
            fmt="string",
        )
        assert stream.time_series == expected
        assert stream.nchans == nchans


@pytest.mark.parametrize(
    "clock_times, clock_values, expected_times, expected_values, context",
    [
        (None, None, [], [], nullcontext()),
        ([], [], [], [], nullcontext()),
        ([0], None, [0], [], nullcontext()),
        (None, [10], [], [10], nullcontext()),
        ([0], [10], [0], [10], nullcontext()),
        ([0, 1], [10, 10], [0, 1], [10, 10], nullcontext()),
        (
            0,
            [0],
            None,
            None,
            pytest.raises(ValueError, match="Clock times must be a list"),
        ),
        (
            [0],
            0,
            None,
            None,
            pytest.raises(ValueError, match="Clock values must be a list"),
        ),
        (
            [0],
            [],
            None,
            None,
            pytest.raises(
                ValueError, match="Clock times and values must be the same length"
            ),
        ),
        (
            [],
            [10],
            None,
            None,
            pytest.raises(
                ValueError, match="Clock times and values must be the same length"
            ),
        ),
        (
            ["0"],
            [10],
            None,
            None,
            pytest.raises(ValueError, match="All clock times must be numeric"),
        ),
        (
            [0],
            ["10"],
            None,
            None,
            pytest.raises(ValueError, match="All clock values must be numeric"),
        ),
    ],
)
def test_clock_offsets(
    clock_times,
    clock_values,
    expected_times,
    expected_values,
    context,
):
    with context:
        stream = MockStreamData(
            clock_times=clock_times,
            clock_values=clock_values,
        )
        assert stream.clock_times == expected_times
        assert stream.clock_values == expected_values


@pytest.mark.parametrize("fmt", [np.float32, np.float64, np.int32, np.int64, "string"])
def test_mock_stream_timeseries_defaults(fmt):
    stream_id = 2
    srate = 1
    tdiff = 1
    # Time-stamps should be returned as a float64 array
    time_stamps = list(range(0, 10))
    clock_times = [0]
    clock_values = [-10]
    stream = MockStreamData(
        stream_id=stream_id,
        srate=srate,
        tdiff=tdiff,
        fmt=fmt,
        time_stamps=time_stamps,
        clock_times=clock_times,
        clock_values=clock_values,
    )
    assert stream.stream_id == stream_id
    assert stream.srate == srate
    assert stream.tdiff == tdiff
    assert np.isdtype(stream.time_stamps.dtype, np.float64)
    np.testing.assert_equal(stream.time_stamps, time_stamps)
    if fmt == "string":
        assert all([isinstance(sample, list) for sample in stream.time_series])
        assert all(
            [
                isinstance(value, str)
                for sample in stream.time_series
                for value in sample
            ]
        )
        np.testing.assert_allclose(
            np.array(stream.time_series, dtype=np.float64)[:, 0], time_stamps
        )
    else:
        assert np.isdtype(stream.time_series.dtype, fmt)
        np.testing.assert_allclose(stream.time_series[:, 0], time_stamps)
    assert stream.nchans == 1
    assert stream.clock_times == clock_times
    assert stream.clock_values == clock_values


@pytest.mark.parametrize("nchans", [1, 2])
@pytest.mark.parametrize("fmt", [np.float64, "string"])
def test_mock_stream_timeseries(fmt, nchans):
    time_stamps = np.arange(0, 1, 0.1)
    samples = range(10, 20)
    if fmt == "string":
        time_series = [[str(x)] * nchans for x in samples]
    else:
        time_series = [[x] * nchans for x in samples]
    stream = MockStreamData(
        time_stamps=time_stamps,
        time_series=time_series,
        fmt=fmt,
    )
    assert np.isdtype(stream.time_stamps.dtype, np.float64)
    np.testing.assert_equal(stream.time_stamps, time_stamps)
    np.testing.assert_equal(stream.time_series, time_series)
    assert stream.nchans == nchans
