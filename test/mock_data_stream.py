import numpy as np
import pytest


class MockStreamData:
    def __init__(
        self,
        *,
        stream_id=1,
        srate=None,
        tdiff=None,
        fmt="float64",
        time_stamps=[],
        time_series=None,
        clock_times=[],
        clock_values=[],
    ):
        self.stream_id = stream_id
        # Validate srate and tdiff if both provided
        if srate is not None and tdiff is not None:
            assert 1 / srate == tdiff
        self.srate = srate
        self.tdiff = tdiff
        self.fmt = fmt
        # Ensure time_stamps is always a float64 array
        self.time_stamps = np.array(time_stamps, dtype="float64")
        # Use time_stamps as time_series data if no time_series data are provided
        if time_series is None:
            time_series = self.time_stamps
            if fmt == "string":
                self.time_series = [[str(x)] for x in time_series]
                self.nchans = 1
            else:
                self.time_series = np.array(
                    time_series.reshape(time_series.size, 1),
                    dtype=fmt,
                )
                self.nchans = 1
        else:
            # Validate time_series data
            if len(time_series) > 0:
                if fmt == "string":
                    assert all([isinstance(sample, list) for sample in time_series]), (
                        "All string samples must be lists."
                    )
                    assert all(
                        [len(sample) == len(time_series[0]) for sample in time_series]
                    ), "All samples must have the same number of channels."
                    assert all(
                        [isinstance(x, str) for sample in time_series for x in sample]
                    ), "All string sample values must be strings."
                    self.time_series = time_series
                    self.nchans = len(time_series[0])
                else:
                    self.time_series = np.array(time_series, dtype=fmt)
                    self.nchans = self.time_series.shape[1]
        # Validate clock offset data
        if clock_times and clock_values:
            assert len(clock_times) == len(clock_values), (
                "Clock times and values must be the same length."
            )
        self.clock_times = clock_times
        self.clock_values = clock_values
        self.segments = []
        self.clock_segments = []


def test_mock_stream_default_timeseries_num():
    stream_id = 2
    srate = 10
    tdiff = 0.1
    fmt = np.float64
    time_stamps = np.arange(0, 1, 0.1)
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
    np.testing.assert_equal(stream.time_stamps, stream.time_series[:, 0])
    assert np.isdtype(stream.time_series.dtype, np.float64)
    assert stream.nchans == 1
    assert stream.clock_times == clock_times
    assert stream.clock_values == clock_values


def test_mock_stream_default_timeseries_str():
    stream_id = 2
    srate = 10
    tdiff = 0.1
    fmt = "string"
    time_stamps = np.arange(0, 1, 0.1)
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
    assert all([isinstance(sample, list) for sample in stream.time_series])
    assert all([isinstance(sample[0], str) for sample in stream.time_series])
    np.testing.assert_equal(
        stream.time_stamps, [float(sample[0]) for sample in stream.time_series]
    )
    assert stream.nchans == 1
    assert stream.clock_times == clock_times
    assert stream.clock_values == clock_values


@pytest.mark.parametrize("nchans", [1, 2])
def test_mock_stream_timeseries_num(nchans):
    fmt = np.float32
    time_stamps = list(range(0, 10))
    time_series = [[x] * nchans for x in range(10, 20)]
    stream = MockStreamData(
        time_stamps=time_stamps,
        time_series=time_series,
        fmt=fmt,
    )
    assert np.isdtype(stream.time_stamps.dtype, np.float64)
    np.testing.assert_equal(stream.time_stamps, time_stamps)
    assert np.isdtype(stream.time_series.dtype, np.float32)
    np.testing.assert_equal(time_series, stream.time_series)
    assert stream.nchans == nchans


@pytest.mark.parametrize("nchans", [1, 2])
def test_mock_stream_timeseries_str(nchans):
    fmt = "string"
    time_stamps = list(range(0, 10))
    time_series = [[str(x)] * nchans for x in range(10, 20)]
    stream = MockStreamData(
        time_stamps=time_stamps,
        time_series=time_series,
        fmt=fmt,
    )
    assert np.isdtype(stream.time_stamps.dtype, np.float64)
    np.testing.assert_equal(stream.time_stamps, time_stamps)
    np.testing.assert_equal(time_series, stream.time_series)
    assert stream.nchans == nchans
