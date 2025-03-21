import numbers

import numpy as np


class MockStreamData:
    """Mock StreamData class for tests.

    Defines the same attributes as StreamData with basic data
    validation, and tries to ensure convenient defaults so that only
    arguments required for a specific test need to be provided.

    Args:
      stream_id : Stream-id (default: 1)
      srate : Sampling rate (default: None)
        Should be 1/tdiff if tdiff also provided.
      tdiff : Sampling interval (default: None)
        Should be 1/srate if srate also provided.
      fmt : XDF format string denoting channel-format (default "float64")
      time_stamps : 1D array-like of time-stamps (default: [])
      time_series : nD array-like of time-series data (default: None)
        If no time_series is provided time_stamp values are use as a convenient default.
        Number of channels is inferred from the shape of the data.
      clock_times : list of clock offset times (default: None)
      clock_values : list of clock offset values (default: None)
    """

    def __init__(
        self,
        *,
        stream_id=1,
        srate=None,
        tdiff=None,
        fmt="float64",
        time_stamps=[],
        time_series=None,
        clock_times=None,
        clock_values=None,
    ):
        self.stream_id = stream_id
        # Validate srate and tdiff if both provided, otherwise initialise with respect
        # to the supplied parameter
        if srate is not None and tdiff is not None:
            if 1 / srate != tdiff:
                raise ValueError("tdiff not equal to 1/srate")
        elif srate:
            tdiff = 1 / srate
        elif tdiff:
            srate = 1 / tdiff
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
            if fmt == "string":
                if not all([isinstance(sample, list) for sample in time_series]):
                    raise ValueError("All string samples must be lists of strings")
                if not all(
                    [len(sample) == len(time_series[0]) for sample in time_series]
                ):
                    raise ValueError(
                        "All samples must have the same number of channels"
                    )
                if not all(
                    [isinstance(x, str) for sample in time_series for x in sample]
                ):
                    raise ValueError("All string sample values must be strings")
                self.time_series = time_series
                if len(time_series) > 0:
                    self.nchans = len(time_series[0])
                else:
                    self.nchans = 1
            else:
                self.time_series = np.array(time_series, dtype=fmt)
                self.nchans = self.time_series.shape[1]
        # Validate clock offset data
        if clock_times is not None:
            if not isinstance(clock_times, list):
                raise ValueError("Clock times must be a list")
            if not all([isinstance(time, numbers.Number) for time in clock_times]):
                raise ValueError("All clock times must be numeric")
            self.clock_times = clock_times
        else:
            self.clock_times = []
        if clock_values is not None:
            if not isinstance(clock_values, list):
                raise ValueError("Clock values must be a list")
            if not all([isinstance(value, numbers.Number) for value in clock_values]):
                raise ValueError("All clock values must be numeric")
            self.clock_values = clock_values
        else:
            self.clock_values = []
        if clock_times is not None and clock_values is not None:
            # If both clock_times and clock_values are provided they must be the same
            # length
            if len(clock_times) != len(clock_values):
                raise ValueError("Clock times and values must be the same length")
        self.segments = []
        self.clock_segments = []
