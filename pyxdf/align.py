import numpy as np
import warnings
from collections import defaultdict, Counter


def _interpolate(
    x: np.ndarray, y: np.ndarray, new_x: np.ndarray, kind="linear"
) -> np.ndarray:
    """Perform interpolation for _align_timestamps

    If scipy is not installed, the method falls back to numpy, and then only
    supports linear interpolation. Otherwise, it supports  ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    """
    try:
        from scipy.interpolate import interp1d

        f = interp1d(
            x,
            y,
            kind=kind,
            axis=0,
            assume_sorted=True,  # speed up
            bounds_error=False,
        )
        return f(new_x)
    except ImportError as e:
        if kind != "linear":
            raise e
        else:
            return np.interp(new_x, xp=x, fp=y, left=np.NaN, right=np.NaN)


def _shift_align(old_timestamps, old_timeseries, new_timestamps):
    # Convert inputs to numpy arrays
    old_timestamps = np.array(old_timestamps)
    old_timeseries = np.array(old_timeseries)
    new_timestamps = np.array(new_timestamps)

    ts_last = old_timestamps[-1]
    ts_first = old_timestamps[0]

    # Initialize variables
    source = []
    target = []

    new_timeseries = np.full((new_timestamps.shape[0], old_timeseries.shape[1]), np.nan)

    too_old = []
    too_young = []

    # Loop through new timestamps to find the closest old timestamp
    # Handle timestamps outside of the segment (too young or too old) different from stamnps from within the segment
    for nix, nts in enumerate(new_timestamps):
        if nts > ts_last:
            too_young.append((nix, nts))
        elif nts < ts_first:
            too_old.append((nix, nts))
        else:
            closest = np.abs(old_timestamps - nts).argmin()
            if closest not in source:  # Ensure unique mapping
                source.append(closest)
                target.append(nix)
            else:
                raise RuntimeError(
                    f"Non-unique mapping. Closest old timestamp for {new_timestamps[nix]} is {old_timestamps[closest]} but that one was already assigned to {new_timestamps[source.index(closest)]}"
                )

    # Handle too old timestamps (those before the first old timestamp)
    for nix, nts in too_old:
        closest = 0  # Assign to the first timestamp
        if closest not in source:  # Ensure unique mapping
            source.append(closest)
            target.append(nix)
            break  # only one, because we only need the edge

    # Handle too young timestamps (those after the last old timestamp)
    for nix, nts in too_young:
        closest = len(old_timestamps) - 1  # Assign to the last timestamp
        if closest not in source:  # Ensure unique mapping
            source.append(closest)
            target.append(nix)
            break  # only one, because we only need the edge

    # Sanity check: all old timestamps should be assigned to at least one new timestamp
    missed = len(old_timestamps) - len(set(source))
    if missed > 0:
        unassigned_old = [i for i in range(len(old_timestamps)) if i not in source]
        raise RuntimeError(
            f"Too few new timestamps. {missed} old timestamps ({unassigned_old}:{old_timestamps[unassigned_old]}) found no corresponding new timestamp because all were already taken by other old timestamps. If your stream has multiple segments, this might be caused by small differences in effective srate between segments. Try different dejittering thresholds or support your own aligned_timestamps."
        )

    # Populate new timeseries with aligned values from old_timeseries
    for chan in range(old_timeseries.shape[1]):
        new_timeseries[target, chan] = old_timeseries[source, chan]

    return new_timeseries


def align_streams(
    streams,  # List[defaultdict]
    align_foo=dict(),  # defaultdict[int, Callable]
    aligned_timestamps=None,  # Optional[List[float]]
    sampling_rate=None,  # Optional[float|int]
):  # -> Tuple[np.ndarray, List[float]]
    """
    A function to


    Args:

        streams: a list of defaultdicts  (i.e. streams) as returned by
                    load_xdf
        align_foo: a dictionary mapping streamIDs (i.e. int) to interpolation
                    callables. These callables must have the signature
                    `interpolate(old_timestamps, old_timeseries, new_timestamps)` and return a np.ndarray. See `_shift_align` and `_interpolate` for examples.
        aligned_timestamps (optional): a list of floats with the new
                    timestamps to be used for alignment/interpolation. This list of timestamps can be irregular and have gaps.
        sampling_rate (optional): a float defining the sampling rate which
                    will be used to calculate aligned_timestamps.

    Return:
        (aligned_timeseries, aligned_timestamps): tuple


    THe user can define either aligned_timestamps or sampling_rate or neither. If neither is defined, the algorithm will take the sampling_rate of the fastest stream and create aligned_timestamps from the oldest sample of all streams to the youngest.

    """

    if sampling_rate is not None and aligned_timestamps is not None:
        raise ValueError(
            "You can not specify aligned_timestamps and sampling_rate at the same time"
        )

    if sampling_rate is None:
        # we pick the effective sampling rate from the  fastest stream
        srates = [stream["info"]["effective_srate"] for stream in streams]
        sampling_rate = max(srates, default=0)
        if sampling_rate <= 0:  # either no valid stream or all streams are async
            warnings.warn(
                "Can not align streams: Fastest effective sampling rate was 0 step = 1 / sampling_rateor smaller."
            )
            return streams

    if aligned_timestamps is None:
        # we pick the oldest and youngest timestamp of all streams
        stamps = [stream["time_stamps"] for stream in streams]
        ts_first = min((min(s) for s in stamps))
        ts_last = max((max(s) for s in stamps))
        full_dur = (
            ts_last - ts_first + (1 / sampling_rate)
        )  # add one sample to include the last sample (see _jitter_removal)
        # Use np.linspace for precise control over the number of points and guaranteed inclusion of the stop value.
        # np.arange is better when you need direct control over step size but may exclude the stop value and accumulate floating-point errors.
        # Choose np.linspace for better precision and np.arange for efficiency with fixed steps.
        # we create new regularized timestamps
        # arange implementation:
        # step = 1 / sampling_rate
        # aligned_timestamps = np.arange(ts_first, ts_last + step / 2, step)
        # linspace implementation:
        # add 1 to the number of samples to include the last sample
        n_samples = int(np.round((full_dur * sampling_rate), 0)) + 1
        aligned_timestamps = np.linspace(ts_first, ts_last, n_samples)

    channels = 0
    for stream in streams:
        # print(stream)
        channels += int(stream["info"]["channel_count"][0])
    # https://stackoverflow.com/questions/1704823/create-numpy-matrix-filled-with-nans The timings show a preference for ndarray.fill(..) as the faster alternative.
    aligned_timeseries = np.empty(
        (
            len(aligned_timestamps),
            channels,
        ),
        dtype=object,
    )
    aligned_timeseries.fill(np.nan)

    chan_start = 0
    chan_end = 0
    for stream in streams:
        sid = stream["info"]["stream_id"]
        align = align_foo.get(sid, _shift_align)
        chan_cnt = int(stream["info"]["channel_count"][0])
        new_timeseries = np.empty((len(aligned_timestamps), chan_cnt), dtype=object)
        new_timeseries.fill(np.nan)
        print("Stream #", sid, " has ", len(stream["info"]["segments"]), "segments")
        for seg_idx, (seg_start, seg_stop) in enumerate(stream["info"]["segments"]):
            print(seg_idx, ": from index ", seg_start, "to ", seg_stop + 1)
            # segments have been created including the stop index, so we need to add 1 to include the last sample
            segment_old_timestamps = stream["time_stamps"][seg_start : seg_stop + 1]
            segment_old_timeseries = stream["time_series"][seg_start : seg_stop + 1]
            # Sanity check for duplicate timestamps
            if len(np.unique(segment_old_timestamps)) != len(segment_old_timestamps):
                raise RuntimeError("Duplicate timestamps found in old_timestamps")
            # apply align function as defined by the user (or default)
            segment_new_timeseries = align(
                segment_old_timestamps,
                segment_old_timeseries,
                aligned_timestamps,
            )
            # pick indices of the NEW timestamps closest to when segments start and stop
            a = stream["time_stamps"][seg_start]
            b = stream["time_stamps"][seg_stop]
            aix = np.argmin(np.abs(aligned_timestamps - a))
            bix = np.argmin(np.abs(aligned_timestamps - b))
            # and store only this aligned segment, leaving the rest as nans (or aligned as other segments)
            new_timeseries[aix : bix + 1] = segment_new_timeseries[aix : bix + 1]

        # store the new timeseries at the respective channel indices in the 2D array
        chan_start = chan_end
        chan_end += chan_cnt
        aligned_timeseries[:, chan_start:chan_end] = new_timeseries
    return aligned_timeseries, aligned_timestamps
