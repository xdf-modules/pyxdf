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
    old_timestamps = np.array(old_timestamps)
    old_timeseries = np.array(old_timeseries)
    new_timestamps = np.array(new_timestamps)
    ts_last = old_timestamps[-1]
    ts_first = old_timestamps[0]    
    source = list()
    target = list()    
    new_timeseries = np.empty((
                               new_timestamps.shape[0],  # new sample count
                               old_timeseries.shape[1], # old channel count
                               ), dtype=object)
    new_timeseries.fill(np.nan)
    too_old = list()
    too_young = list()
    for nix, nts in enumerate(new_timestamps):
        closest = (np.abs(old_timestamps - nts)).argmin()
        # remember the edge cases, 
        if (nts>ts_last): 
            too_young.append((nix, nts))
        elif (nts < ts_first):
            too_old.append((nix,nts))
        else:
            closest = (np.abs(old_timestamps - nts)).argmin()
            source.append(closest)
            target.append(nix)
    # check the edge cases, 
    for nix, nts in reversed(too_old):
        closest = (np.abs(old_timestamps - nts)).argmin()
        if (closest not in source):
            source.append(closest)
            target.append(nix)
        break
    for nix, nts in too_young:
        closest = (np.abs(old_timestamps - nts)).argmin()
        if (closest not in source):
            source.append(closest)
            target.append(nix)
        break
    
    if len(set(source)) != len(old_timestamps):
        missed = len(old_timestamps)-len(set(source))
        raise RuntimeError(f"Too few new timestamps. {missed} of {len(old_timestamps)} old samples could not be assigned.")
    if len(set(source)) != len(source): #non-unique mapping            
        cnt = Counter(source)        
        toomany = defaultdict(list)
        for v,n in zip(source, target):
            if cnt[v] != 1:
                toomany[old_timestamps[source[v]]].append(new_timestamps[target[n]])
        for k,v in toomany.items():
            print("The old time_stamp ", k,
                "is a closest neighbor of", len(v) ,"new time_stamps:", v)
        raise RuntimeError("Can not align streams. Could not create an unique mapping")
    for chan in range(old_timeseries.shape[1]):
        new_timeseries[target, chan] = old_timeseries[source,chan]
    return new_timeseries


def align_streams(streams, # List[defaultdict]
                  align_foo=dict(), # defaultdict[int, Callable] 
                  aligned_timestamps=None, # Optional[List[float]]
                  sampling_rate=None # Optional[float|int]
): # -> Tuple[np.ndarray, List[float]]
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
        raise ValueError("You can not specify aligned_timestamps and sampling_rate at the same time")
    
    if sampling_rate is None:
         # we pick the effective sampling rate from the  fastest stream
        srates = [stream["info"]["effective_srate"] for stream in streams]
        sampling_rate = max(srates, default=0)
        if sampling_rate <= 0:  # either no valid stream or all streams are async
            warnings.warn("Can not align streams: Fastest effective sampling rate was 0 or smaller.")
            return streams
        
    
    if aligned_timestamps is None:        
        # we pick the oldest and youngest timestamp of all streams
        stamps = [stream["time_stamps"] for stream in streams]        
        ts_first = min((min(s) for s in stamps))      
        ts_last = max((max(s) for s in stamps))  
        full_dur = ts_last-ts_first
        step = 1/sampling_rate
        # we create new regularized timestamps
        aligned_timestamps = np.arange(ts_first, ts_last+step/2, step)
        # using np.linspace only differs in step if n_samples is different (as n_samples must be an integer number (see implementation below). 
        # therefore we stick with np.arange (in spite of possible floating point error accumulation, but to make sure that ts_last is included, we add a half-step. This therefore comes at the cost of a overshoot, but i consider this acceptable considering this stamp would only be from one stream, and not part of all other and therefore is kind of arbitray anyways.
        # linspace implementation:
        # n_samples = int(np.round((full_dur * sampling_rate),0))+1
        # aligned_timestamps = np.linspace(ts_first, ts_last, n_samples)       
        
    channels = 0
    for stream in streams:
        # print(stream)
        channels += int(stream["info"]["channel_count"][0])
    # https://stackoverflow.com/questions/1704823/create-numpy-matrix-filled-with-nans The timings show a preference for ndarray.fill(..) as the faster alternative.
    aligned_timeseries = np.empty((len(aligned_timestamps),
                                   channels,), dtype=object)
    aligned_timeseries.fill(np.nan)

    chan_start = 0    
    chan_end = 0
    for stream in streams:
        sid = stream["info"]["stream_id"]
        align = align_foo.get(sid, _shift_align) 
        chan_cnt = int(stream["info"]["channel_count"][0])
        new_timeseries = np.empty((len(aligned_timestamps), chan_cnt), dtype=object)
        new_timeseries.fill(np.nan)
        for seg_start, seg_stop in stream["info"]["segments"]:            
            _new_timeseries = align(
                stream["time_stamps"][seg_start:seg_stop+1], 
                stream["time_series"][seg_start:seg_stop+1], 
                aligned_timestamps)
            # pick indices of the NEW timestamps closest to when segments start and stop
            a = stream["time_stamps"][seg_start]
            b = stream["time_stamps"][seg_stop]
            aix = np.argmin(np.abs(aligned_timestamps-a))
            bix = np.argmin(np.abs(aligned_timestamps-b))            
            # and store only this aligned segment, leaving the rest as nans (or aligned as other segments)
            new_timeseries[aix:bix+1] = _new_timeseries[aix:bix+1]

        # store the new timeseries at the respective channel indices in the 2D array
        chan_start = chan_end
        chan_end += chan_cnt
        aligned_timeseries[:, chan_start:chan_end] = new_timeseries
    return aligned_timeseries, aligned_timestamps
