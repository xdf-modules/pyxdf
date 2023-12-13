import argparse
import time
from typing import List
from dataclasses import dataclass

import numpy as np
import pyxdf


def _create_info_from_xdf_stream_header(header):
    import pylsl
    new_info = pylsl.StreamInfo(
        name=header["name"][0],
        type=header["type"][0],
        channel_count=int(header["channel_count"][0]),
        nominal_srate=float(header["nominal_srate"][0]),
        channel_format=header["channel_format"][0],
        source_id=header["source_id"][0],
    )
    desc = new_info.desc()
    if "desc" in header and header["desc"][0] is not None:
        _desc = header["desc"][0]
        # https://github.com/sccn/xdf/wiki/EEG-Meta-Data
        if "acquisition" in _desc:
            _acq = _desc["acquisition"][0]
            acq = desc.append_child("acquisition")
            for k in ["manufacturer", "model", "precision", "compensated_lag"]:
                if k in _acq:
                    acq.append_child_value(k, _acq[k][0])
        if "channels" in _desc:
            _chans = _desc["channels"][0]
            chans = desc.append_child("channels")
            for _ch in _chans["channel"]:
                ch = chans.append_child("channel")
                for k in ["label", "unit", "type", "scaling_factor"]:
                    if k in _ch:
                        ch.append_child_value(k, _ch[k][0])
                # loc = ch.append_child("location")
                # for dim_ix, dim in enumerate(["X", "Y", "Z"]):
                #     loc.append_child_value(dim, str(elec_coords[ch_ix, dim_ix]))
    return new_info


def main(fname: str):
    import pylsl
    streams, header = pyxdf.load_xdf(fname)

    @dataclass
    class Streamer:
        stream_ix: int
        name: str
        tvec: np.ndarray
        info: pylsl.StreamInfo
        outlet: pylsl.StreamOutlet
        srate: float

    # First iterate over all streams to calculate some globals.
    xdf_t0 = np.inf
    wrap_dur = 0
    max_rate = 0
    for strm in streams:
        tvec = strm["time_stamps"]
        srate = float(strm["info"]["nominal_srate"][0])
        if len(tvec) > 0:
            xdf_t0 = min(xdf_t0, tvec[0])
            wrap_dur = max(wrap_dur, tvec[-1] - tvec[0])
            if srate != pylsl.IRREGULAR_RATE:
                max_rate = max(max_rate, srate)
                wrap_dur = max(wrap_dur, tvec[-1] - tvec[0] + 1 / srate)

    streamers: List[Streamer] = []
    for strm_ix, strm in enumerate(streams):
        tvec = strm["time_stamps"]
        srate = float(strm["info"]["nominal_srate"][0])
        if len(tvec) > 0:
            new_info: pylsl.StreamInfo = _create_info_from_xdf_stream_header(strm["info"])
            new_outlet: pylsl.StreamOutlet = pylsl.StreamOutlet(new_info)
            streamers.append(Streamer(strm_ix, new_info.name(), tvec - xdf_t0, new_info, new_outlet, srate))

    # Prepare variables to keep track of progress
    start_time = pylsl.local_clock()
    last_time = start_time
    try:
        while True:
            b_wrap = False
            t_now = pylsl.local_clock()

            if (t_now - start_time) > wrap_dur:
                # We have passed the file limit. Trim this iteration until the end of file only.
                t_now = start_time + wrap_dur
                b_wrap = True

            # Get the slice of data (ref t=0) since the last time.
            dat_win_start = last_time - start_time
            dat_win_stop = t_now - start_time

            for streamer in streamers:
                b_dat = np.logical_and(
                    streamer.tvec >= dat_win_start,
                    streamer.tvec < dat_win_stop
                )
                if np.any(b_dat):
                    if streamer.srate > 0:
                        streamer.outlet.push_chunk(streams[streamer.stream_ix]["time_series"][b_dat],
                                                   timestamp=start_time + streamer.tvec[b_dat][-1])
                    else:
                        # Irregular rate, like events and markers
                        for dat_idx in np.where(b_dat)[0]:
                            sample = streams[streamer.stream_ix]["time_series"][dat_idx]
                            streamer.outlet.push_sample(sample,
                                                        timestamp=start_time + streamer.tvec[dat_idx])
                            # print(f"Pushed sample: {sample}")

            last_time = t_now
            if b_wrap:
                start_time = t_now

            if max_rate > 0:
                if (pylsl.local_clock() - last_time) < (1 / max_rate):
                    # Sleep until we have at least 1 more sample of the fastest stream.
                    time.sleep(1 / max_rate)
            else:
                # All our streams are irregular. Sleep a constant 5 msec.
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("TODO: Shutdown outlets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playback an XDF file over LSL streams.")
    parser.add_argument(
        "filename",
        type=str,
        help="Path to the XDF file"
    )
    args = parser.parse_args()
    main(args.filename)
