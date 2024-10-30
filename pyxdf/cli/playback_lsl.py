import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pylsl

import pyxdf


def _create_info_from_xdf_stream_header(header):
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


@dataclass
class Streamer:
    stream_ix: int
    name: str
    tvec: np.ndarray
    info: pylsl.StreamInfo
    outlet: pylsl.StreamOutlet
    srate: float


class LSLPlaybackClock:
    def __init__(
        self,
        rate: float = 1.0,
        loop_time: float = 0.0,
        max_sample_rate: Optional[float] = None,
    ):
        if rate != 1.0:
            print(
                "WARNING!! rate != 1.0; it is impossible to synchronize playback "
                "streams with real time streams."
            )
        self.rate: float = rate  # Maximum rate is loop_time / avg_update_interval
        self._boundary = loop_time
        self._max_srate = max_sample_rate
        decr = (1 / self._max_srate) if self._max_srate else 2 * sys.float_info.epsilon
        self._wall_start: float = pylsl.local_clock() - decr / 2
        self._file_read_s: float = 0  # File read header in seconds
        self._prev_file_read_s: float = (
            0  # File read header in seconds for previous iteration
        )
        self._n_loop: int = 0

    def reset(self, reset_file_position: bool = False) -> None:
        decr = (1 / self._max_srate) if self._max_srate else 2 * sys.float_info.epsilon
        self._wall_start = (
            pylsl.local_clock() - decr / 2 - self._file_read_s / self.rate
        )
        self._n_loop = 0
        if reset_file_position:
            self._file_read_s = 0
            self._prev_file_read_s = 0

    def set_rate(self, rate: float) -> None:
        self.rate = rate
        # Note: We do not update file_read_s and prev_file_read_s.
        # Changing the playback rate does not change where we are in the file.
        self.reset(reset_file_position=False)

    def update(self):
        self._prev_file_read_s = self._file_read_s
        wall_elapsed = pylsl.local_clock() - self._wall_start
        _file_read_s = self.rate * wall_elapsed
        if self._boundary and self._prev_file_read_s == self._boundary:
            # Previous iteration ended at the file boundary; wrap around and reset.
            self._prev_file_read_s = 0.0
            self._n_loop += 1
        overrun = self._n_loop * self._boundary if self._boundary else 0
        self._file_read_s = _file_read_s - overrun
        if self._boundary and self._file_read_s >= self._boundary:
            # Previous was below boundary, now above boundary.
            # Truncate _file_read_s to align exactly on the boundary;
            # we will loop on the next iteration.
            self._file_read_s = self._boundary

    @property
    def step_range(self) -> tuple[float, float]:
        return self._prev_file_read_s, self._file_read_s

    @property
    def t0(self) -> float:
        return self._wall_start + self._n_loop * self._boundary

    def sleep(self, duration: Optional[float] = None) -> None:
        if duration is None:
            if self._max_srate <= 0:
                duration = 0.005
            else:
                # Check to see if the current time is not already beyond the expected
                # time of the next iteration.
                step_time = 1 / self._max_srate
                now_read_s = self.rate * (pylsl.local_clock() - self._wall_start)
                next_read_s = self._file_read_s + step_time
                duration = max(next_read_s - now_read_s, 0)
        time.sleep(duration / self.rate)


def main(
    fname: str,
    playback_speed: float = 1.0,
    loop: bool = False,
    wait_for_consumer: bool = False,
):
    streams, _ = pyxdf.load_xdf(fname)

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

    # Create list of Streamer objects
    streamers: List[Streamer] = []
    for strm_ix, strm in enumerate(streams):
        tvec = strm["time_stamps"]
        srate = float(strm["info"]["nominal_srate"][0])
        if len(tvec) > 0:
            new_info: pylsl.StreamInfo = _create_info_from_xdf_stream_header(
                strm["info"]
            )
            new_outlet: pylsl.StreamOutlet = pylsl.StreamOutlet(new_info)
            streamers.append(
                Streamer(
                    strm_ix, new_info.name(), tvec - xdf_t0, new_info, new_outlet, srate
                )
            )

    # Create timer to manage playback.
    timer = LSLPlaybackClock(
        rate=playback_speed,
        loop_time=wrap_dur if loop else None,
        max_sample_rate=max_rate,
    )
    read_heads = {_.name: 0 for _ in streamers}
    b_push = not wait_for_consumer  # A flag to indicate we can push samples.
    try:
        while True:
            if not b_push:
                # We are looking for consumers.
                time.sleep(0.01)
                have_consumers = [
                    streamer.outlet.have_consumers() for streamer in streamers
                ]
                # b_push = any(have_consumers)
                b_push = all(have_consumers)
                if b_push:
                    timer.reset()
                else:
                    continue
            timer.update()
            t_start, t_stop = timer.step_range
            all_streams_exhausted = True
            for streamer in streamers:
                start_idx = read_heads[streamer.name] if t_start > 0 else 0
                stop_idx = np.searchsorted(streamer.tvec, t_stop)
                if stop_idx > start_idx:
                    all_streams_exhausted = False
                    if streamer.srate > 0:
                        sl = np.s_[start_idx:stop_idx]
                        push_dat = streams[streamer.stream_ix]["time_series"][sl]
                        push_ts = timer.t0 + streamer.tvec[sl][-1]
                        streamer.outlet.push_chunk(push_dat, timestamp=push_ts)
                    else:
                        # Irregular rate, like events and markers
                        for dat_idx in range(start_idx, stop_idx):
                            sample = streams[streamer.stream_ix]["time_series"][dat_idx]
                            streamer.outlet.push_sample(
                                sample, timestamp=timer.t0 + streamer.tvec[dat_idx]
                            )
                            # print(f"Pushed sample: {sample}")
                    read_heads[streamer.name] = stop_idx

            if not loop and all_streams_exhausted:
                print("Playback finished.")
                break
            timer.sleep()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Deleting outlets...")
        for streamer in streamers:
            del streamer.outlet
        print("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Playback an XDF file over LSL streams."
    )
    parser.add_argument("filename", type=str, help="Path to the XDF file.")
    parser.add_argument(
        "--playback_speed", type=float, default=1.0, help="Playback speed multiplier."
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop playback of the file."
    )
    parser.add_argument(
        "--wait_for_consumer",
        action="store_true",
        help="Wait for consumer before starting playback.",
    )
    args = parser.parse_args()
    main(
        args.filename,
        playback_speed=args.playback_speed,
        loop=args.loop,
        wait_for_consumer=args.wait_for_consumer,
    )
