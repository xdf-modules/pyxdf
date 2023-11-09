# Authors: Christian Kothe & the Intheon pyxdf team
#          Chadwick Boulay
#          Tristan Stenner
#          Clemens Brunner
#
# License: BSD (2-clause)

"""Defines the function load_xdf, which imports XDF files.

This function is closely following the load_xdf reference implementation.
"""
import io
import struct
import itertools
import gzip
from xml.etree.ElementTree import fromstring
from collections import OrderedDict, defaultdict
import logging
from pathlib import Path
import warnings
import numpy as np


__all__ = ["load_xdf"]

logger = logging.getLogger(__name__)


class StreamData:
    """Temporary per-stream data."""

    def __init__(self, xml):
        """Init a new StreamData object from a stream header."""
        fmts = dict(
            double64=np.float64,
            float32=np.float32,
            string=object,
            int32=np.int32,
            int16=np.int16,
            int8=np.int8,
            int64=np.int64,
        )
        # number of channels
        self.nchns = int(xml["info"]["channel_count"][0])
        # nominal sampling rate in Hz
        self.srate = float(xml["info"]["nominal_srate"][0])
        # format string (int8, int16, int32, float32, double64, string)
        self.fmt = xml["info"]["channel_format"][0]
        # list of time-stamp chunks (each an ndarray, in seconds)
        self.time_stamps = []
        # list of time-series chunks (each an ndarray or list of lists)
        self.time_series = []
        # list of clock offset measurement times (in seconds)
        self.clock_times = []
        # list of clock offset measurement values (in seconds)
        self.clock_values = []
        # last observed time stamp, for delta decompression
        self.last_timestamp = 0.0
        # nominal sampling interval, in seconds, for delta decompression
        self.tdiff = 1.0 / self.srate if self.srate > 0 else 0.0
        self.effective_srate = 0.0
        # pre-calc some parsing parameters for efficiency
        if self.fmt != "string":
            self.dtype = np.dtype(fmts[self.fmt])
            # number of bytes to read from stream to handle one sample
            self.samplebytes = self.nchns * self.dtype.itemsize


def load_xdf(
    filename,
    select_streams=None,
    *,
    on_chunk=None,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=True,
    jitter_break_threshold_seconds=1,
    jitter_break_threshold_samples=500,
    clock_reset_threshold_seconds=5,
    clock_reset_threshold_stds=5,
    clock_reset_threshold_offset_seconds=1,
    clock_reset_threshold_offset_stds=10,
    winsor_threshold=0.0001,
    verbose=None,
    sync_timestamps=False,
    use_samplingrate="highest",
    overlap_timestamps=False,
):
    """Import an XDF file.

    This is an importer for multi-stream XDF (Extensible Data Format)
    recordings. All information covered by the XDF 1.0 specification is
    imported, plus any additional meta-data associated with streams or with the
    container file itself.

    See https://github.com/sccn/xdf/ for more information on XDF.

    The function supports several additional features, such as robust time
    synchronization, support for breaks in the data, as well as some other
    defects.

    Args:
        filename : Name of the file to import (*.xdf or *.xdfz).

        select_streams : int | list[int] | list[dict] | None
          One or more stream IDs to load. Accepted values are:
          - int or list[int]: load only specified stream IDs, e.g.
            select_streams=5 loads only the stream with stream ID 5, whereas
            select_streams=[2, 4] loads only streams with stream IDs 2 and 4.
          - list[dict]: load only streams matching a query, e.g.
            select_streams=[{'type': 'EEG'}] loads all streams of type 'EEG'.
            Entries within a dict must all match a stream, e.g.
            select_streams=[{'type': 'EEG', 'name': 'TestAMP'}] matches streams
            with both type 'EEG' *and* name 'TestAMP'. If
            select_streams=[{'type': 'EEG'}, {'name': 'TestAMP'}], streams
            matching either the type *or* the name will be loaded.
          - None: load all streams (default).

        verbose : Passing True will set logging level to DEBUG, False will set
          it to WARNING, and None will use root logger level. (default: None)

        synchronize_clocks : Whether to enable clock synchronization based on
          ClockOffset chunks. (default: true)

        dejitter_timestamps : Whether to perform jitter removal for regularly
          sampled streams. (default: true)

        sync_timestamps: {bool str}
            sync timestamps of all streams sample-wise with the stream of the
            highest effective sampling rate (or as set in use_samplingrate). Using sync_timestamps with any
            method other than linear has dependency on scipy, which is not a
            hard requirement of pyxdf. If scipy is not installed in your
            environment, the method supports linear interpolation with
            numpy.

            False -> no syncing
            True -> linear syncing
            str:<'linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
            ‘previous’, ‘next’> for method inherited from
            scipy.interpolate.interp1d.

        use_samplingrate: {int str} = "highest"
            resamples all channels to an effective sampling rate as given by this argument, using the method as defined by sync_timestamps. Will do nothing if sync_timestamps is False or the sampling rate is not larger than zero.
            int>0: resample to this sampling rate.
            str:<'highest'> use the highest

        overlap_timestamps: bool
            If true, return only overlapping streams, i.e. all streams
            are limited to periods where all streams have data. (default: True)
            If false, depends on whether sync_timestamps is set. If set, it
            expands all streams to include the earliest and latest timestamp of
            any stream, if not set it simply return streams independently.
            `
        on_chunk : Function that is called for each chunk of data as it is
           being retrieved from the file; the function is allowed to modify
           the data (for example, sub-sample it). The four input arguments
           are (1) the matrix of [#channels x #samples] values (either numeric
           or 2D array of strings), (2) the vector of unprocessed local time
           stamps (one per sample), (3) the info struct for the stream (same as
           the .info field in the final output, but without the
           .effective_srate sub-field), and (4) the scalar stream number
           (1-based integers). The three return values are (1) the (optionally
           modified) data, (2) the (optionally modified) time stamps, and (3)
           the (optionally modified) header. (default: [])

        Parameters for advanced failure recovery in clock synchronization:

        handle_clock_resets : Whether the importer should check for potential
          resets of the clock of a stream (e.g. computer restart during
          recording, or hot-swap). Only useful if the recording system supports
          recording under such circumstances. (default: true)

        clock_reset_threshold_stds : A clock reset must be accompanied by a
          ClockOffset chunk being delayed by at least this many standard
          deviations from the distribution. (default: 5)

        clock_reset_threshold_seconds : A clock reset must be accompanied by a
          ClockOffset chunk being delayed by at least this many seconds.
          (default: 5)

        clock_reset_threshold_offset_stds : A clock reset must be accompanied
          by a ClockOffset difference that lies at least this many standard
          deviations from the distribution. (default: 10)

        clock_reset_threshold_offset_seconds : A clock reset must be
          accompanied by a ClockOffset difference that is at least this many
          seconds away from the median. (default: 1)

        winsor_threshold : A threshold above which the clock offsets will be
          treated robustly (i.e., like outliers), in seconds. (default: 0.0001)

        Parameters for jitter removal in the presence of data breaks:

        jitter_break_threshold_seconds : An interruption in a regularly-sampled
          stream of at least this many seconds will be considered as a
          potential break (if also the jitter_break_threshold_samples is
          crossed) and multiple segments will be returned. (default: 1)

        jitter_break_threshold_samples : An interruption in a regularly-sampled
          stream of at least this many samples will be considered as a
          potential break (if also the jitter_break_threshold_samples is
          crossed) and multiple segments will be returned. (default: 500)

    Returns:
        streams : list[dict] (one dict for each stream)
          Dicts have the following content:
          - 'time_series': Contains the time series as a [#Channels x #Samples]
            array of the type declared in ['info']['channel_format'].
          - 'time_stamps': Contains the time stamps for each sample (synced
            across streams).
          - 'info': Contains the meta-data of the stream (all values are
            strings).
          - 'name': Name of the stream.
          - 'type': Content type of the stream ('EEG', 'Events', ...).
          - 'channel_format': Value format ('int8', 'int16', 'int32', 'int64',
            'float32', 'double64', 'string').
          - 'nominal_srate': Nominal sampling rate of the stream (as declared
            by the device); zero for streams with irregular sampling rate.
          - 'effective_srate': Effective (measured) sampling rate of the
            stream if regular (otherwise omitted).
          - 'desc': Dict with any domain-specific meta-data.

        fileheader : Dict with file header contents in the 'info' field.

    Examples:
        >>> streams, fileheader = load_xdf('myrecording.xdf')
    """
    if verbose is not None:
        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    logger.info("Importing XDF file %s..." % filename)

    # if select_streams is an int or a list of int, load only streams
    # associated with the corresponding stream IDs
    # if select_streams is a list of dicts, use this to query and load streams
    # associated with these properties
    if select_streams is None:
        pass
    elif isinstance(select_streams, int):
        select_streams = [select_streams]
    elif all([isinstance(elem, dict) for elem in select_streams]):
        select_streams = match_streaminfos(
            resolve_streams(filename), select_streams
        )
        if not select_streams:  # no streams found
            raise ValueError("No matching streams found.")
    elif not all([isinstance(elem, int) for elem in select_streams]):
        raise ValueError(
            "Argument 'select_streams' must be an int, a list of ints or a "
            "list of dicts."
        )

    # dict of returned streams, in order of appearance, indexed by stream id
    streams = OrderedDict()
    # dict of per-stream temporary data (StreamData), indexed by stream id
    temp = {}
    # XML content of the file header chunk
    fileheader = None

    with open_xdf(filename) as f:
        # for each chunk
        while True:
            # noinspection PyBroadException
            try:
                # read [NumLengthBytes], [Length]
                chunklen = _read_varlen_int(f)
            except EOFError:
                break
            except Exception:
                logger.exception("Error reading chunk length")
                # If there's more data available (i.e. a read() succeeds),
                # find the next boundary chunk
                if f.read(1):
                    logger.warning(
                        "got zero-length chunk, scanning forward to next "
                        "boundary chunk."
                    )
                    # move the stream position one byte back
                    f.seek(-1, 1)
                    if _scan_forward(f):
                        continue
                logger.info("  reached end of file.")
                break

            # read [Tag]
            tag = struct.unpack("<H", f.read(2))[0]
            log_str = " Read tag: {} at {} bytes, length={}"
            log_str = log_str.format(tag, f.tell(), chunklen)
            StreamId = None
            if tag in [2, 3, 4, 6]:
                _streamid = f.read(4)
                try:
                    StreamId = struct.unpack("<I", _streamid)[0]
                except struct.error:
                    # we scan forward to next (hopefully) valid block in a bid
                    # to load as much of the file as possible If the StreamId
                    # could not be parsed correctly, it will be None. We
                    # therefore also need to continue, because otherwise we
                    # might end up in one the tag-specific branches which
                    # expect a valid StreamId
                    log_str += (
                        ", StreamId is corrupt, scanning forward to next "
                        "boundary chunk."
                    )
                    logger.error(log_str)
                    _scan_forward(f)
                    continue
                else:
                    # to be executed if no exception was raised
                    log_str += ", StreamId={}".format(StreamId)
                    logger.debug(log_str)

            if StreamId is not None and select_streams is not None:
                if StreamId not in select_streams:
                    f.read(chunklen - 2 - 4)  # skip remaining chunk contents
                    continue

            # read the chunk's [Content]...
            if tag == 1:
                # read [FileHeader] chunk
                xml_string = f.read(chunklen - 2)
                fileheader = _xml2dict(fromstring(xml_string))
            elif tag == 2:
                # read [StreamHeader] chunk...
                # read [Content]
                xml_string = f.read(chunklen - 6)
                decoded_string = xml_string.decode("utf-8", "replace")
                hdr = _xml2dict(fromstring(decoded_string))
                streams[StreamId] = hdr
                logger.debug("  found stream " + hdr["info"]["name"][0])
                # initialize per-stream temp data
                temp[StreamId] = StreamData(hdr)
            elif tag == 3:
                # read [Samples] chunk...
                # noinspection PyBroadException
                try:
                    nsamples, stamps, values = _read_chunk3(f, temp[StreamId])
                    logger.debug(
                        "  reading [%s,%s]" % (temp[StreamId].nchns, nsamples)
                    )
                    # optionally send through the on_chunk function
                    if on_chunk is not None:
                        values, stamps, streams[StreamId] = on_chunk(
                            values, stamps, streams[StreamId], StreamId
                        )
                    # append to the time series...
                    temp[StreamId].time_series.append(values)
                    temp[StreamId].time_stamps.append(stamps)
                except Exception as e:
                    # an error occurred (perhaps a chopped-off file): emit a
                    # warning and scan forward to the next recognized chunk
                    logger.error(
                        "found likely XDF file corruption (%s), "
                        "scanning forward to next boundary chunk." % e
                    )
                    _scan_forward(f)
            elif tag == 6:
                # read [StreamFooter] chunk
                xml_string = f.read(chunklen - 6)
                streams[StreamId]["footer"] = _xml2dict(fromstring(xml_string))
            elif tag == 4:
                # read [ClockOffset] chunk
                temp[StreamId].clock_times.append(
                    struct.unpack("<d", f.read(8))[0]
                )
                temp[StreamId].clock_values.append(
                    struct.unpack("<d", f.read(8))[0]
                )
            else:
                # skip other chunk types (Boundary, ...)
                f.read(chunklen - 2)

    # Concatenate the signal across chunks
    for stream in temp.values():
        if stream.time_stamps:
            # stream with non-empty list of chunks
            stream.time_stamps = np.concatenate(stream.time_stamps)
            if stream.fmt == "string":
                stream.time_series = list(itertools.chain(*stream.time_series))
            else:
                stream.time_series = np.concatenate(stream.time_series)
        else:
            # stream without any chunks
            stream.time_stamps = np.zeros((0,))
            if stream.fmt == "string":
                stream.time_series = []
            else:
                stream.time_series = np.zeros((stream.nchns, 0))

    # perform (fault-tolerant) clock synchronization if requested
    if synchronize_clocks:
        logger.info("  performing clock synchronization...")
        temp = _clock_sync(
            temp,
            handle_clock_resets,
            clock_reset_threshold_stds,
            clock_reset_threshold_seconds,
            clock_reset_threshold_offset_stds,
            clock_reset_threshold_offset_seconds,
            winsor_threshold,
        )

    # perform jitter removal if requested
    if dejitter_timestamps:
        logger.info("  performing jitter removal...")
        temp = _jitter_removal(
            temp,
            jitter_break_threshold_seconds,
            jitter_break_threshold_samples,
        )
    else:
        for stream in temp.values():
            if len(stream.time_stamps) > 1:
                duration = stream.time_stamps[-1] - stream.time_stamps[0]
                stream.effective_srate = len(stream.time_stamps) / duration
            else:
                stream.effective_srate = 0.0

    # sync sampling with the fastest timeseries by interpolation / shifting
    for k in streams.keys():
        stream = streams[k]
        tmp = temp[k]
        tmp.channel_format = stream["info"]["channel_format"][0]

    if sync_timestamps:
        if type(sync_timestamps) is not str:
            sync_timestamps = "linear"
            logger.warning('sync_timestamps defaults to "linear"')
        temp = _sync_timestamps(
            temp, kind=sync_timestamps, use_samplingrate=use_samplingrate
        )

    # limit streams to their overlapping periods
    if overlap_timestamps:
        temp = _limit_streams_to_overlap(temp)

    for k in streams.keys():
        stream = streams[k]
        tmp = temp[k]
        if "stream_id" in stream["info"]:  # this is non-standard
            logger.warning(
                "Found existing 'stream_id' key with value {} in "
                "StreamHeader XML. Using the 'stream_id' value {} "
                "from the beginning of the StreamHeader chunk "
                "instead.".format(stream["info"]["stream_id"], k)
            )
        stream["info"]["stream_id"] = k
        stream["info"]["effective_srate"] = tmp.effective_srate
        stream["time_series"] = tmp.time_series
        stream["time_stamps"] = tmp.time_stamps
    streams = [s for s in streams.values()]
    return streams, fileheader


def open_xdf(file):
    """Open XDF file for reading.
    :type file: str | pathlib.Path | io.RawIOBase
        File name or already opened file
    """

    if isinstance(file, (io.RawIOBase, io.BufferedIOBase)):
        if isinstance(file, io.TextIOBase):
            raise ValueError("file has to be opened in binary mode")
        f = file
    else:
        filename = Path(file)  # ensure convert to pathlib object
        # check absolute path after following symlinks
        if not filename.resolve().exists():
            raise Exception("file %s does not exist." % filename)

        if filename.suffix == ".xdfz" or filename.suffixes == [".xdf", ".gz"]:
            f = gzip.open(str(filename), "rb")
        else:
            f = open(str(filename), "rb")
    if f.read(4) != b"XDF:":  # magic bytes
        raise IOError("Invalid XDF file {}".format(file))
    return f


def _read_chunk3(f, s):
    # read [NumSampleBytes], [NumSamples]
    nsamples = _read_varlen_int(f)
    # allocate space
    stamps = np.zeros((nsamples,))
    if s.fmt == "string":
        # read a sample comprised of strings
        values = [[None] * s.nchns for _ in range(nsamples)]
        # for each sample...
        for k in range(nsamples):
            # read or deduce time stamp
            if f.read(1) != b"\x00":
                stamps[k] = struct.unpack("<d", f.read(8))[0]
            else:
                stamps[k] = s.last_timestamp + s.tdiff
            s.last_timestamp = stamps[k]
            # read the values
            for ch in range(s.nchns):
                raw = f.read(_read_varlen_int(f))
                values[k][ch] = raw.decode(errors="replace")
    else:
        # read a sample comprised of numeric values
        values = np.zeros((nsamples, s.nchns), dtype=s.dtype)
        # read buffer
        raw = bytearray(s.nchns * values.dtype.itemsize)
        # for each sample...
        for k in range(values.shape[0]):
            # read or deduce time stamp
            if f.read(1) != b"\x00":
                stamps[k] = struct.unpack("<d", f.read(8))[0]
            else:
                stamps[k] = s.last_timestamp + s.tdiff
            s.last_timestamp = stamps[k]
            # read the values
            f.readinto(raw)
            # no fromfile(), see
            # https://github.com/numpy/numpy/issues/13319
            values[k, :] = np.frombuffer(raw, dtype=s.dtype, count=s.nchns)
    return nsamples, stamps, values


_read_varlen_int_buf = bytearray(1)


def _read_varlen_int(f):
    """Read a variable-length integer."""
    if not f.readinto(_read_varlen_int_buf):
        raise EOFError()
    nbytes = _read_varlen_int_buf[0]
    if nbytes == 1:
        return ord(f.read(1))
    elif nbytes == 4:
        return struct.unpack("<I", f.read(4))[0]
    elif nbytes == 8:
        return struct.unpack("<Q", f.read(8))[0]
    else:
        raise RuntimeError("invalid variable-length integer encountered.")


def _xml2dict(t):
    """Convert an attribute-less etree.Element into a dict."""
    dd = defaultdict(list)
    for dc in map(_xml2dict, list(t)):
        for k, v in dc.items():
            dd[k].append(v)
    return {t.tag: dd or t.text}


def _scan_forward(f):
    """Scan forward through file object until after the next boundary chunk."""
    blocklen = 2**20
    signature = bytes(
        [
            0x43,
            0xA5,
            0x46,
            0xDC,
            0xCB,
            0xF5,
            0x41,
            0x0F,
            0xB3,
            0x0E,
            0xD5,
            0x46,
            0x73,
            0x83,
            0xCB,
            0xE4,
        ]
    )
    while True:
        curpos = f.tell()
        block = f.read(blocklen)
        matchpos = block.find(signature)
        if matchpos != -1:
            f.seek(curpos + matchpos + len(signature))
            logger.debug("  scan forward found a boundary chunk.")
            return True
        if len(block) < blocklen:
            logger.debug("  scan forward reached end of file with no match.")
            return False


def _clock_sync(
    streams,
    handle_clock_resets=True,
    reset_threshold_stds=5,
    reset_threshold_seconds=5,
    reset_threshold_offset_stds=10,
    reset_threshold_offset_seconds=1,
    winsor_threshold=0.0001,
):
    for stream in streams.values():
        if len(stream.time_stamps) > 0:
            clock_times = stream.clock_times
            clock_values = stream.clock_values
            if not clock_times:
                continue

            # Detect clock resets (e.g., computer restarts during recording) if
            # requested, this is only for cases where "everything goes wrong"
            # during recording note that this is a fancy feature that is not
            # needed for normal XDF compliance.
            if handle_clock_resets and len(clock_times) > 1:
                # First detect potential breaks in the synchronization data;
                # this is only necessary when the importer should be able to
                # deal with recordings where the computer that served a stream
                # was restarted or hot-swapped during an ongoing recording, or
                # the clock was reset otherwise.

                time_diff = np.diff(clock_times)
                value_diff = np.abs(np.diff(clock_values))
                median_ival = np.median(time_diff)
                median_slope = np.median(value_diff)

                # points where a glitch in the timing of successive clock
                # measurements happened
                mad = (
                    np.median(np.abs(time_diff - median_ival))
                    + np.finfo(float).eps
                )
                cond1 = time_diff < 0
                cond2 = (time_diff - median_ival) / mad > reset_threshold_stds
                cond3 = time_diff - median_ival > reset_threshold_seconds
                time_glitch = cond1 | (cond2 & cond3)

                # Points where a glitch in successive clock value estimates
                # happened
                mad = (
                    np.median(np.abs(value_diff - median_slope))
                    + np.finfo(float).eps
                )
                cond1 = value_diff < 0
                cond2 = (
                    value_diff - median_slope
                ) / mad > reset_threshold_offset_stds
                cond3 = (
                    value_diff - median_slope > reset_threshold_offset_seconds
                )
                value_glitch = cond1 | (cond2 & cond3)
                resets_at = time_glitch & value_glitch

                # Determine the [begin,end] index ranges between resets
                if not any(resets_at):
                    ranges = [(0, len(clock_times) - 1)]
                else:
                    indices = np.where(resets_at)[0]
                    indices = np.hstack(
                        (0, indices, indices + 1, len(resets_at) - 1)
                    )
                    ranges = np.reshape(indices, (2, -1)).T

            # Otherwise we just assume that there are no clock resets
            else:
                ranges = [(0, len(clock_times) - 1)]

            # Calculate clock offset mappings for each data range
            coef = []
            for range_i in ranges:
                if range_i[0] != range_i[1]:
                    start, stop = range_i[0], range_i[1] + 1
                    X = np.column_stack(
                        [
                            np.ones((stop - start,)),
                            np.array(clock_times[start:stop])
                            / winsor_threshold,
                        ]
                    )
                    y = np.array(clock_values[start:stop]) / winsor_threshold
                    # noinspection PyTypeChecker
                    _coefs = _robust_fit(X, y)
                    _coefs[0] *= winsor_threshold
                    coef.append(_coefs)
                else:
                    coef.append((clock_values[range_i[0]], 0))

            # Apply the correction to all time stamps
            if len(ranges) == 1:
                stream.time_stamps += coef[0][0] + (
                    coef[0][1] * stream.time_stamps
                )
            else:
                for coef_i, range_i in zip(coef, ranges):
                    r = slice(range_i[0], range_i[1])
                    stream.time_stamps[r] += (
                        coef_i[0] + coef_i[1] * stream.time_stamps[r]
                    )
    return streams


def _jitter_removal(streams, threshold_seconds=1, threshold_samples=500):
    for stream_id, stream in streams.items():
        stream.effective_srate = 0  # will be recalculated if possible
        nsamples = len(stream.time_stamps)
        if nsamples > 0 and stream.srate > 0:
            # Identify breaks in the time_stamps
            diffs = np.diff(stream.time_stamps)
            b_breaks = diffs > np.max(
                (threshold_seconds, threshold_samples * stream.tdiff)
            )
            # find indices (+ 1 to compensate for lost sample in np.diff)
            break_inds = np.where(b_breaks)[0] + 1

            # Get indices delimiting segments without breaks
            # 0th sample is a segment start and last sample is a segment stop
            seg_starts = np.hstack(([0], break_inds))
            seg_stops = np.hstack((break_inds - 1, nsamples - 1))  # inclusive
            stream.seg_starts = seg_starts
            stream.seg_stops = seg_stops
            # Process each segment separately
            for start_ix, stop_ix in zip(seg_starts, seg_stops):
                # Calculate time stamps assuming constant intervals within each
                # segment (stop_ix + 1 because we want inclusive closing range)
                idx = np.arange(start_ix, stop_ix + 1, 1)[:, None]
                X = np.concatenate((np.ones_like(idx), idx), axis=1)
                y = stream.time_stamps[idx]
                mapping = np.linalg.lstsq(X, y, rcond=-1)[0]
                stream.time_stamps[idx] = mapping[0] + mapping[1] * idx

            # Recalculate effective_srate if possible
            counts = (seg_stops + 1) - seg_starts
            if np.any(counts):
                # Calculate range segment duration (assuming last sample
                # duration was exactly 1 * stream.tdiff)
                durations = (
                    stream.time_stamps[seg_stops] + stream.tdiff
                ) - stream.time_stamps[seg_starts]
                stream.effective_srate = np.sum(counts) / np.sum(durations)

        srate, effective_srate = stream.srate, stream.effective_srate
        if srate != 0 and np.abs(srate - effective_srate) / srate > 0.1:
            msg = (
                "Stream %d: Calculated effective sampling rate %.4f Hz is"
                " different from specified rate %.4f Hz."
            )
            logger.warning(msg, stream_id, effective_srate, srate)

    return streams


# noinspection PyTypeChecker
def _robust_fit(A, y, rho=1, iters=1000):
    """Perform a robust linear regression using the Huber loss function.

    solves the following problem via ADMM for x:
        minimize 1/2*sum(huber(A*x - y))

    Args:
        A : design matrix
        y : target variable
        rho : augmented Lagrangian variable (default: 1)
        iters : number of iterations to perform (default: 1000)

    Returns:
        x : solution for x

    Based on the ADMM Matlab codes also found at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

    """
    A = np.copy(A)  # Don't mutate input.
    offset = np.min(A[:, 1])
    A[:, 1] -= offset
    Aty = np.dot(A.T, y)
    L = np.linalg.cholesky(np.dot(A.T, A))
    U = L.T
    z = np.zeros_like(y)
    u = z
    x = z
    for k in range(iters):
        x = np.linalg.solve(U, (np.linalg.solve(L, Aty + np.dot(A.T, z - u))))
        d = np.dot(A, x) - y + u
        d_inv = np.divide(1, d, where=d != 0)
        tmp = np.maximum(0, (1 - (1 + 1 / rho) * np.abs(d_inv)))
        z = rho / (1 + rho) * d + 1 / (1 + rho) * tmp * d
        u = d - z
    x[0] -= x[1] * offset
    return x


def match_streaminfos(stream_infos, parameters):
    """Find stream IDs matching specified criteria.

    Parameters
    ----------
    stream_infos : list of dicts
        List of dicts containing information on each stream. This information
        can be obtained using the function resolve_streams.
    parameters : list of dicts
        List of dicts containing key/values that should be present in streams.
        Examples: [{"name": "Keyboard"}] matches all streams with a "name"
                  field equal to "Keyboard".
                  [{"name": "Keyboard"}, {"type": "EEG"}] matches all streams
                  with a "name" field equal to "Keyboard" and all streams with
                  a "type" field equal to "EEG".
    """
    matches = []
    match = False
    for request in parameters:
        for info in stream_infos:
            for key in request.keys():
                match = info[key] == request[key]
                if not match:
                    break
            if match:
                matches.append(info["stream_id"])

    return list(set(matches))  # return unique values


def resolve_streams(fname):
    """Resolve streams in given XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    stream_infos : list of dicts
        List of dicts containing information on each stream.
    """
    return parse_chunks(parse_xdf(fname))


def parse_xdf(fname):
    """Parse and return chunks contained in an XDF file.

    Parameters
    ----------
    fname : str
        Name of the XDF file.

    Returns
    -------
    chunks : list
        List of all chunks contained in the XDF file.
    """
    chunks = []
    with open_xdf(fname) as f:
        for chunk in _read_chunks(f):
            chunks.append(chunk)
    return chunks


def parse_chunks(chunks):
    """Parse chunks and extract information on individual streams."""
    streams = []
    for chunk in chunks:
        if chunk["tag"] == 2:  # stream header chunk
            # if you edit, check for consistency with parsing in load_xdf
            streams.append(
                dict(
                    stream_id=chunk["stream_id"],
                    name=chunk.get("name"),  # optional
                    type=chunk.get("type"),  # optional
                    source_id=chunk.get("source_id"),  # optional
                    created_at=chunk.get("created_at"),  # optional
                    uid=chunk.get("uid"),  # optional
                    session_id=chunk.get("session_id"),  # optional
                    hostname=chunk.get("hostname"),  # optional
                    channel_count=int(chunk["channel_count"]),
                    channel_format=chunk["channel_format"],
                    nominal_srate=float(chunk["nominal_srate"]),
                )
            )
    return streams


def _read_chunks(f):
    """Read and yield XDF chunks.

    Parameters
    ----------
    f : file handle
        File handle of XDF file.


    Yields
    ------
    chunk : dict
        XDF chunk.
    """
    while True:
        chunk = dict()
        try:
            chunk["nbytes"] = _read_varlen_int(f)
        except EOFError:
            return
        chunk["tag"] = struct.unpack("<H", f.read(2))[0]
        if chunk["tag"] in [2, 3, 4, 6]:
            chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
            if chunk["tag"] == 2:  # parse StreamHeader chunk
                # if you edit, check for consistency with parsing in load_xdf
                msg = f.read(chunk["nbytes"] - 6).decode("utf-8", "replace")
                xml = fromstring(msg)
                chunk = {**chunk, **_parse_streamheader(xml)}
            else:  # skip remaining chunk contents
                f.seek(chunk["nbytes"] - 6, 1)
        else:
            f.seek(chunk["nbytes"] - 2, 1)  # skip remaining chunk contents
        yield chunk


def _parse_streamheader(xml):
    """Parse stream header XML."""
    return {el.tag: el.text for el in xml if el.tag != "desc"}


def _interpolate(
    x: np.ndarray, y: np.ndarray, new_x: np.ndarray, kind="linear"
) -> np.ndarray:
    """Perform interpolation for _sync_timestamps

    If scipy is not installed, the method falls back to numpy, and then only
    supports linear interpolation.
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


def _sync_timestamps(streams, kind="linear", use_samplingrate="highest"):
    """Sync all streams to the fastest sampling rate by shifting or upsampling.

    Depending on a streams channel-format, extrapolation is performed using
    with NaNs (numerical formats) or with [''] (string format).

    Interpolation is only performed for numeric values, and depending on the
    kind argument which is inherited from scipy.interpolate.interp1d. Consider
    that If the channel format is an integer type (i.e. 'int8', 'int16',
    'int32', or 'int64'), integer output is enforced by rounding the values.
    Additionally, consider that not all interpolation methods are convex, i.e.
    for some kinds, you might receive values above or below the desired
    integer type. There is no correction implemented for this, as it is assumed
    this is a desired behavior if you give the appropriate argument.

    For string formats, events are shifted towards the nearest feasible
    timepoint. Any time-stamps without a marker get then assigned an empty
    marker, i.e. [''].
    """
    # selecting the stream with the highest effective sampling rate
    srates = [stream.effective_srate for stream in streams.values()]
    max_fs = max(srates, default=0)
    if max_fs == 0:  # either no valid stream or all streams are async
        return streams
    if srates.count(max_fs) > 1:
        # highly unlikely, with floating point precision and sampling noise
        # but anyways: better safe than sorry
        logger.warning(
            "I found at least two streams with identical effective "
            "srate. I select one at random for syncing timestamps."
        )

    # selecting maximal time range of the whole recording
    # streams with fs=0 might are not dejittered be default, and therefore
    # indexing first and last might miss the earliest/latest
    # we therefore take the min and max timestamp
    stamps = [stream.time_stamps for stream in streams.values()]
    ts_first = min((min(s) for s in stamps))  # earliest sample of all streams
    ts_last = max((max(s) for s in stamps))  # latest sample of all streams

    # generate new timestamps
    # based on extrapolation of the fastest timestamps towards the maximal
    # time range of the whole recording
    if use_samplingrate == "highest":
        new_srate = max_fs
    elif type(use_samplingrate) == int:
        new_srate = use_samplingrate
    else:
        raise NotImplementedError(
            f"Unknown parameter for use_samplingrate: {use_samplingrate}"
        )
    fs_step = 1.0 / new_srate
    if new_srate == max_fs:
        new_timestamps = stamps[
            srates.index(max_fs)
        ]  # pick fastest stream regardless of finally used srate
        num_steps = int((new_timestamps[0] - ts_first) / fs_step) + 1
        front_stamps = np.linspace(ts_first, new_timestamps[0], num_steps)
        num_steps = int((ts_last - new_timestamps[-1]) / fs_step) + 1
        end_stamps = np.linspace(new_timestamps[-1], ts_last, num_steps)
        new_timestamps = np.concatenate(
            (front_stamps, new_timestamps[1:-1], end_stamps), axis=0
        )
    else:
        new_timestamps = np.arange(ts_first, ts_last, fs_step)

    # interpolate or shift all streams to the new timestamps
    for stream in streams.values():
        if (stream.channel_format == "string") and (
            stream.effective_srate == 0
        ):
            # you can't really interpolate strings; and streams with srate=0
            # don't have a real sampling rate. One approach to sync them is to
            # shift their events to the nearest timestamp of the new
            # timestamps
            shifted_x = []
            for x in stream.time_stamps:
                argmin = (abs(new_timestamps - x)).argmin()
                shifted_x.append(new_timestamps[argmin])

            shifted_y = []
            for x in new_timestamps:
                try:
                    idx = shifted_x.index(x)
                    y = stream.time_series[idx]
                    shifted_y.append([y])
                except ValueError:
                    shifted_y.append([""])

            stream.time_series = np.asanyarray((shifted_y))
            stream.time_stamps = new_timestamps

        elif stream.channel_format in [
            "float32",
            "double64",
            "int8",
            "int16",
            "int32",
            "int64",
        ]:
            # continuous interpolation is possible using interp1d
            # discrete interpolation requires some finetuning
            # bounds_error=False replaces everything outside of interpolation
            # zone with NaNs
            y = stream.time_series
            x = stream.time_stamps

            stream.time_series = _interpolate(x, y, new_timestamps, kind)
            stream.time_stamps = new_timestamps

            if stream.channel_format in ["int8", "int16", "int32", "int64"]:
                # i am stuck with float64s, as integers have no nans
                # therefore i round to the nearest integer instead
                stream.time_series = np.around(stream.time_series, 0)
        elif (stream.channel_format == "string") and (
            stream.effective_srate != 0
        ):
            warnings.warn(
                "Can't interpolate a channel with channel format string and an effective sampling rate != 0"
            )
        else:
            raise NotImplementedError(
                "Don't know how to sync sampling for "
                "channel_format="
                "{}".format(stream.channel_format)
            )
        stream.effective_srate = new_srate

    return streams


def _limit_streams_to_overlap(streams):
    """takes streams, returns streams limited to time periods overlapping
    between all streams

    The overlapping periods start and end for each streams with the first and
    last sample completely within the overlapping period.
    If time_stamps have been synced, these are the same time-points for all
    streams. Consider that in the case of unsynced time-stamps, the time-stamps
    can not be exactly equal!
    """
    ts_first, ts_last = [], []
    for stream in streams.values():
        # skip streams with fs=0 or if they send strings, because they might
        # just not yet have send anything on purpose (i.e. markers)
        # while other data was already  being recorded.
        if (
            stream.effective_srate != 0
            and stream.channel_format != "string"
        ):
            # extrapolation in _sync_timestamps is done with NaNs
            not_extrapolated = np.where(~np.isnan(stream.time_series))[0]
            ts_first.append(min(stream.time_stamps[not_extrapolated]))
            ts_last.append(max(stream.time_stamps[not_extrapolated]))

    ts_first = max(ts_first)
    ts_last = min(ts_last)
    for stream in streams.values():
        # use np.around to prevent floating point hickups
        around = np.around(stream.time_stamps, 15)
        a = np.where(around >= ts_first)[0]
        b = np.where(around <= ts_last)[0]
        select = np.intersect1d(a, b)
        if type(stream.time_stamps) is list:
            stream.time_stamps = [stream.time_stamps[s] for s in select]
        else:
            stream.time_stamps = stream.time_stamps[select]

        if type(stream.time_series) is list:
            stream.time_series = [stream.time_series[s] for s in select]
        else:
            stream.time_series = stream.time_series[select]

    return streams
