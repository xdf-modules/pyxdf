import io
import logging
import struct

import numpy as np
import pytest

from pyxdf import load_xdf
from pyxdf.pyxdf import StreamData, _parse_numeric_samples, _robust_fit

SRATE = 100
T0 = 1000.123


def _varlen(n):
    return b"\x01" + struct.pack("<B", n) if n < 256 else b"\x04" + struct.pack("<I", n)


def _chunk(tag, content, stream_id=None, len_delta=0):
    body = struct.pack("<H", tag)
    if stream_id is not None:
        body += struct.pack("<I", stream_id)
    body += content
    return _varlen(len(body) + len_delta) + body


def _header(fmt, nchns):
    return {
        "info": {
            "channel_count": [str(nchns)],
            "nominal_srate": [str(SRATE)],
            "channel_format": [fmt],
        }
    }


def _samples(data, stamps, has_ts):
    """Encode the content of a [Samples] chunk."""
    parts = [_varlen(len(data))]
    for values, stamp, flag in zip(data, stamps, has_ts):
        parts.append(b"\x08" + struct.pack("<d", stamp) if flag else b"\x00")
        parts.append(values.astype(values.dtype.newbyteorder("<")).tobytes())
    return b"".join(parts)


def _make_stream(fmt="float32", nchns=3, nsamples=200, chunk_size=10, mode="all"):
    """Create samples, the time stamps the reader should return, and the chunks."""
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((nsamples, nchns)) * 100).astype(
        StreamData(_header(fmt, nchns)).dtype
    )
    true_stamps = T0 + np.arange(nsamples) / SRATE + rng.normal(0, 1e-4, nsamples)
    has_ts = {
        "all": np.ones(nsamples, dtype=bool),
        "none": np.zeros(nsamples, dtype=bool),
        "first": np.arange(nsamples) % chunk_size == 0,
        "mixed": rng.random(nsamples) < 0.5,
    }[mode]
    # time stamps without a value are deduced from the previous one
    stamps = np.zeros(nsamples)
    last = 0.0
    for k in range(nsamples):
        last = true_stamps[k] if has_ts[k] else last + 1 / SRATE
        stamps[k] = last
    chunks = [
        _samples(
            data[i : i + chunk_size],
            true_stamps[i : i + chunk_size],
            has_ts[i : i + chunk_size],
        )
        for i in range(0, nsamples, chunk_size)
    ]
    return data, stamps, chunks


def _make_file(fmt, nchns, chunks, len_deltas=None):
    hdr = (
        "<?xml version='1.0'?><info><name>Test</name><type>EEG</type>"
        f"<channel_count>{nchns}</channel_count><nominal_srate>{SRATE}</nominal_srate>"
        f"<channel_format>{fmt}</channel_format></info>"
    ).encode()
    parts = [
        b"XDF:",
        _chunk(1, b"<?xml version='1.0'?><info><version>1.0</version></info>"),
        _chunk(2, hdr, 1),
    ]
    len_deltas = len_deltas or [0] * len(chunks)
    parts += [_chunk(3, c, 1, d) for c, d in zip(chunks, len_deltas)]
    return b"".join(parts)


def _load(raw):
    streams, _ = load_xdf(
        io.BytesIO(raw), synchronize_clocks=False, dejitter_timestamps=False
    )
    return streams[0]


@pytest.mark.parametrize(
    "fmt", ["int8", "int16", "int32", "int64", "float32", "double64"]
)
@pytest.mark.parametrize("mode", ["all", "none", "first", "mixed"])
def test_read_numeric_samples(fmt, mode):
    """Samples and (deduced) time stamps are read exactly."""
    data, stamps, chunks = _make_stream(fmt=fmt, mode=mode)
    stream = _load(_make_file(fmt, data.shape[1], chunks))
    assert stream["time_series"].dtype == data.dtype
    assert stream["time_series"].flags.writeable
    np.testing.assert_array_equal(stream["time_series"], data)
    np.testing.assert_array_equal(stream["time_stamps"], stamps)


@pytest.mark.parametrize("mode", ["all", "none", "first", "mixed"])
@pytest.mark.parametrize("len_delta", [-100, -9, -8, -1, 1, 8, 9, 1000])
@pytest.mark.parametrize("wrong_chunk", [0, -1])
def test_read_numeric_samples_wrong_chunk_length(mode, len_delta, wrong_chunk):
    """Samples are still read if the chunk length does not match the samples."""
    data, stamps, chunks = _make_stream(mode=mode)
    # a wrong length in the first chunk must not affect later chunks, and a length in
    # the last chunk that reaches past the end of the file must not drop it
    len_deltas = [0] * len(chunks)
    len_deltas[wrong_chunk] = len_delta
    stream = _load(_make_file("float32", data.shape[1], chunks, len_deltas))
    np.testing.assert_array_equal(stream["time_series"], data)
    np.testing.assert_array_equal(stream["time_stamps"], stamps)


@pytest.mark.parametrize("mode", ["all", "none", "first", "mixed"])
@pytest.mark.parametrize("cut", [1, 4, 9, 13])
def test_read_numeric_samples_truncated(mode, cut, caplog):
    """A chunk cut off at the end of the file is dropped instead of misread."""
    data, stamps, chunks = _make_stream(mode=mode)
    raw = _make_file("float32", data.shape[1], chunks)[:-cut]
    with caplog.at_level(logging.ERROR, logger="pyxdf.pyxdf"):
        stream = _load(raw)
    assert "corruption" in caplog.text
    n = len(data) - 10  # the last chunk is dropped
    np.testing.assert_array_equal(stream["time_series"], data[:n])
    np.testing.assert_array_equal(stream["time_stamps"], stamps[:n])


def test_read_numeric_samples_empty_chunk():
    """Chunks without samples are read without affecting the deduced time stamps."""
    data, stamps, chunks = _make_stream(mode="first")
    chunks.insert(1, _varlen(0))
    stream = _load(_make_file("float32", data.shape[1], chunks))
    np.testing.assert_array_equal(stream["time_series"], data)
    np.testing.assert_array_equal(stream["time_stamps"], stamps)


@pytest.mark.parametrize(
    "has_ts, extra_bytes",
    [
        ([True, True, False], 8),  # sized as if every sample had a time stamp
        ([True, False, False], -8),  # sized as if no sample had a time stamp
        ([True, False, False], 8),  # sized for more time stamps than flagged
        ([False, True, False], -1),  # sized for fewer time stamps than flagged
    ],
)
def test_parse_numeric_samples_flag_mismatch(has_ts, extra_bytes):
    """Time stamp flags that do not match the number of bytes are detected."""
    s = StreamData(_header("float32", 2))
    data = np.ones((len(has_ts), 2), dtype=np.float32)
    buf = _samples(data, np.ones(len(has_ts)), has_ts)[2:]  # without [NumSamples]
    buf = buf + b"\x00" * extra_bytes if extra_bytes > 0 else buf[:extra_bytes]
    assert _parse_numeric_samples(buf, len(has_ts), s) is None


def _robust_fit_reference(A, y, rho=1, iters=1000):
    """Original ADMM implementation of _robust_fit, before it was simplified."""
    A = np.copy(A)
    offset = np.min(A[:, 1])
    A[:, 1] -= offset
    Aty = np.dot(A.T, y)
    L = np.linalg.cholesky(np.dot(A.T, A))
    U = L.T
    z = np.zeros_like(y)
    u = z
    x = z
    for _ in range(iters):
        x = np.linalg.solve(U, (np.linalg.solve(L, Aty + np.dot(A.T, z - u))))
        d = np.dot(A, x) - y + u
        d_inv = np.zeros_like(d)
        np.divide(1, d, out=d_inv, where=d != 0)
        tmp = np.maximum(0, (1 - (1 + 1 / rho) * np.abs(d_inv)))
        z = rho / (1 + rho) * d + 1 / (1 + rho) * tmp * d
        u = d - z
    x[0] -= x[1] * offset
    return x


@pytest.mark.parametrize("n", [2, 20, 500])
@pytest.mark.parametrize("rho", [0.5, 1, 2])
def test_robust_fit_matches_reference(n, rho):
    """The simplified ADMM iteration gives the same result as the original one."""
    rng = np.random.default_rng(n)
    winsor_threshold = 1e-4
    t = np.sort(rng.uniform(1e4, 1e4 + 5 * n, n))
    v = 0.01 + 2e-6 * (t - 1e4) + rng.normal(0, 1e-5, n)
    v[rng.choice(n, max(1, n // 10), replace=False)] += 5e-3  # outliers
    A = np.column_stack([np.ones(n), t / winsor_threshold])
    y = v / winsor_threshold
    np.testing.assert_allclose(
        _robust_fit(A, y, rho=rho), _robust_fit_reference(A, y, rho=rho), rtol=1e-9
    )
