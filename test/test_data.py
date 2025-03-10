from pathlib import Path

import numpy as np
import pytest

from pyxdf import load_xdf, match_streaminfos, resolve_streams

# requires git clone https://github.com/xdf-modules/example-files.git into the root
# pyxdf folder
path = Path("example-files")
files = {
    key: path / value
    for key, value in {
        "minimal": "minimal.xdf",
        "clock_resets": "clock_resets.xdf",
        "empty_streams": "empty_streams.xdf",
    }.items()
    if (path / value).exists()
}


@pytest.mark.skipif("minimal" not in files, reason="File not found.")
def test_match_streaminfos():
    """Test matching stream infos."""
    path = files["minimal"]
    stream_infos = resolve_streams(path)

    parameters = [{"name": "SendDataString"}]
    matches = match_streaminfos(stream_infos, parameters)
    assert matches == [46202862]

    parameters = [{"name": "senddatastring"}]
    matches = match_streaminfos(stream_infos, parameters)
    assert matches == []

    parameters = [{"name": "senddatastring"}]
    matches = match_streaminfos(stream_infos, parameters, case_sensitive=False)
    assert matches == [46202862]


@pytest.mark.parametrize("synchronize_clocks", [False, True])
@pytest.mark.skipif("minimal" not in files, reason="File not found.")
def test_minimal_file(synchronize_clocks):
    path = files["minimal"]
    streams, header = load_xdf(
        path,
        synchronize_clocks=synchronize_clocks,
    )

    assert header["info"]["version"][0] == "1.0"

    # Stream ID: 0
    i = 0
    assert len(streams) == 2
    assert streams[i]["info"]["name"][0] == "SendDataC"
    assert streams[i]["info"]["type"][0] == "EEG"
    assert streams[i]["info"]["channel_count"][0] == "3"
    assert streams[i]["info"]["nominal_srate"][0] == "10"
    assert streams[i]["info"]["channel_format"][0] == "int16"
    assert streams[i]["info"]["created_at"][0] == "50942.723319709003"
    assert streams[i]["info"]["desc"][0] is None
    assert streams[i]["info"]["uid"][0] == "xdfwriter_11_int"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 0
    assert streams[i]["info"]["effective_srate"] == pytest.approx(10)
    assert streams[i]["info"]["segments"] == [(0, 8)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 8)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "5.1"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "5.9"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "9"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "50979.76"
    assert first_clock_offset["value"][0] == "-.01"

    # Time-series data
    s = np.array(
        [
            [192, 255, 238],
            [12, 22, 32],
            [13, 23, 33],
            [14, 24, 34],
            [15, 25, 35],
            [12, 22, 32],
            [13, 23, 33],
            [14, 24, 34],
            [15, 25, 35],
        ],
        dtype=np.int16,
    )
    t = np.array([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
    if synchronize_clocks:
        # Shift time according to clock offsets: time-stamps earlier
        # than the first clock-offset measurement are considered part of
        # the first clock segment
        t = t - 0.1
    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets
    clock_times = np.asarray([6.1, 7.1])
    clock_values = np.asarray([-0.1, -0.1])

    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_allclose(streams[i]["clock_values"], clock_values)

    # Stream ID: 0x02C0FFEE
    i = 1
    assert streams[i]["info"]["name"][0] == "SendDataString"
    assert streams[i]["info"]["type"][0] == "StringMarker"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["nominal_srate"][0] == "10"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["created_at"][0] == "50942.723319709003"
    assert streams[i]["info"]["desc"][0] is None
    assert streams[i]["info"]["uid"][0] == "xdfwriter_11_str"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 0x02C0FFEE
    assert streams[i]["info"]["effective_srate"] == pytest.approx(10)
    assert streams[i]["info"]["segments"] == [(0, 8)]

    # Footer should be identical to Stream 0
    assert streams[i]["footer"] == streams[0]["footer"]

    # Time-series data
    s = [
        [
            '<?xml version="1.0"?><info><writer>LabRecorder xdfwriter'
            "</writer><first_timestamp>5.1</first_timestamp><last_timestamp>"
            "5.9</last_timestamp><sample_count>9</sample_count>"
            "<clock_offsets><offset><time>50979.76</time><value>-.01</value>"
            "</offset><offset><time>50979.86</time><value>-.02</value>"
            "</offset></clock_offsets></info>"
        ],
        ["Hello"],
        ["World"],
        ["from"],
        ["LSL"],
        ["Hello"],
        ["World"],
        ["from"],
        ["LSL"],
    ]
    t = np.array([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
    # Synchronization should have no effect on this stream due to no
    # clock_offsets
    assert streams[i]["time_series"] == s
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets: does not match the footer but there are no
    # clock_offsets for this stream in the file
    clock_times = np.asarray([])
    clock_values = np.asarray([])

    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_equal(streams[i]["clock_values"], clock_values)


@pytest.mark.parametrize("jitter_break_threshold_seconds", [0.11, 0.09])
@pytest.mark.skipif("minimal" not in files, reason="File not found.")
def test_minimal_file_segments(jitter_break_threshold_seconds):
    path = files["minimal"]
    streams, header = load_xdf(
        path,
        dejitter_timestamps=True,
        jitter_break_threshold_seconds=jitter_break_threshold_seconds,
        jitter_break_threshold_samples=0,
    )
    for stream in streams:
        tdiff = 1 / float(stream["info"]["nominal_srate"][0])
        if jitter_break_threshold_seconds > tdiff:
            assert stream["info"]["segments"] == [(0, 8)]
            assert stream["info"]["effective_srate"] == pytest.approx(10)
        else:
            # Pathological case where every sample is a segment
            assert stream["info"]["segments"] == [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
            ]
            assert stream["info"]["effective_srate"] == pytest.approx(0)


@pytest.mark.parametrize("dejitter_timestamps", [False, True])
@pytest.mark.parametrize("synchronize_clocks", [False, True])
@pytest.mark.skipif("clock_resets" not in files, reason="File not found.")
def test_clock_resets_file(synchronize_clocks, dejitter_timestamps):
    path = files["clock_resets"]
    streams, header = load_xdf(
        path,
        synchronize_clocks=synchronize_clocks,
        dejitter_timestamps=dejitter_timestamps,
    )

    assert header["info"]["version"][0] == "1.0"

    assert len(streams) == 2

    # Stream ID: 1
    i = 0
    assert streams[i]["info"]["name"][0] == "MyMarkerStream"
    assert streams[i]["info"]["type"][0] == "Markers"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["nominal_srate"][0] == "0"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["source_id"][0] == "myuidw43536"
    assert streams[i]["info"]["version"][0] == "1.1000000000000001"
    assert streams[i]["info"]["created_at"][0] == "564076.02850699995"
    assert streams[i]["info"]["uid"][0] == "1efcb4a6-8894-4014-b404-4b6f6b2205f2"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "BP-LP-022"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16572"
    assert streams[i]["info"]["v4service_port"][0] == "16572"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "16572"
    assert streams[i]["info"]["v6service_port"][0] == "16572"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 1
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 174)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 90), (91, 174)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "653153.2121885"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "259.6538279"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "175"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "653156.02616855"
    assert first_clock_offset["value"][0] == "-652340.2838639501"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "264.6430016000002"
    assert last_clock_offset["value"][0] == "1121.165595"

    # Clock offsets: Test against footer
    assert streams[i]["clock_times"][0] == pytest.approx(
        float(first_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][0] == pytest.approx(
        float(first_clock_offset["value"][0]), abs=1e-4
    )
    assert streams[i]["clock_times"][-1] == pytest.approx(
        float(last_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][-1] == pytest.approx(
        float(last_clock_offset["value"][0]), abs=1e-4
    )

    # Stream ID: 2
    i = 1
    assert streams[i]["info"]["name"][0] == "BioSemi"
    assert streams[i]["info"]["type"][0] == "EEG"
    assert streams[i]["info"]["channel_count"][0] == "8"
    assert streams[i]["info"]["nominal_srate"][0] == "100"
    assert streams[i]["info"]["channel_format"][0] == "float32"
    assert streams[i]["info"]["source_id"][0] == "myuid34234"
    assert streams[i]["info"]["version"][0] == "1.1000000000000001"
    assert streams[i]["info"]["created_at"][0] == "653103.26692229998"
    assert streams[i]["info"]["uid"][0] == "fa3e14ab-b621-480e-a9d5-c740f0e47140"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "BP-LP-022"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16573"
    assert streams[i]["info"]["v4service_port"][0] == "16573"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "16573"
    assert streams[i]["info"]["v6service_port"][0] == "16573"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 2
    if dejitter_timestamps:
        assert streams[i]["info"]["effective_srate"] == pytest.approx(92.934, abs=1e-3)
        assert streams[i]["info"]["segments"] == [(0, 12875), (12876, 27814)]
    else:
        # Effective srate will be incorrect.
        assert streams[i]["info"]["effective_srate"] != pytest.approx(92.934, abs=1e-3)
        assert streams[i]["info"]["segments"] == [(0, 27814)]

    assert streams[i]["info"]["clock_segments"] == (
        [(0, 12875), (12876, 27814)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "653150.379117"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "261.9267033"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "27815"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "653156.0261441499"
    assert first_clock_offset["value"][0] == "-652340.28383985"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "264.6385764000001"
    assert last_clock_offset["value"][0] == "1121.1656319"

    # Clock offsets: Test against footer
    assert streams[i]["clock_times"][0] == pytest.approx(
        float(first_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][0] == pytest.approx(
        float(first_clock_offset["value"][0]), abs=1e-4
    )
    assert streams[i]["clock_times"][-1] == pytest.approx(
        float(last_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][-1] == pytest.approx(
        float(last_clock_offset["value"][0]), abs=1e-4
    )


@pytest.mark.parametrize("dejitter_timestamps", [False, True])
@pytest.mark.parametrize("synchronize_clocks", [False, True])
@pytest.mark.skipif("empty_streams" not in files, reason="File not found.")
def test_empty_streams_file(synchronize_clocks, dejitter_timestamps):
    path = files["empty_streams"]
    streams, header = load_xdf(
        path,
        synchronize_clocks=synchronize_clocks,
        dejitter_timestamps=dejitter_timestamps,
    )

    assert header["info"]["version"][0] == "1.0"

    assert len(streams) == 4

    # Stream ID: 1
    i = 2
    assert streams[i]["info"]["name"][0] == "ctrl"
    assert streams[i]["info"]["type"][0] == "control"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["source_id"][0] == "kassia"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "91684.87631725401"
    assert streams[i]["info"]["uid"][0] == "4740b9ba-d45d-4e2b-9a4b-ee966d9b56df"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16572"
    assert streams[i]["info"]["v4service_port"][0] == "16572"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 1
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 0)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 0)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "91725.014004246"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "91725.014004246"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "1"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "91716.691545932"
    assert first_clock_offset["value"][0] == "-1.889200211735442e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "91746.69261952251"
    assert last_clock_offset["value"][0] == "-3.837950498564169e-05"

    # Time-series data
    s = [['{"state": 2}']]
    t = np.array([91725.01400425])

    assert streams[i]["time_series"] == s
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets: Test against footer
    assert streams[i]["clock_times"][0] == pytest.approx(
        float(first_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][0] == pytest.approx(
        float(first_clock_offset["value"][0]), abs=1e-4
    )
    assert streams[i]["clock_times"][-1] == pytest.approx(
        float(last_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][-1] == pytest.approx(
        float(last_clock_offset["value"][0]), abs=1e-4
    )

    # Stream ID: 2
    i = 3
    assert streams[i]["info"]["name"][0] == "Empty marker stream: test stream 0 counter"
    assert streams[i]["info"]["type"][0] == "data"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["source_id"][0] == "test_stream.py:191748:0"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "91696.18816467900"
    assert streams[i]["info"]["uid"][0] == "6f7e0288-10b8-4f48-89f8-0381ce20922f"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16574"
    assert streams[i]["info"]["v4service_port"][0] == "16574"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 2
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == []
    assert streams[i]["info"]["clock_segments"] == []

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "0"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "91716.691513728"
    assert first_clock_offset["value"][0] == "-2.540599962230772e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "91746.69276649199"
    assert last_clock_offset["value"][0] == "-2.026499714702368e-05"

    # Time-series data
    s = []
    t = np.array([], dtype=np.float64)

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_equal(streams[i]["time_stamps"], t)

    # Clock offsets: Test against footer
    assert streams[i]["clock_times"][0] == pytest.approx(
        float(first_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][0] == pytest.approx(
        float(first_clock_offset["value"][0]), abs=1e-4
    )
    assert streams[i]["clock_times"][-1] == pytest.approx(
        float(last_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][-1] == pytest.approx(
        float(last_clock_offset["value"][0]), abs=1e-4
    )

    # Stream ID: 3
    i = 0
    assert streams[i]["info"]["name"][0] == "Empty data stream: test stream 0 counter"
    assert streams[i]["info"]["type"][0] == "data"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "float32"
    assert streams[i]["info"]["source_id"][0] == "test_stream.py:191790:0"
    assert streams[i]["info"]["nominal_srate"][0] == "1.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "91699.83395166400"
    assert streams[i]["info"]["uid"][0] == "ff430a18-9954-43f5-bb5f-f5589e3aa6a2"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16575"
    assert streams[i]["info"]["v4service_port"][0] == "16575"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"
    channels = streams[i]["info"]["desc"][0]["channels"][0]
    assert channels["channel"][0]["label"][0] == "ch:00"
    assert channels["channel"][0]["type"][0] == "misc"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 3
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == []
    assert streams[i]["info"]["clock_segments"] == []

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "0"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "91716.6915301265"
    assert first_clock_offset["value"][0] == "-2.211050014011562e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "91746.69269425601"
    assert last_clock_offset["value"][0] == "-1.128000440075994e-05"

    # Time-series data
    s = np.zeros((0, 1), dtype=np.int32)
    t = np.array([], dtype=np.float64)

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_equal(streams[i]["time_stamps"], t)

    # Clock offsets: Test against footer
    assert streams[i]["clock_times"][0] == pytest.approx(
        float(first_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][0] == pytest.approx(
        float(first_clock_offset["value"][0]), abs=1e-4
    )
    assert streams[i]["clock_times"][-1] == pytest.approx(
        float(last_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][-1] == pytest.approx(
        float(last_clock_offset["value"][0]), abs=1e-4
    )

    # Stream ID: 4
    i = 1
    assert streams[i]["info"]["name"][0] == "Data stream: test stream 0 counter"
    assert streams[i]["info"]["type"][0] == "data"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "int32"
    assert streams[i]["info"]["source_id"][0] == "test_stream.py:191695:0"
    assert streams[i]["info"]["nominal_srate"][0] == "1.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "91690.55328314200"
    assert streams[i]["info"]["uid"][0] == "bc60b7bb-e632-407c-b3db-e42f9cad4179"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16573"
    assert streams[i]["info"]["v4service_port"][0] == "16573"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"
    channels = streams[i]["info"]["desc"][0]["channels"][0]
    assert channels["channel"][0]["label"][0] == "ch:00"
    assert channels["channel"][0]["type"][0] == "misc"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 4
    assert streams[i]["info"]["effective_srate"] == pytest.approx(1)
    assert streams[i]["info"]["segments"] == [(0, 9)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 9)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "91725.21394789348"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "91735.21394789348"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "10"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "91716.6915717245"
    assert first_clock_offset["value"][0] == "-1.94335007108748e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "91746.69274602149"
    assert last_clock_offset["value"][0] == "-3.694550105137751e-05"

    # Time-series data
    s = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=np.int32)
    t = np.array(
        [
            91725.21394789,
            91726.21394789,
            91727.21394789,
            91728.21394789,
            91729.21394789,
            91730.21394789,
            91731.21394789,
            91732.21394789,
            91733.21394789,
            91734.21394789,
        ]
    )

    np.testing.assert_equal(streams[i]["time_series"], s)
    if synchronize_clocks:
        np.testing.assert_almost_equal(streams[i]["time_stamps"], t, decimal=4)
    else:
        np.testing.assert_almost_equal(streams[i]["time_stamps"], t, decimal=8)

    # Dejittering should have negligible effect because ground-truth timestamps have
    # zero jitter
    assert np.std(np.diff(streams[i]["time_stamps"])) < 6e-11

    # Clock offsets: Test against footer
    assert streams[i]["clock_times"][0] == pytest.approx(
        float(first_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][0] == pytest.approx(
        float(first_clock_offset["value"][0]), abs=1e-4
    )
    assert streams[i]["clock_times"][-1] == pytest.approx(
        float(last_clock_offset["time"][0]), abs=1e-6
    )
    assert streams[i]["clock_values"][-1] == pytest.approx(
        float(last_clock_offset["value"][0]), abs=1e-4
    )
