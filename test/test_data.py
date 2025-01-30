from pathlib import Path

import numpy as np
import pytest

from pyxdf import load_xdf

# requires git clone https://github.com/xdf-modules/example-files.git
# into the root pyxdf folder
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


@pytest.mark.parametrize("dejitter_timestamps", [False, True])
@pytest.mark.parametrize("synchronize_clocks", [False, True])
@pytest.mark.skipif("minimal" not in files, reason="File not found.")
def test_minimal_file(synchronize_clocks, dejitter_timestamps):
    path = files["minimal"]
    streams, header = load_xdf(
        path,
        synchronize_clocks=synchronize_clocks,
        dejitter_timestamps=dejitter_timestamps,
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

    # Info added by pyxdf.
    assert streams[i]["info"]["stream_id"] == 0
    assert streams[i]["info"]["effective_srate"] == pytest.approx(10)
    assert streams[i]["info"]["segments"] == [(0, 8)]

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
        # Shift time according to clock offsets.
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

    # Info added by pyxdf.
    assert streams[i]["info"]["stream_id"] == 0x02C0FFEE
    assert streams[i]["info"]["effective_srate"] == pytest.approx(10)
    assert streams[i]["info"]["segments"] == [(0, 8)]

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "5.1"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "5.9"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "9"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "50979.76"
    assert first_clock_offset["value"][0] == "-.01"

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
    assert streams[i]["time_series"] == s
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets - does not match the footer but it is in the file.
    clock_times = np.asarray([])
    clock_values = np.asarray([])

    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_equal(streams[i]["clock_values"], clock_values)

    streams, header = load_xdf(
        path,
        synchronize_clocks=synchronize_clocks,
        dejitter_timestamps=dejitter_timestamps,
        jitter_break_threshold_seconds=0.001,
        jitter_break_threshold_samples=1,
    )
    if dejitter_timestamps:
        assert streams[0]["info"]["segments"] == [(0, 0), (1, 3), (4, 8)]
    else:
        assert streams[0]["info"]["segments"] == [(0, 8)]


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
    i = 1
    assert streams[i]["info"]["name"][0] == "Data stream: test stream 0 counter"
    assert streams[i]["info"]["type"][0] == "data"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "int32"
    assert streams[i]["info"]["source_id"][0] == "test_stream.py:525352:0"
    assert streams[i]["info"]["nominal_srate"][0] == "1.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "401309.0364671120"
    assert streams[i]["info"]["uid"][0] == "25e1bd13-340f-499c-bb91-5d1e75cec535"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16576"
    assert streams[i]["info"]["v4service_port"][0] == "16575"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"
    channels = streams[i]["info"]["desc"][0]["channels"][0]
    assert channels["channel"][0]["label"][0] == "ch:00"
    assert channels["channel"][0]["type"][0] == "misc"

    # Info added by pyxdf.
    assert streams[i]["info"]["stream_id"] == 1
    assert streams[i]["info"]["effective_srate"] == pytest.approx(1)
    assert streams[i]["info"]["segments"] == [(0, 9)]

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "401340.7970979316"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "401350.7970979316"
    # Sample count is off by one.
    assert streams[i]["footer"]["info"]["sample_count"][0] == "9"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "401322.6950535755"
    assert first_clock_offset["value"][0] == "-3.67984757758677e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "401372.696774303"
    assert last_clock_offset["value"][0] == "-3.553100395947695e-05"

    # Time-series data
    s = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=np.int32)
    t = np.array(
        [
            401340.79706958,
            401341.79706948,
            401342.79706938,
            401343.79706927,
            401344.79706917,
            401345.79706907,
            401346.79706896,
            401347.79706886,
            401348.79706875,
            401349.79706865,
        ]
    )

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets
    np.testing.assert_equal(streams[i]["clock_times"][0], 401322.6950535755)
    np.testing.assert_equal(streams[i]["clock_values"][0], -3.67984757758677e-05)
    np.testing.assert_equal(streams[i]["clock_times"][-1], 401372.696774303)
    np.testing.assert_equal(streams[i]["clock_values"][-1], -3.553100395947695e-05)

    # Stream ID: 2
    i = 0
    assert streams[i]["info"]["name"][0] == "Empty data stream: test stream 0 counter"
    assert streams[i]["info"]["type"][0] == "data"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "float32"
    assert streams[i]["info"]["source_id"][0] == "test_stream.py:525257:0"
    assert streams[i]["info"]["nominal_srate"][0] == "1.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "401285.3015719900"
    assert streams[i]["info"]["uid"][0] == "30608cb9-b177-420d-9d60-3ce0f07949af"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16574"
    assert streams[i]["info"]["v4service_port"][0] == "16573"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"
    channels = streams[i]["info"]["desc"][0]["channels"][0]
    assert channels["channel"][0]["label"][0] == "ch:00"
    assert channels["channel"][0]["type"][0] == "misc"

    # Info added by pyxdf.
    assert streams[i]["info"]["stream_id"] == 2
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == []

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "0"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "401322.695044571"
    assert first_clock_offset["value"][0] == "-3.130998811684549e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "401372.6966923515"
    assert last_clock_offset["value"][0] == "-1.937249908223748e-05"

    # Time-series data
    s = np.zeros((0, 1), dtype=np.int32)
    t = np.array([], dtype=np.float64)

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets
    np.testing.assert_equal(streams[i]["clock_times"][0], 401322.69504457095)
    np.testing.assert_equal(streams[i]["clock_values"][0], -3.130998811684549e-05)
    np.testing.assert_equal(streams[i]["clock_times"][-1], 401372.6966923515)
    np.testing.assert_equal(streams[i]["clock_values"][-1], -1.9372499082237482e-05)

    # Stream ID: 3
    i = 2
    assert streams[i]["info"]["name"][0] == "Empty marker stream: test stream 0 counter"
    assert streams[i]["info"]["type"][0] == "data"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["source_id"][0] == "test_stream.py:525304:0"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["stream_id"] == 3
    assert streams[i]["info"]["segments"] == []
    assert streams[i]["info"]["clock_segments"] == []
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "401297.3977076210"
    assert streams[i]["info"]["uid"][0] == "3ece2528-0c45-4e6f-9a00-7eb1a7f7bd84"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16575"
    assert streams[i]["info"]["v4service_port"][0] == "16574"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"

    # Info added by pyxdf.
    assert streams[i]["info"]["stream_id"] == 3
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == []

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "0"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "0"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "401322.6951932265"
    assert first_clock_offset["value"][0] == "-2.594449324533343e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "401372.6966891775"
    assert last_clock_offset["value"][0] == "-1.620649709366262e-05"

    # Time-series data
    s = []
    t = np.array([], dtype=np.float64)

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets
    np.testing.assert_equal(streams[i]["clock_times"][0], 401322.6951932265)
    np.testing.assert_equal(streams[i]["clock_values"][0], -2.5944493245333433e-05)
    np.testing.assert_equal(streams[i]["clock_times"][-1], 401372.69668917754)
    np.testing.assert_equal(streams[i]["clock_values"][-1], -1.620649709366262e-05)

    # Stream ID: 4
    i = 3
    assert streams[i]["info"]["name"][0] == "ctrl"
    assert streams[i]["info"]["type"][0] == "control"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["source_id"][0] == "kassia"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["version"][0] == "1.100000000000000"
    assert streams[i]["info"]["created_at"][0] == "401261.9233872890"
    assert streams[i]["info"]["uid"][0] == "eb31d8f6-b57a-4e45-bc5a-fa98573d6503"
    assert streams[i]["info"]["session_id"][0] == "default"
    assert streams[i]["info"]["hostname"][0] == "kassia"
    assert streams[i]["info"]["v4address"][0] is None
    assert streams[i]["info"]["v4data_port"][0] == "16573"
    assert streams[i]["info"]["v4service_port"][0] == "16572"
    assert streams[i]["info"]["v6address"][0] is None
    assert streams[i]["info"]["v6data_port"][0] == "0"
    assert streams[i]["info"]["v6service_port"][0] == "0"
    assert streams[i]["info"]["desc"][0]["manufacturer"][0] == "pylsltools"

    # Info added by pyxdf.
    assert streams[i]["info"]["stream_id"] == 4
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 0)]

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "401340.597121355"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "401340.597121355"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "0"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "401322.69519626"
    assert first_clock_offset["value"][0] == "-2.898101229220629e-05"
    last_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][-1]
    assert last_clock_offset["time"][0] == "401372.6968414925"
    assert last_clock_offset["value"][0] == "-2.722250064834952e-05"

    # Time-series data
    s = [['{"state": 2}']]
    t = np.array([401340.59709634])

    assert streams[i]["time_series"] == s
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock offsets
    np.testing.assert_equal(streams[i]["clock_times"][0], 401322.69519626)
    np.testing.assert_equal(streams[i]["clock_values"][0], -2.8981012292206287e-05)
    np.testing.assert_equal(streams[i]["clock_times"][-1], 401372.6968414925)
    np.testing.assert_equal(streams[i]["clock_values"][-1], -2.7222500648349524e-05)
