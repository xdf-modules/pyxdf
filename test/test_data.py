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


@pytest.mark.skipif("minimal" not in files, reason="File not found.")
def test_minimal_file():
    path = files["minimal"]
    streams, header = load_xdf(path)

    assert header["info"]["version"][0] == "1.0"

    assert len(streams) == 2
    assert streams[0]["info"]["name"][0] == "SendDataC"
    assert streams[0]["info"]["type"][0] == "EEG"
    assert streams[0]["info"]["channel_count"][0] == "3"
    assert streams[0]["info"]["nominal_srate"][0] == "10"
    assert streams[0]["info"]["channel_format"][0] == "int16"
    assert streams[0]["info"]["stream_id"] == 0
    assert streams[0]["info"]["segments"] == [(0, 8)]

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
    t = np.array([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
    np.testing.assert_array_equal(streams[0]["time_series"], s)
    np.testing.assert_array_almost_equal(streams[0]["time_stamps"], t)

    clock_times = np.asarray([6.1, 7.1])
    clock_values = np.asarray([-0.1, -0.1])

    np.testing.assert_array_equal(streams[0]["clock_times"], clock_times)
    np.testing.assert_array_almost_equal(streams[0]["clock_values"], clock_values)

    assert streams[1]["info"]["name"][0] == "SendDataString"
    assert streams[1]["info"]["type"][0] == "StringMarker"
    assert streams[1]["info"]["channel_count"][0] == "1"
    assert streams[1]["info"]["nominal_srate"][0] == "10"
    assert streams[1]["info"]["channel_format"][0] == "string"
    assert streams[1]["info"]["stream_id"] == 0x02C0FFEE

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
    assert streams[1]["time_series"] == s
    np.testing.assert_array_almost_equal(streams[1]["time_stamps"], t)

    clock_times = np.asarray([])
    clock_values = np.asarray([])

    np.testing.assert_array_equal(streams[1]["clock_times"], clock_times)
    np.testing.assert_array_almost_equal(streams[1]["clock_values"], clock_values)

    streams, header = load_xdf(
        path,
        jitter_break_threshold_seconds=0.001,
        jitter_break_threshold_samples=1,
    )
    assert streams[0]["info"]["segments"] == [(0, 0), (1, 3), (4, 8)]


@pytest.mark.parametrize("dejitter_timestamps", [False, True])
def test_empty_streams_file(dejitter_timestamps):
    path = files["empty_streams"]
    streams, header = load_xdf(
        path,
        synchronize_clocks=False,
        dejitter_timestamps=dejitter_timestamps,
    )

    assert header["info"]["version"][0] == "1.0"

    assert len(streams) == 4

    # Stream ID: 1
    assert streams[1]["info"]["name"][0] == "Data stream: test stream 0 counter"
    assert streams[1]["info"]["type"][0] == "data"
    assert streams[1]["info"]["channel_count"][0] == "1"
    assert streams[1]["info"]["channel_format"][0] == "int32"
    assert streams[1]["info"]["nominal_srate"][0] == "1.000000000000000"
    assert streams[1]["info"]["stream_id"] == 1
    assert streams[1]["info"]["segments"] == [(0, 9)]

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

    np.testing.assert_equal(streams[1]["time_series"], s)
    np.testing.assert_allclose(streams[1]["time_stamps"], t)

    # Stream ID: 2
    assert streams[0]["info"]["name"][0] == "Empty data stream: test stream 0 counter"
    assert streams[0]["info"]["type"][0] == "data"
    assert streams[0]["info"]["channel_count"][0] == "1"
    assert streams[0]["info"]["channel_format"][0] == "float32"
    assert streams[0]["info"]["nominal_srate"][0] == "1.000000000000000"
    assert streams[0]["info"]["stream_id"] == 2
    assert streams[0]["info"]["segments"] == []

    s = np.zeros((0, 1), dtype=np.int32)
    t = np.array([], dtype=np.float64)

    np.testing.assert_equal(streams[0]["time_series"], s)
    np.testing.assert_allclose(streams[0]["time_stamps"], t)

    # Stream ID: 3
    assert streams[2]["info"]["name"][0] == "Empty marker stream: test stream 0 counter"
    assert streams[2]["info"]["type"][0] == "data"
    assert streams[2]["info"]["channel_count"][0] == "1"
    assert streams[2]["info"]["channel_format"][0] == "string"
    assert streams[2]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[2]["info"]["stream_id"] == 3
    assert streams[2]["info"]["segments"] == []
    assert streams[2]["info"]["clock_segments"] == []

    s = []
    t = np.array([], dtype=np.float64)

    np.testing.assert_equal(streams[2]["time_series"], s)
    np.testing.assert_allclose(streams[2]["time_stamps"], t)

    # Stream ID: 4
    assert streams[3]["info"]["name"][0] == "ctrl"
    assert streams[3]["info"]["type"][0] == "control"
    assert streams[3]["info"]["channel_count"][0] == "1"
    assert streams[3]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[3]["info"]["channel_format"][0] == "string"
    assert streams[3]["info"]["stream_id"] == 4
    assert streams[3]["info"]["segments"] == [(0, 0)]

    s = [['{"state": 2}']]
    t = np.array([401340.59709634])

    assert streams[3]["time_series"] == s
    np.testing.assert_allclose(streams[3]["time_stamps"], t)
