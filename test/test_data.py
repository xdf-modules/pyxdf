from pathlib import Path

import numpy as np
import pytest

from pyxdf import load_xdf

# requires git clone https://github.com/xdf-modules/example-files.git
# into the root pyxdf folder
path = Path("example-files")
extensions = ["*.xdf", "*.xdfz", "*.xdf.gz"]
files = []
for ext in extensions:
    files.extend(path.glob(ext))
files = [str(file) for file in files]


@pytest.mark.parametrize("file", files)
def test_load_file(file):
    streams, header = load_xdf(file)

    if file.endswith("minimal.xdf"):
        assert header["info"]["version"][0] == "1.0"

        assert len(streams) == 2
        assert streams[0]["info"]["name"][0] == "SendDataC"
        assert streams[0]["info"]["type"][0] == "EEG"
        assert streams[0]["info"]["channel_count"][0] == "3"
        assert streams[0]["info"]["nominal_srate"][0] == "10"
        assert streams[0]["info"]["channel_format"][0] == "int16"
        assert streams[0]["info"]["stream_id"] == 0

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
            file, jitter_break_threshold_seconds=0.001, jitter_break_threshold_samples=1
        )
        assert streams[0]["info"]["segments"] == [(0, 0), (1, 3), (4, 8)]
