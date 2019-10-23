from pathlib import Path
from pyxdf import load_xdf
import pytest
import numpy as np


# requires git clone https://github.com/xdf-modules/example-files.git
# into the root xdf-python folder
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

        s = np.array([[192, 255, 238],
                      [12, 22, 32],
                      [13, 23, 33],
                      [14, 24, 34],
                      [15, 25, 35],
                      [12, 22, 32],
                      [13, 23, 33],
                      [14, 24, 34],
                      [15, 25, 35]], dtype=np.int16)
        t = np.array([5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
        np.testing.assert_array_equal(streams[0]["time_series"], s)
        np.testing.assert_array_almost_equal(streams[0]["time_stamps"], t)

        assert streams[1]["info"]["name"][0] == "SendDataString"
        assert streams[1]["info"]["type"][0] == "StringMarker"
        assert streams[1]["info"]["channel_count"][0] == "1"
        assert streams[1]["info"]["nominal_srate"][0] == "10"
        assert streams[1]["info"]["channel_format"][0] == "string"
        assert streams[1]["info"]["stream_id"] == 0x02C0FFEE

        s = [['<?xml version="1.0"?><info><writer>LabRecorder xdfwriter'
              '</writer><first_timestamp>5.1</first_timestamp><last_timestamp>'
              '5.9</last_timestamp><sample_count>9</sample_count>'
              '<clock_offsets><offset><time>50979.76</time><value>-.01</value>'
              '</offset><offset><time>50979.86</time><value>-.02</value>'
              '</offset></clock_offsets></info>'],
             ['Hello'],
             ['World'],
             ['from'],
             ['LSL'],
             ['Hello'],
             ['World'],
             ['from'],
             ['LSL']]
        t = np.array([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
        assert streams[1]["time_series"] == s
        np.testing.assert_array_almost_equal(streams[1]["time_stamps"], t)
