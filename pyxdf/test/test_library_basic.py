import pyxdf
import pyxdf.pyxdf
import pytest
import io
import os


#%% test
def test_load_xdf_present():
    """
    Check that pyxdf has the all important load_xdf.
    This is nothing more than a placeholder so the CI system has a test to pass.
    """
    assert(hasattr(pyxdf, 'load_xdf'))


def test_read_varlen_int():
    """"""

    def vla(data: bytes):
        return pyxdf.pyxdf._read_varlen_int(io.BytesIO(data))

    assert vla(b'\x01\xfd') == 0xfd
    assert vla(b'\x04\xfd\x12\x00\x34') == 0x340012fd
    assert vla(b'\x08\xfd\x12\x00\x34\x12\x34\x56\x78') == 0x78563412340012fd
    with pytest.raises(RuntimeError):
        vla(b'\x00')


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_files')


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, 'xdf_sample.xdf'))
def test_load_sample(datafiles):
    for xdf_file in datafiles.listdir():
        streams, fileheader = pyxdf.load_xdf(xdf_file)

        stream_names = []
        for strm in streams:
            assert('info' in strm)
            assert('name' in strm['info'])
            assert(isinstance(strm['info']['name'], list))
            stream_names.append(strm['info']['name'][0])
            assert(stream_names[-1] in ['MousePosition', 'MouseButtons',
                                        'Keyboard', 'AudioCaptureWin'])

            assert('time_series' in strm)
            import numpy as np
            assert(isinstance(strm['time_series'], (np.ndarray, list)))
            assert('time_stamps' in strm)
            assert(isinstance(strm['time_stamps'], np.ndarray))
            if isinstance(strm['time_series'], np.ndarray):
                assert(strm['time_stamps'].shape[0] == strm['time_series'].shape[0])
            else:
                assert(strm['time_stamps'].shape[0] == len(strm['time_series']))
