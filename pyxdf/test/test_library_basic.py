import pyxdf
import pyxdf.pyxdf
import pytest
import io


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
