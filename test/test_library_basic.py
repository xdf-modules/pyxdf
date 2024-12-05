import io

import pytest

import pyxdf
import pyxdf.pyxdf


# %% test
def test_load_xdf_present():
    """
    Check that pyxdf has the all important load_xdf.
    This is nothing more than a placeholder so the CI system has a test to pass.
    """
    assert hasattr(pyxdf, "load_xdf")


def test_read_varlen_int():
    """"""

    def vla(data: bytes):
        return pyxdf.pyxdf._read_varlen_int(io.BytesIO(data))

    assert vla(b"\x01\xfd") == 0xFD
    assert vla(b"\x04\xfd\x12\x00\x34") == 0x340012FD
    assert vla(b"\x08\xfd\x12\x00\x34\x12\x34\x56\x78") == 0x78563412340012FD
    with pytest.raises(RuntimeError):
        vla(b"\x00")


def test_load_from_memory():
    testfile = b"XDF:\01\n\02\00 \00\00\00<x/>"
    f = pyxdf.pyxdf.open_xdf(io.BytesIO(testfile))
    assert isinstance(f, io.BytesIO)
    assert f.read()[-4:] == b"<x/>"

    chunks = pyxdf.pyxdf.parse_xdf(io.BytesIO(testfile))
    assert len(chunks) == 1
    assert chunks[0]["stream_id"] == 32
