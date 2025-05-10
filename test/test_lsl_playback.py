import importlib
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "example-files" / "minimal.xdf"


@pytest.mark.skipif(not path.exists(), reason="File not found.")
@pytest.mark.skipif(
    not importlib.util.find_spec("pylsl"), reason="requires the pylsl library"
)
def test_lsl_playback():
    """
    Test the LSL playback functionality.
    """
    from pyxdf.cli.playback_lsl import main as playback_main

    playback_main(str(path), playback_speed=10.0, loop=False, wait_for_consumer=False)
