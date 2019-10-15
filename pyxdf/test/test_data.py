from pathlib import Path
from pyxdf import load_xdf
import pytest


path = Path("../../example-files")
extensions = ["*.xdf", "*.xdfz", "*.xdf.gz"]
files = []
for ext in extensions:
    files.extend(path.glob(ext))
files = [str(file) for file in files]


@pytest.mark.parametrize("file", files)
def test_load_file(file):
    load_xdf(file)
