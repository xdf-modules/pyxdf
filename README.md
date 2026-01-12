[![Latest PyPI Release](https://img.shields.io/pypi/v/pyxdf)](https://pypi.org/project/pyxdf/)
[![Latest Conda Release](https://img.shields.io/conda/vn/conda-forge/pyxdf)](https://anaconda.org/conda-forge/pyxdf)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/github/license/xdf-modules/xdf-python)

# PyXDF

PyXDF is a Python importer for [XDF](https://github.com/sccn/xdf) files.

## Sample usage

The following example loads `minimal.xdf` (which you can download from [here](https://raw.githubusercontent.com/xdf-modules/example-files/master/minimal.xdf)) and prints/plots all markers: 

``` python
import matplotlib.pyplot as plt
import numpy as np

import pyxdf

data, header = pyxdf.load_xdf("minimal.xdf")

for stream in data:
    y = stream["time_series"]

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream["time_stamps"], y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        plt.plot(stream["time_stamps"], y)
    else:
        raise RuntimeError("Unknown stream format")

plt.show()
```

Note that you need to have `matplotlib` installed to run this example (which is not a dependency of PyXDF itself).

## Command line interface (CLI)

PyXDF has a command line interface (CLI) with the following commands:

* `print_metadata` will print basic metadata for each found stream.
    * `python -m pyxdf.cli.print_metadata -f=/path/to/my.xdf`
* `playback_lsl` will open an XDF file, then replay its data in an infinite loop, but using current timestamps (this is useful for prototyping online processing):
    * `python -m pyxdf.cli.playback_lsl /path/to/my.xdf --loop`

## Installation

The latest stable version can be installed with `pip install pyxdf`.

For the latest development version, use `pip install git+https://github.com/xdf-modules/pyxdf.git`.

## For maintainers

As soon as a new release is created on GitHub (using a tag labeled e.g. `v1.16.3`), a [PyPI package](https://pypi.org/project/pyxdf/) is automatically created with the version number matching the release tag.
