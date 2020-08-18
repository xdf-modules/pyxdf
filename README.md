[![Latest PyPI Release](https://img.shields.io/pypi/v/pyxdf)](https://pypi.org/project/pyxdf/)
[![Latest Conda Release](https://img.shields.io/conda/vn/conda-forge/pyxdf)](https://anaconda.org/conda-forge/pyxdf)
![Python 3.5+](https://img.shields.io/badge/python-3.5+-green.svg)
![License](https://img.shields.io/github/license/xdf-modules/xdf-python)

pyXDF
=====

pyXDF is a Python importer for [XDF](https://github.com/sccn/xdf) files.

## Sample usage

``` python
import pyxdf
import matplotlib.pyplot as plt
import numpy as np

data, header = pyxdf.load_xdf('test.xdf')

for stream in data:
    y = stream['time_series']

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        plt.plot(stream['time_stamps'], y)
    else:
        raise RuntimeError('Unknown stream format')

plt.show()
```

## Installation

The latest stable version can be installed with `pip install pyxdf`.

For the latest development version, use `pip install git+https://github.com/xdf-modules/pyxdf.git`.

## For maintainers

A new release is automatically uploaded to PyPI. Therefore, as soon as a new release is created on GitHub (using a tag labeled e.g. `v1.16.3`), a PyPI package is created with the version number matching the release tag.
