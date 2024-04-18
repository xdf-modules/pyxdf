# Authors: Christian Kothe & the Intheon pyxdf team
#          Chadwick Boulay
#          Clemens Brunner
#
# License: BSD (2-clause)

"""Python setup script for the pyxdf distribution package."""

from setuptools import setup, find_packages
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyxdf",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Python library for importing XDF (Extensible Data Format)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xdf-modules/xdf-Python",
    author="Christian Kothe",
    author_email="christian.kothe@intheon.io",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "XDF",
        "pyxdf",
        "LSL",
        "Lab Streaming Layer",
        "file format",
        "biosignals",
        "stream",
    ],
    packages=find_packages(exclude=["pyxdf.test*"]),
    install_requires=["numpy"],
    extras_require={},
    package_data={},
    data_files=[],
    entry_points={},
)
