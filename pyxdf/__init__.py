# Authors: Christian Kothe & the Intheon pyxdf team
#          Clemens Brunner
#
# License: BSD (2-clause)

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = None

from .pyxdf import load_xdf, match_streaminfos, resolve_streams

__all__ = [load_xdf, resolve_streams, match_streaminfos]
