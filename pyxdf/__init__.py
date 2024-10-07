# Authors: Christian Kothe & the Intheon pyxdf team
#          Clemens Brunner
#
# License: BSD (2-clause)

try:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:  # package is not installed
        __version__ = None
except ImportError:  # pkg_resources is not available
    __version__ = None
from .pyxdf import load_xdf, resolve_streams, match_streaminfos, align_streams

__all__ = [load_xdf, resolve_streams, match_streaminfos, align_streams]
