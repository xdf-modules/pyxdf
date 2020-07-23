# Authors: Christian Kothe & the Intheon pyxdf team
#          Tristan Stenner
#
# License: BSD (2-clause)

from os.path import abspath, join, dirname
import logging
import pyxdf
import sys


logging.basicConfig(level=logging.DEBUG)  # Use logging.INFO to reduce output
if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = abspath(join(dirname(__file__), '..', '..', 'xdf_sample.xdf'))
streams, fileheader = pyxdf.load_xdf(fname)

print("Found {} streams:".format(len(streams)))
for ix, stream in enumerate(streams):
    msg = "Stream {}: {} - type {} - uid {} - shape {} at {} (effective {}) Hz"
    print(msg.format(
        ix + 1, stream['info']['name'][0],
        stream['info']['type'][0],
        stream['info']['uid'][0],
        (int(stream['info']['channel_count'][0]), len(stream['time_stamps'])),
        stream['info']['nominal_srate'][0],
        stream['info']['effective_srate'])
    )
    if any(stream['time_stamps']):
        duration = stream['time_stamps'][-1] - stream['time_stamps'][0]
        print("\tDuration: {} s".format(duration))
print("Done.")
