# Authors: Christian Kothe & the Intheon pyxdf team
#          Tristan Stenner
#          Chadwick Boulay
#
# License: BSD (2-clause)
import argparse
import logging
from os.path import abspath, dirname, join

import pyxdf


def main(fname: str):
    logging.basicConfig(level=logging.DEBUG)  # Use logging.INFO to reduce output
    streams, _ = pyxdf.load_xdf(fname)

    print("Found {} streams:".format(len(streams)))
    for ix, stream in enumerate(streams):
        msg = (
            "Stream {}: {} - type {} - uid {} - shape {} in {} segments at {} "
            "(effective {}) Hz"
        )
        print(
            msg.format(
                ix + 1,
                stream["info"]["name"][0],
                stream["info"]["type"][0],
                stream["info"]["uid"][0],
                (int(stream["info"]["channel_count"][0]), len(stream["time_stamps"])),
                len(stream["info"]["segments"]),
                stream["info"]["nominal_srate"][0],
                stream["info"]["effective_srate"],
            )
        )
        if any(stream["time_stamps"]):
            duration = stream["time_stamps"][-1] - stream["time_stamps"][0]
            print("\tDuration: {} s".format(duration))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a XDF file's metadata")
    parser.add_argument(
        "-f",
        type=str,
        help="Path to the XDF file",
        default=abspath(join(dirname(__file__), "..", "..", "..", "xdf_sample.xdf")),
    )
    args = parser.parse_args()
    main(args.f)
