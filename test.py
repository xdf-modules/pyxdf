import matplotlib.pyplot as plt
import pyxdf

if __name__ == "__main__":
    fname = "/home/rtgugg/Downloads/sub-13_ses-S001_task-HCT_run-001_eeg.xdf"
    # streams, header = pyxdf.load_xdf(
    #     fname, select_streams=[2, 5]
    # )  # EEG and ACC streams

    # pyxdf.align_streams(streams)

    streams, header = pyxdf.load_xdf(fname, select_streams=[2])  # EEG stream
    plt.plot(streams[0]["time_stamps"])
    plt.show()
    pyxdf.align_streams(streams)
