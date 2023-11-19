from pyxdf.pyxdf import _shift_align
import numpy as np

def test_shift_align():
    old_timestamps = np.linspace(1, 1.5, 51)
    old_timeseries = np.empty((51,1))    
    old_timeseries[:,0] = np.linspace(0, 50, 51)
    new_timestamps = np.linspace(1.001, 1.5001, 51)
    new_timeseries = _shift_align(old_timestamps, old_timeseries, new_timestamps)
    for x, y, xhat in zip(old_timestamps, old_timeseries, new_timestamps):
        print(x, xhat, y[0])
    new_timestamps = np.linspace(0.99, 1.499, 51)
    _shift_align(old_timestamps, old_timeseries, new_timestamps)