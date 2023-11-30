from pyxdf.pyxdf import _shift_align
import numpy as np
import pytest
old_timestamps = np.linspace(1.0, 1.5, 51)
old_timeseries = np.empty((51,1))    
old_timeseries[:,0] = np.linspace(0, 50, 51)

def test_shift_align_too_few_new_stamps():
    # not all old samples were assigned
    new_timestamps = np.linspace(1.001, 1.5001, 50)
    with pytest.raises(RuntimeError):
        new_timeseries = _shift_align(old_timestamps, old_timeseries, new_timestamps)

def test_shift_align_slightly_later():
    print("\n==================")
    new_timestamps = np.arange(1.001, 1.5011, 0.01)
    new_timeseries = _shift_align(old_timestamps, old_timeseries, new_timestamps)
    for x, y, xhat, yhat in zip(old_timestamps, old_timeseries, new_timestamps, new_timeseries):
        print(f"{x:3.4f} -> {xhat:3.4f} = {y[0]:3.0f} / {yhat[0]:3.0f} ")
        assert y == yhat
    
def test_shift_align_slightly_earlier():
    print("\n==================")    
    new_timestamps = np.arange(0.999, 1.499, 0.01)
    new_timeseries= _shift_align(old_timestamps, old_timeseries, new_timestamps)
    
    for x, y, xhat, yhat in zip(old_timestamps, old_timeseries, new_timestamps, new_timeseries):
        print(f"{x:3.4f} -> {xhat:3.4f} = {y[0]:3.0f} / {yhat[0]:3.0f} ")
        assert y == yhat

def test_shift_align_jittered():
    print("\n==================")
    jittered_timestamps = np.random.randn(*old_timestamps.shape)*0.0005
    jittered_timestamps += old_timestamps
    new_timeseries = _shift_align(jittered_timestamps, old_timeseries, old_timestamps)
    for x, y, xhat, yhat in zip(jittered_timestamps, old_timeseries, old_timestamps, new_timeseries):
        print(f"{x:3.4f} -> {xhat:3.4f} = {y[0]:3.0f} / {yhat[0]:3.0f} ")
        assert y == yhat

def test_shift_align_edges():
    print("\n==================")
    new_timestamps = np.arange(0.5, 1.7, 0.01)
    # idx = np.argwhere(old_timestamps<1.2)[-1][0]+1 
    idx = len(old_timestamps)
    new_timeseries = _shift_align(old_timestamps[:idx], old_timeseries, new_timestamps)    
    for xhat, yhat in zip(new_timestamps, new_timeseries):        
        delta = xhat-old_timestamps
        idx_old = np.argmin(np.abs(delta))
        x = old_timestamps[idx_old]        
        if xhat < 1.00 or xhat > 1.5001:
            y = np.nan
        else:
            y = old_timeseries[idx_old,0]
        print(f"{x:3.4f} -> {xhat:3.4f} = {y:3.0f} / {yhat[0]:3.0f} ")
        if (np.isnan(y)):
            assert np.isnan(yhat[0])
        else:
            assert y == yhat[0]