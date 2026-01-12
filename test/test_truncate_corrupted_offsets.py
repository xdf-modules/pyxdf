"""Tests for _truncate_corrupted_offsets (pylsl#67 bug fix).

These tests verify the automatic detection and truncation of extra samples
and corrupted clock offsets caused by the pylsl#67 bug, where destroying an
LSL outlet while an inlet is still connected causes an extra sample with
potentially garbage clock offset values to be recorded.

See: https://github.com/labstreaminglayer/pylsl/issues/67
"""

import numpy as np
from mock_data_stream import MockStreamData

from pyxdf.pyxdf import _clock_sync, _truncate_corrupted_offsets


def test_no_truncation_when_sample_count_matches():
    """Normal data with matching sample count should not be modified."""
    n_samples = 100
    n_offsets = 20
    clock_tdiff = 5

    # Create normal stream data
    time_stamps = np.linspace(0, 100, n_samples)
    clock_times = [i * clock_tdiff for i in range(n_offsets)]
    clock_values = [1e-6 * (i % 3 - 1) for i in range(n_offsets)]  # Small jitter

    temp = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    streams = {1: {"footer": {"info": {"sample_count": [str(n_samples)]}}}}

    # Store original values for comparison
    orig_timestamps = temp[1].time_stamps.copy()
    orig_clock_times = temp[1].clock_times.copy()
    orig_clock_values = temp[1].clock_values.copy()

    # Apply truncation
    temp = _truncate_corrupted_offsets(temp, streams)

    # Verify nothing was modified
    np.testing.assert_array_equal(temp[1].time_stamps, orig_timestamps)
    assert temp[1].clock_times == orig_clock_times
    assert temp[1].clock_values == orig_clock_values


def test_no_truncation_when_clock_offsets_normal():
    """Extra samples should NOT be truncated when clock offsets are normal.

    This tests the case where footer sample_count is off-by-1 due to race
    conditions (e.g., final sample arrives after footer is written), but the
    data is valid. Without evidence of corruption (corrupted clock offset),
    the extra samples should be preserved.
    """
    n_samples = 100
    n_offsets = 20
    clock_tdiff = 5

    # Create stream with extra sample (n_samples + 1) but normal clock offsets
    time_stamps = np.linspace(0, 101, n_samples + 1)  # One extra sample
    clock_times = [i * clock_tdiff for i in range(n_offsets)]
    clock_values = [1e-6 * (i % 3 - 1) for i in range(n_offsets)]  # Normal values

    temp = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    # Footer says n_samples, but we have n_samples + 1
    streams = {1: {"footer": {"info": {"sample_count": [str(n_samples)]}}}}

    # Store original values
    orig_timestamps = temp[1].time_stamps.copy()
    orig_clock_times = temp[1].clock_times.copy()
    orig_clock_values = temp[1].clock_values.copy()

    # Apply truncation
    temp = _truncate_corrupted_offsets(temp, streams)

    # Verify samples were NOT truncated (clock offsets are normal, no evidence
    # of pylsl#67, liblsl#246 bug, so we keep the extra samples)
    np.testing.assert_array_equal(temp[1].time_stamps, orig_timestamps)
    assert len(temp[1].time_stamps) == n_samples + 1

    # Verify clock offsets were NOT truncated
    assert temp[1].clock_times == orig_clock_times
    assert temp[1].clock_values == orig_clock_values


def test_no_truncation_when_no_extra_samples():
    """Anomalous clock offset should NOT be truncated without extra samples.

    This tests the case where clock offsets look anomalous (e.g., due to
    legitimate clock resets) but there are no extra samples. Without extra
    samples, we have no evidence of the pylsl#67 bug, so we should not
    truncate the clock offset.
    """
    n_samples = 100
    n_offsets = 20
    clock_tdiff = 5
    clock_offset_value = -0.001  # Normal offset: -1ms

    # Create stream with matching sample count but "anomalous" last clock offset
    # This could be a legitimate clock reset, not pylsl#67 - liblsl#246 corruption
    time_stamps = np.linspace(0, 100, n_samples)
    clock_times = [i * clock_tdiff for i in range(n_offsets)]
    clock_values = [clock_offset_value] * n_offsets

    # Add an "anomalous" clock offset (could be legitimate clock reset)
    clock_times.append(clock_times[-1] + 800000)  # Huge time jump
    clock_values.append(-750000)  # Huge value change

    temp = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    # Footer matches sample count (no extra samples)
    streams = {1: {"footer": {"info": {"sample_count": [str(n_samples)]}}}}

    # Store original values
    orig_timestamps = temp[1].time_stamps.copy()
    orig_clock_times = temp[1].clock_times.copy()
    orig_clock_values = temp[1].clock_values.copy()

    # Apply truncation
    temp = _truncate_corrupted_offsets(temp, streams)

    # Verify NOTHING was truncated (no extra samples = no evidence of pylsl#67)
    np.testing.assert_array_equal(temp[1].time_stamps, orig_timestamps)
    assert temp[1].clock_times == orig_clock_times
    assert temp[1].clock_values == orig_clock_values


def test_truncation_of_samples_and_corrupted_clock_offset():
    """Extra sample and corrupted clock offset should both be truncated."""
    n_samples = 100
    n_offsets = 20
    clock_tdiff = 5
    clock_offset_value = -0.001  # Normal offset: -1ms

    # Create stream with extra sample AND corrupted last clock offset
    time_stamps = np.linspace(0, 101, n_samples + 1)  # One extra sample
    clock_times = [i * clock_tdiff for i in range(n_offsets)]
    clock_values = [clock_offset_value] * n_offsets

    # Corrupt the last clock offset (mimics the real bug pattern)
    clock_times.append(clock_times[-1] + 800000)  # Huge time jump
    clock_values.append(-750000)  # Huge corrupted value

    temp = {
        1: MockStreamData(
            time_stamps=time_stamps,
            clock_times=clock_times,
            clock_values=clock_values,
        )
    }
    streams = {1: {"footer": {"info": {"sample_count": [str(n_samples)]}}}}

    # Apply truncation
    temp = _truncate_corrupted_offsets(temp, streams)

    # Verify samples were truncated
    assert len(temp[1].time_stamps) == n_samples

    # Verify corrupted clock offset was removed
    assert len(temp[1].clock_times) == n_offsets
    assert len(temp[1].clock_values) == n_offsets

    # Verify the remaining clock values are the original normal ones
    assert all(v == clock_offset_value for v in temp[1].clock_values)

    # Verify clock sync now works correctly (regression doesn't get corrupted)
    temp = _clock_sync(temp)
    expected_timestamps = (
        np.linspace(0, 101, n_samples + 1)[:n_samples] + clock_offset_value
    )
    np.testing.assert_allclose(
        temp[1].time_stamps,
        expected_timestamps,
        atol=1e-6,
    )
