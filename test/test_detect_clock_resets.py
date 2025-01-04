import numpy as np
import pytest
from pyxdf.pyxdf import _detect_clock_resets

from mock_data_stream import MockStreamData

# Test error condition


@pytest.mark.parametrize("n_clock_offsets", [0, 1])
def test_detect_resets_length_error(n_clock_offsets):
    clock_times = list(range(0, n_clock_offsets))
    clock_values = [0] * n_clock_offsets
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    time_thresh_stds = 5
    time_thresh_secs = 5
    value_thresh_stds = 10
    value_thresh_secs = 1
    with pytest.raises(ValueError):
        _detect_clock_resets(
            stream,
            time_thresh_stds,
            time_thresh_secs,
            value_thresh_stds,
            value_thresh_secs,
        )


# No clock resets


@pytest.mark.parametrize(
    "n_clock_offsets, expected",
    [(n, [(0, n - 1)]) for n in range(2, 10)],
)
def test_clock_no_resets(n_clock_offsets, expected):
    clock_tdiff = 5
    duration = n_clock_offsets * clock_tdiff
    clock_times = list(range(0, duration, clock_tdiff))
    clock_values = [0] * n_clock_offsets
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    time_thresh_stds = 5
    time_thresh_secs = 5
    value_thresh_stds = 10
    value_thresh_secs = 1
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected


def test_clock_no_resets_with_jitter():
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    jit_std = 0.02
    rng = np.random.default_rng(9)
    clock_times = np.arange(0, duration, clock_tdiff)
    clock_times = (
        clock_times + rng.standard_normal(n_clock_offsets) * jit_std
    ).tolist()
    clock_values = (rng.standard_normal(n_clock_offsets) * jit_std).tolist()
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    time_thresh_stds = 5
    time_thresh_secs = 5
    value_thresh_stds = 10
    value_thresh_secs = 1
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == [(0, 4)]


# Clock resets


@pytest.mark.parametrize(
    "time_thresh_stds, time_thresh_secs, value_thresh_stds, value_thresh_secs, expected",
    [
        (5, 5, 10, 5, [(0, 0), (1, 4)]),
        (2.71e16, 5, 10, 5, [(0, 4)]),
        (5, 6, 10, 5, [(0, 4)]),
        (5, 5, 2.71e16, 5, [(0, 4)]),
        (5, 5, 10, 6, [(0, 4)]),
    ],
)
def test_clock_jumps_forward(
    time_thresh_stds,
    time_thresh_secs,
    value_thresh_stds,
    value_thresh_secs,
    expected,
):
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    clock_times = list(range(0, duration, clock_tdiff))
    clock_values = [0] * n_clock_offsets
    # Reset after first clock offset measurement
    clock_times[0] = -6
    clock_values[0] = 6
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected


@pytest.mark.parametrize(
    "time_thresh_stds, time_thresh_secs, value_thresh_stds, value_thresh_secs, expected",
    [
        (5, 5, 10, 5, [(0, 0), (1, 4)]),
        (142, 5, 10, 5, [(0, 4)]),
        (5, 6, 10, 5, [(0, 4)]),
        (5, 5, 1285, 5, [(0, 4)]),
        (5, 5, 10, 6, [(0, 4)]),
    ],
)
def test_clock_jumps_forward_with_jitter(
    time_thresh_stds,
    time_thresh_secs,
    value_thresh_stds,
    value_thresh_secs,
    expected,
):
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    jit_std = 0.02
    rng = np.random.default_rng(9)
    clock_times = np.arange(0, duration, clock_tdiff)
    clock_times = (
        clock_times + rng.standard_normal(n_clock_offsets) * jit_std
    ).tolist()
    clock_values = (rng.standard_normal(n_clock_offsets) * jit_std).tolist()
    # Reset after first clock offset measurement
    clock_times[0] = -6
    clock_values[0] = 6
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected


@pytest.mark.parametrize(
    "time_thresh_stds, time_thresh_secs, value_thresh_stds, value_thresh_secs, expected",
    [
        (5, 5, 10, 5, [(0, 4), (5, 9)]),
        (312, 5, 10, 5, [(0, 9)]),
        (5, 6, 10, 5, [(0, 9)]),
        (5, 5, 101, 5, [(0, 9)]),
        (5, 5, 10, 6, [(0, 9)]),
    ],
)
def test_clock_jumps_forward_with_jitter_eq_len(
    time_thresh_stds,
    time_thresh_secs,
    value_thresh_stds,
    value_thresh_secs,
    expected,
):
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    jit_std = 0.02
    rng = np.random.default_rng(9)
    clock_times = np.arange(0, duration * 2, clock_tdiff)
    clock_times[n_clock_offsets:] += 6
    clock_times = (
        clock_times + rng.standard_normal(len(clock_times)) * jit_std
    ).tolist()
    clock_values = rng.standard_normal(len(clock_times)) * jit_std
    clock_values[n_clock_offsets:] -= 6
    clock_values = clock_values.tolist()
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected


@pytest.mark.parametrize(
    "time_thresh_stds, time_thresh_secs, value_thresh_stds, value_thresh_secs, expected",
    [
        # Negative time intervals are always resets
        (np.inf, np.inf, np.inf, np.inf, [(0, 0), (1, 4)]),
    ],
)
def test_clock_jumps_backward(
    time_thresh_stds,
    time_thresh_secs,
    value_thresh_stds,
    value_thresh_secs,
    expected,
):
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    clock_times = list(range(0, duration, clock_tdiff))
    clock_values = [0] * n_clock_offsets
    # Reset after first clock offset measurement
    clock_times[0] = 6
    clock_values[0] = -6
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected


@pytest.mark.parametrize(
    "time_thresh_stds, time_thresh_secs, value_thresh_stds, value_thresh_secs, expected",
    [
        # Negative time intervals are always resets
        [np.inf, np.inf, np.inf, np.inf, [(0, 0), (1, 4)]],
    ],
)
def test_clock_jumps_backward_with_jitter(
    time_thresh_stds,
    time_thresh_secs,
    value_thresh_stds,
    value_thresh_secs,
    expected,
):
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    jit_std = 0.02
    rng = np.random.default_rng(9)
    clock_times = np.arange(0, duration, clock_tdiff)
    clock_times = (
        clock_times + rng.standard_normal(n_clock_offsets) * jit_std
    ).tolist()
    clock_values = (rng.standard_normal(n_clock_offsets) * jit_std).tolist()
    # Reset after first clock offset measurement
    clock_times[0] = 6
    clock_values[0] = -6
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected


@pytest.mark.parametrize(
    "time_thresh_stds, time_thresh_secs, value_thresh_stds, value_thresh_secs, expected",
    [
        # Negative time intervals are always resets
        [np.inf, np.inf, np.inf, np.inf, [(0, 4), (5, 9)]],
    ],
)
def test_clock_jumps_backward_with_jitter_eq_len(
    time_thresh_stds,
    time_thresh_secs,
    value_thresh_stds,
    value_thresh_secs,
    expected,
):
    clock_tdiff = 5
    n_clock_offsets = 5
    duration = n_clock_offsets * clock_tdiff
    jit_std = 0.02
    rng = np.random.default_rng(9)
    clock_times = np.arange(0, duration * 2, clock_tdiff)
    clock_times[n_clock_offsets:] -= 6
    clock_times = (
        clock_times + rng.standard_normal(len(clock_times)) * jit_std
    ).tolist()
    clock_values = rng.standard_normal(len(clock_times)) * jit_std
    clock_values[n_clock_offsets:] += 6
    clock_values = clock_values.tolist()
    stream = MockStreamData(
        clock_times=clock_times,
        clock_values=clock_values,
    )
    segments = _detect_clock_resets(
        stream,
        time_thresh_stds,
        time_thresh_secs,
        value_thresh_stds,
        value_thresh_secs,
    )
    # Inclusive range
    assert segments == expected
