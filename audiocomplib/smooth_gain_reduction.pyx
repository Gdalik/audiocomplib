# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Cython-accelerated gain reduction smoothing (v0.2.0).

Fast exponential envelope smoothing with per-sample variable release times.
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp

ctypedef np.float64_t DTYPE_t


def smooth_gain_reduction(
    np.ndarray[DTYPE_t, ndim=1] target_gain_reduction,
    double attack_time_ms,
    np.ndarray[DTYPE_t, ndim=1] release_times_ms,
    int sample_rate,
    double last_gain_reduction = 1.0
):
    """
    Smooth gain reduction envelope with exponential attack/release.

    Applies first-order exponential smoothing to a target gain reduction curve.
    Attack time is constant for all samples. Release time is per-sample (variable).

    Args:
        target_gain_reduction (np.ndarray): Unsmoothed linear gain (0-1), shape (n_samples,).
        attack_time_ms (float): Attack time in milliseconds.
        release_times_ms (np.ndarray): Per-sample release times (ms), shape (n_samples,).
        sample_rate (int): Sample rate in Hz.
        last_gain_reduction (float): Previous gain value for continuity. Default: 1.0.

    Returns:
        np.ndarray: Smoothed gain reduction, shape (n_samples,).
    """
    cdef int n_samples = len(target_gain_reduction)
    cdef np.ndarray[DTYPE_t, ndim=1] smoothed = np.zeros(n_samples, dtype=np.float64)
    cdef double current_gain = last_gain_reduction
    cdef double attack_coeff, release_coeff
    cdef int attack_samples, release_samples
    cdef int i

    # Pre-calculate attack coefficient
    attack_samples = max(1, <int>(attack_time_ms * sample_rate / 1000.0))
    attack_coeff = exp(-1.0 / attack_samples)

    for i in range(n_samples):
        # Calculate release coefficient from per-sample release time
        release_samples = <int>(release_times_ms[i] * sample_rate / 1000.0)
        release_samples = max(1, release_samples)
        release_coeff = exp(-1.0 / release_samples)

        # Attack or release
        if target_gain_reduction[i] < current_gain:
            current_gain = attack_coeff * current_gain + (1 - attack_coeff) * target_gain_reduction[i]
        else:
            current_gain = release_coeff * current_gain + (1 - release_coeff) * target_gain_reduction[i]

        smoothed[i] = current_gain

    return smoothed
