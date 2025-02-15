import numpy as np
cimport numpy as np


def smooth_gain_reduction(np.ndarray[np.float64_t, ndim=1] gain_reduction, double attack_coeff, double release_coeff):
    """
    Apply exponential smoothing for attack and release phases to the gain reduction array.

    Args:
        gain_reduction (np.ndarray): The input gain reduction values as a 1D array.
        attack_coeff (float): The attack coefficient for smoothing.
        release_coeff (float): The release coefficient for smoothing.

    Returns:
        np.ndarray: The smoothed gain reduction values.
    """
    cdef int i
    cdef int n = gain_reduction.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] smoothed_gain_reduction = np.empty(n, dtype=np.float64)
    smoothed_gain_reduction[0] = gain_reduction[0]  # Initialize with the first value

    for i in range(1, n):
        if gain_reduction[i] < smoothed_gain_reduction[i - 1]:
            # Attack phase: gain reduction is rising
            smoothed_gain_reduction[i] = attack_coeff * smoothed_gain_reduction[i - 1] + (1 - attack_coeff) * gain_reduction[i]
        else:
            # Release phase: gain reduction is falling
            smoothed_gain_reduction[i] = release_coeff * smoothed_gain_reduction[i - 1] + (1 - release_coeff) * gain_reduction[i]

    return smoothed_gain_reduction
