import numpy as np
cimport numpy as np


# Apply exponential smoothing for attack/release
cpdef np.ndarray apply_gain_reduction(np.ndarray[double, ndim=1] gain_reduction, double attack_coeff, double release_coeff):
    cdef int i
    cdef int n = gain_reduction.shape[0]

    # Loop through the array to apply smoothing for attack and release phases
    for i in range(1, n):
        if gain_reduction[i] < gain_reduction[i - 1]:
            # Attack phase
            gain_reduction[i] = attack_coeff * gain_reduction[i - 1] + (1 - attack_coeff) * gain_reduction[i]
        else:
            # Release phase
            gain_reduction[i] = release_coeff * gain_reduction[i - 1] + (1 - release_coeff) * gain_reduction[i]

    return gain_reduction
