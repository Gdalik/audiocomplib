"""
Pure Python fallback for smooth_gain_reduction (v0.2.0).

Identical behavior to Cython version.
Expected performance: ~50-100ms per 1M samples (vs ~5ms for Cython).
"""

import numpy as np


def smooth_gain_reduction(
    target_gain_reduction,
    attack_time_ms,
    release_times_ms,
    sample_rate,
    last_gain_reduction=1.0
):
    """
    Smooth gain reduction with per-sample release times (Python fallback).

    See smooth_gain_reduction.pyx for full documentation.
    """
    n_samples = len(target_gain_reduction)
    smoothed = np.zeros(n_samples, dtype=np.float64)
    current_gain = last_gain_reduction

    # Pre-calculate attack coefficient
    attack_samples = max(1, int(attack_time_ms * sample_rate / 1000.0))
    attack_coeff = np.exp(-1.0 / attack_samples)

    for i in range(n_samples):
        # Calculate release coefficient
        release_samples = max(1, int(release_times_ms[i] * sample_rate / 1000.0))
        release_coeff = np.exp(-1.0 / release_samples)

        # Attack or release
        if target_gain_reduction[i] < current_gain:
            current_gain = (
                attack_coeff * current_gain
                + (1 - attack_coeff) * target_gain_reduction[i]
            )
        else:
            current_gain = (
                release_coeff * current_gain
                + (1 - release_coeff) * target_gain_reduction[i]
            )

        smoothed[i] = current_gain

    return smoothed
