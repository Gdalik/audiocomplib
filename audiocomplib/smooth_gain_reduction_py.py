import numpy as np


def smooth_gain_reduction(gain_reduction: np.ndarray, attack_coeff: float, release_coeff: float,
                          last_gain_reduction=None) -> np.ndarray:
    """
    Apply exponential smoothing for attack and release phases to the gain reduction array.

    Args:
        gain_reduction (np.ndarray): The input gain reduction values as a 1D array.
        attack_coeff (float): The attack coefficient for smoothing.
        release_coeff (float): The release coefficient for smoothing.
        last_gain_reduction (float or None): The last gain reduction value from the previous chunk (if provided).

    Returns:
        np.ndarray: The smoothed gain reduction values.
    """
    n = gain_reduction.shape[0]
    smoothed_gain_reduction = np.empty_like(gain_reduction)  # Create a new array to store the smoothed values

    # Initialize the first sample
    if last_gain_reduction is None:
        # If no previous state is provided, start with the first input/target gain reduction value
        smoothed_gain_reduction[0] = gain_reduction[0]
    else:
        # If previous state is provided, start smoothing from index 0
        coeff = attack_coeff if gain_reduction[0] < last_gain_reduction else release_coeff
        smoothed_gain_reduction[0] = coeff * last_gain_reduction + (1 - coeff) * gain_reduction[0]

    # Loop through the array to apply smoothing for attack and release phases
    for i in range(1, n):
        prev = smoothed_gain_reduction[i-1]
        target = gain_reduction[i]
        if target < prev:
            # Attack phase: gain reduction is rising
            smoothed_gain_reduction[i] = attack_coeff * prev + (1 - attack_coeff) * target
        else:
            # Release phase: gain reduction is falling
            smoothed_gain_reduction[i] = release_coeff * prev + (1 - release_coeff) * target

    return smoothed_gain_reduction