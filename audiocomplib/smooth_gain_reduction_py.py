import numpy as np

def smooth_gain_reduction(gain_reduction: np.ndarray, attack_coeff: float, release_coeff: float) -> np.ndarray:
    """
    Apply exponential smoothing for attack and release phases to the gain reduction array.

    Args:
        gain_reduction (np.ndarray): The input gain reduction values as a 1D array.
        attack_coeff (float): The attack coefficient for smoothing.
        release_coeff (float): The release coefficient for smoothing.

    Returns:
        np.ndarray: The smoothed gain reduction values.
    """
    n = gain_reduction.shape[0]
    smoothed_gain_reduction = np.empty_like(gain_reduction)  # Create a new array to store the smoothed values
    smoothed_gain_reduction[0] = gain_reduction[0]  # Initialize with the first value

    # Loop through the array to apply smoothing for attack and release phases
    for i in range(1, n):
        if gain_reduction[i] < smoothed_gain_reduction[i - 1]:
            # Attack phase: gain reduction is rising
            smoothed_gain_reduction[i] = attack_coeff * smoothed_gain_reduction[i - 1] + (1 - attack_coeff) * gain_reduction[i]
        else:
            # Release phase: gain reduction is falling
            smoothed_gain_reduction[i] = release_coeff * smoothed_gain_reduction[i - 1] + (1 - release_coeff) * gain_reduction[i]

    return smoothed_gain_reduction