import numpy as np
from .audio_dynamics import AudioDynamics


class PeakLimiter(AudioDynamics):
    """Peak limiter to cut off peaks and reduce the peak factor (crest factor) of the signal."""

    def __init__(self, threshold: float = -1.0, attack_time_ms: float = 0.1, release_time_ms: float = 1.0,
                 realtime=False):
        """
        Initialize the peak limiter.

        Args:
            threshold (float): The threshold level in dB. Defaults to -1.0.
            attack_time_ms (float): The attack time in milliseconds. Defaults to 0.1.
            release_time_ms (float): The release time in milliseconds. Defaults to 1.0.
            realtime (bool): True if the effect is used for real-time processing (in chunks). Defaults to False.
        """
        super().__init__(threshold, attack_time_ms, release_time_ms, realtime=realtime)

    def target_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the target gain reduction before attack/release smoothing for limiter.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).

        Returns:
            np.ndarray: The linear gain reduction values between 0 and 1.
        """
        max_amplitude = self._compute_max_amplitude(signal)
        max_amplitude = np.maximum(max_amplitude, 1e-10)  # Ensure max_amplitude is never zero

        return np.where(max_amplitude > self.threshold_linear, self.threshold_linear / max_amplitude, 1.0)
