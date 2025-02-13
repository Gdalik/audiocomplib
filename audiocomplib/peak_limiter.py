import numpy as np
from audiocomplib.apply_gain_reduction import apply_gain_reduction  # Import Cython function


class PeakLimiter:
    def __init__(self, threshold: float = -1.0, attack_time_ms: float = 0.1, release_time_ms: float = 1.0):
        """
        Initialize the PeakLimiter with limiting parameters.

        Parameters:
        - threshold: Threshold level in dB (default: -1.0 dB)
        - attack_time_ms: Attack time in milliseconds (default: 0.1 ms)
        - release_time_ms: Release time in milliseconds (default: 1.0 ms)
        """
        self.threshold: float = threshold
        self.attack_time_ms: float = attack_time_ms
        self.release_time_ms: float = release_time_ms
        self._gain_reduction: np.ndarray | None = None  # Store gain reduction in linear scale (not dBFS)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold in dB for the limiter."""
        self.threshold = threshold

    def set_attack_time(self, attack_time_ms: float) -> None:
        """Set the attack time in milliseconds."""
        self.attack_time_ms = attack_time_ms

    def set_release_time(self, release_time_ms: float) -> None:
        """Set the release time in milliseconds."""
        self.release_time_ms = release_time_ms

    def _calculate_gain_reduction(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Calculate the gain reduction during the limiting process.

        Parameters:
        - signal: numpy array with shape (channels, samples)
        - sample_rate: Sample rate of the audio signal (e.g., 44100 Hz)

        Returns:
        - _gain_reduction: numpy array with gain reduction values in linear scale (0 to 1)
        """
        if signal.ndim != 2:
            raise ValueError("Input signal must be a 2D array with shape (channels, samples).")

        channels, samples = signal.shape

        # Convert threshold from dB to linear scale
        threshold_linear = 10 ** (self.threshold / 20)

        # Convert attack and release times from milliseconds to samples
        attack_time_samples = int(self.attack_time_ms * sample_rate / 1000)
        release_time_samples = int(self.release_time_ms * sample_rate / 1000)

        # Attack and release coefficients
        attack_coeff = np.exp(-1 / attack_time_samples) if attack_time_samples > 0 else 0
        release_coeff = np.exp(-1 / release_time_samples) if release_time_samples > 0 else 0

        # Precompute max amplitude across all channels for each sample
        max_amplitude = np.max(np.abs(signal), axis=0)

        # Suppress divide-by-zero warnings for log10 calculation
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate gain reduction based on threshold
        gain_reduction = np.where(max_amplitude > threshold_linear,
                                  threshold_linear / max_amplitude, 1.0)

        # Apply attack/release smoothing (call Cython function)
        gain_reduction_smooth = apply_gain_reduction(gain_reduction.astype(np.float64), attack_coeff, release_coeff)

        # Store gain reduction for future reference
        self._gain_reduction = gain_reduction_smooth

        return gain_reduction_smooth

    def process(self, input_signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Process the input audio signal with the peak limiter.

        Parameters:
        - input_signal: numpy array with shape (channels, samples)
        - sample_rate: Sample rate of the audio signal (e.g., 44100 Hz)

        Returns:
        - compressed_signal: numpy array with compressed audio signal
        """
        # Calculate the gain reduction during the process
        gain_reduction_smooth = self._calculate_gain_reduction(input_signal, sample_rate)

        # Apply the gain reduction to the signal (in linear scale)
        compressed_signal = input_signal * gain_reduction_smooth

        return compressed_signal

    def get_gain_reduction(self) -> np.ndarray:
        """
        Retrieve the stored gain reduction values in dBFS.

        Returns:
        - gain_reduction_dbfs: numpy array with gain reduction values in dBFS
        """
        if self._gain_reduction is None:
            raise ValueError("Gain reduction has not been calculated yet. Please process a signal first.")

        # Convert linear scale gain reduction to dBFS for output
        gain_reduction_dbfs = 20 * np.log10(self._gain_reduction)

        return gain_reduction_dbfs
