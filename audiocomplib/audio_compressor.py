import numpy as np
from typing import Optional
from audiocomplib.apply_gain_reduction import apply_gain_reduction  # Import Cython function


class AudioCompressor:
    def __init__(self, threshold: float = -10.0, ratio: float = 4.0, attack_time_ms: float = 1.0,
                 release_time_ms: float = 100.0, knee_width: float = 3.0):
        """
        Initialize the AudioCompressor with compression parameters.

        Parameters:
        - threshold: Threshold level in dB (default -10 dB)
        - ratio: Compression ratio (default 4:1)
        - attack_time_ms: Attack time in milliseconds (default 1 ms)
        - release_time_ms: Release time in milliseconds (default 100 ms)
        - knee_width: Soft knee width in dB (default 3 dB)
        """

        self.threshold = threshold
        self.ratio = ratio
        self.attack_time_ms = attack_time_ms
        self.release_time_ms = release_time_ms
        self.knee_width = knee_width
        self._gain_reduction: Optional[np.ndarray] = None  # Store gain reduction in linear scale (not dBFS)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold in dB for compression."""
        self.threshold = threshold

    def set_ratio(self, ratio: float) -> None:
        """Set the compression ratio (e.g., 4:1)."""
        self.ratio = ratio

    def set_attack_time(self, attack_time_ms: float) -> None:
        """Set the attack time in milliseconds."""
        self.attack_time_ms = attack_time_ms

    def set_release_time(self, release_time_ms: float) -> None:
        """Set the release time in milliseconds."""
        self.release_time_ms = release_time_ms

    def set_knee_width(self, knee_width: float) -> None:
        """Set the soft knee width in dB."""
        self.knee_width = knee_width

    def _compute_compression_factor(self, amplitude_dB: np.ndarray) -> np.ndarray:
        """Calculate the compression factor with soft knee logic."""
        knee_start = self.threshold - self.knee_width / 2
        compression_factor = np.where(
            (amplitude_dB > knee_start) & (amplitude_dB < self.threshold),
            1 + (self.ratio - 1) * ((amplitude_dB - knee_start) / self.knee_width),
            np.where(amplitude_dB >= self.threshold, self.ratio, 1)
        )
        return compression_factor

    def _calculate_gain_reduction(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Calculate the gain reduction during the compression process.

        Parameters:
        - signal: numpy array with shape (channels, samples)
        - sample_rate: Sample rate of the audio signal (e.g., 44100 Hz)

        Returns:
        - _gain_reduction: numpy array with gain reduction values in linear scale (0 to 1)
        """
        if signal.ndim != 2:
            raise ValueError("Input signal must be a 2D array with shape (channels, samples).")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive value.")

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

        # Convert amplitude to dB for each sample
        amplitude_dB = 20 * np.log10(np.maximum(max_amplitude, 1e-10))  # Avoid log(0)

        # Soft knee logic and compression calculation
        compression_factor = self._compute_compression_factor(amplitude_dB)

        # Compute desired gain reduction in linear scale (no dB conversion yet)
        desired_gain_reduction = np.where(
            (amplitude_dB > self.threshold),
            threshold_linear * (max_amplitude / threshold_linear) ** (1 / compression_factor),
            max_amplitude
        )


        # Apply attack/release smoothing
        target_gain_reduction = np.where(max_amplitude != 0, desired_gain_reduction / max_amplitude, 1.0)

        gain_reduction = apply_gain_reduction(target_gain_reduction,
                                              attack_coeff, release_coeff)

        # Store gain reduction in linear scale for internal use
        self._gain_reduction = gain_reduction

        return self._gain_reduction

    def process(self, input_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process the input audio signal and apply compression.

        Parameters:
        - input_signal: numpy array with shape (channels, samples)
        - sample_rate: Sample rate of the audio signal (e.g., 44100 Hz)

        Returns:
        - compressed_signal: numpy array with compressed audio signal
        """
        # Calculate gain reduction during the process
        self._calculate_gain_reduction(input_signal, sample_rate)

        # Apply the gain reduction to the signal (linear scale)
        compressed_signal = input_signal * self._gain_reduction

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
