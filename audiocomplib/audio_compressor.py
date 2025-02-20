import numpy as np
from .audio_dynamics import AudioDynamics
from .smooth_gain_reduction_init import smooth_gain_reduction


class AudioCompressor(AudioDynamics):
    """Audio compressor for dynamic range compression."""

    def __init__(self, threshold: float = -10.0, ratio: float = 4.0, attack_time_ms: float = 1.0,
                 release_time_ms: float = 100.0, knee_width: float = 3.0, makeup_gain: float = 0.0, realtime=False):
        """
        Initialize the audio compressor.

        Args:
            threshold (float): The threshold level in dB. Defaults to -10.0.
            ratio (float): The compression ratio. Defaults to 4.0.
            attack_time_ms (float): The attack time in milliseconds. Defaults to 1.0.
            release_time_ms (float): The release time in milliseconds. Defaults to 100.0.
            knee_width (float): The knee width in dB for soft knee compression. Defaults to 3.0.
            makeup_gain (float): The make-up gain in dB. Defaults to 0.0
            realtime (bool): True if the effect is used for real-time processing (in chunks). Defaults to False.
        """
        super().__init__(threshold, attack_time_ms, release_time_ms, realtime=realtime)
        self.ratio = ratio
        self.knee_width = knee_width
        self.makeup_gain = makeup_gain

    def set_ratio(self, ratio: float) -> None:
        """
        Set the compression ratio.

        Args:
            ratio (float): The new compression ratio.
        """
        self.ratio = ratio

    def set_knee_width(self, knee_width: float) -> None:
        """
        Set the knee width for soft knee compression.

        Args:
            knee_width (float): The new knee width in dB.
        """
        self.knee_width = knee_width

    def set_makeup_gain(self, makeup_gain: float) -> None:
        """
        Set the make-up gain after the compression

        Args:
             makeup_gain (float): The new make-up gain in dB.
        """
        self.makeup_gain = makeup_gain

    def process(self, input_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        result = super().process(input_signal, sample_rate)

        # Calculate linear make-up gain and apply it to the output
        gain_k = 10 ** (self.makeup_gain / 20)
        return result * gain_k

    def _compute_compression_factor(self, amplitude_dB: np.ndarray) -> np.ndarray:
        """
        Compute the compression factor based on the input amplitude in dB.

        Args:
            amplitude_dB (np.ndarray): The input amplitude in dB.

        Returns:
            np.ndarray: The compression factor for each sample.
        """
        if self.knee_width == 0:
            # Hard knee: no smooth transition, just apply ratio above threshold
            return np.where(amplitude_dB >= self.threshold, self.ratio, 1)
        else:
            # Soft knee: apply smooth transition around the threshold
            knee_start = self.threshold - self.knee_width / 2
            return np.where(
                (amplitude_dB > knee_start) & (amplitude_dB < self.threshold),
                1 + (self.ratio - 1) * ((amplitude_dB - knee_start) / self.knee_width),
                np.where(amplitude_dB >= self.threshold, self.ratio, 1)
            )

    def _calculate_gain_reduction(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Calculate the gain reduction for the compressor.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).
            sample_rate (int): The sample rate of the input signal in Hz.

        Returns:
            np.ndarray: The gain reduction values to be applied to the signal.
        """
        self._validate_input_signal(signal, sample_rate)
        max_amplitude = self._compute_max_amplitude(signal)
        max_amplitude = np.maximum(max_amplitude, 1e-10)  # Ensure max_amplitude is never zero
        amplitude_dB = 20 * np.log10(max_amplitude)  # Avoid log(0) since max_amplitude is >= 1e-10

        compression_factor = self._compute_compression_factor(amplitude_dB)
        desired_gain_reduction = np.where(
            amplitude_dB > self.threshold,
            self.threshold_linear * (max_amplitude / self.threshold_linear) ** (1 / compression_factor),
            max_amplitude
        )

        target_gain_reduction = np.where(max_amplitude > 1e-10, desired_gain_reduction / max_amplitude, 1.0)
        self._gain_reduction = smooth_gain_reduction(target_gain_reduction, self.attack_coeff, self.release_coeff,
                                                     last_gain_reduction=self._last_gain_reduction_loaded)

        return self._gain_reduction
