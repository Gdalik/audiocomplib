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

    def target_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the target gain reduction before attack/release smoothing for compressor.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).

        Returns:
            np.ndarray: The linear gain reduction values between 0 and 1.
        """
        # Compute the maximum amplitude of the signal
        max_amplitude = self._compute_max_amplitude(signal)
        max_amplitude = np.maximum(max_amplitude, 1e-10)  # Ensure max_amplitude is never zero

        # Convert amplitude to dB
        amplitude_dB = 20 * np.log10(max_amplitude)  # Avoid log(0) since max_amplitude is >= 1e-10

        # Conditions for soft knee region
        knee_start = self.threshold - self.knee_width / 2
        knee_end = self.threshold + self.knee_width / 2
        in_soft_knee = (amplitude_dB > knee_start) & (amplitude_dB < knee_end)

        # Compute desired gain reduction
        if self.knee_width == 0:
            # Hard knee: no smooth transition, just apply ratio above threshold
            desired_gain_reduction = np.where(
                amplitude_dB > self.threshold,
                (max_amplitude / self.threshold_linear) ** (1 / self.ratio - 1),
                1.0
            )
        else:
            # Soft knee: apply second-order interpolation within the knee region
            # Quadratic interpolation: y = a * x^2 + b * x + c
            # Boundary conditions:
            # 1. At x = knee_start, y = 1 (no compression)
            # 2. At x = knee_end, y = (10^(knee_end/20) / threshold_linear)^(1/ratio - 1) (hard knee gain reduction)
            # 3. Derivative at x = knee_start is 0 (smooth transition)
            x1 = knee_start
            x2 = knee_end
            y1 = 1.0
            y2 = (10 ** (knee_end / 20) / self.threshold_linear) ** (1 / self.ratio - 1)

            # Solve for coefficients a, b, c
            a = (y2 - y1) / (x2 - x1) ** 2
            b = -2 * a * x1
            c = y1 - a * x1 ** 2 - b * x1

            # Compute gain reduction within the knee region
            gain_reduction_knee = a * amplitude_dB ** 2 + b * amplitude_dB + c

            # Apply gain reduction
            desired_gain_reduction = np.where(
                in_soft_knee,
                gain_reduction_knee,
                np.where(amplitude_dB >= knee_end, (max_amplitude / self.threshold_linear) ** (1 / self.ratio - 1), 1.0)
            )

            # Ensure gain reduction is within [0, 1]
        gain_reduction = np.clip(desired_gain_reduction, 0.0, 1.0)
        return gain_reduction

    def _calculate_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the gain reduction for the compressor.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).

        Returns:
            np.ndarray: The gain reduction values to be applied to the signal.
        """

        target_gain_reduction = self.target_gain_reduction(signal).astype(dtype=np.float64)
        self._gain_reduction = smooth_gain_reduction(target_gain_reduction, self.attack_coeff, self.release_coeff,
                                                     last_gain_reduction=self._last_gain_reduction_loaded)

        return self._gain_reduction
