"""
Audio compressor with optional depth-based variable release (v0.2.0).

Reduces dynamic range by applying a compression ratio above threshold.
Combines soft-knee for musicality with optional depth-based variable release.
"""

import numpy as np
from .audio_dynamics import AudioDynamics


class AudioCompressor(AudioDynamics):
    """
    Audio compressor with optional depth-based variable release.

    Reduces dynamic range by applying a compression ratio to signals exceeding
    the threshold. Features soft-knee for smooth, musical compression and optional
    depth-based variable release (deeper compression = slower release).

    COMPRESSION ALGORITHM:
        - Below threshold: no processing (unity gain)
        - Above threshold: gain reduction = (input - threshold) / ratio
        - Soft-knee: smooth quadratic transition (optional)
        - Variable release: scales with compression depth (optional)
        - Makeup gain: post-compression level compensation

    Example:
        compressor = AudioCompressor(threshold=-10.0, ratio=4.0, variable_release=True)
        output = compressor.process(audio, sample_rate=48000)
    """

    def __init__(
            self,
            threshold: float = -10.0,
            ratio: float = 4.0,
            attack_time_ms: float = 1.0,
            release_time_ms: float = 100.0,
            knee_width: float = 3.0,
            makeup_gain: float = 0.0,
            realtime: bool = False,
            variable_release: bool = True,
            max_release_multiplier: float = 2.0
    ):
        """
        Initialize audio compressor.

        Args:
            threshold (float): Compression threshold in dBFS. Default: -10.0.
            ratio (float): Compression ratio (e.g., 4.0 = 4:1). Default: 4.0.
            attack_time_ms (float): Attack time in milliseconds. Default: 1.0.
            release_time_ms (float): Base release time in milliseconds. Default: 100.0.
                With variable_release=True: scales with compression depth.
            knee_width (float): Soft-knee width in dB. Default: 3.0 (0 = hard knee).
            makeup_gain (float): Makeup gain in dB. Default: 0.0.
            realtime (bool): Real-time mode. Default: False.
            variable_release (bool): Enable depth-based variable release. Default: True.
            max_release_multiplier (float): Max release multiplier. Default: 2.0.
        """
        super().__init__(
            threshold,
            attack_time_ms,
            release_time_ms,
            realtime=realtime,
            variable_release=variable_release,
            max_release_multiplier=max_release_multiplier
        )
        self.ratio = ratio
        self.knee_width = knee_width
        self.makeup_gain = makeup_gain

    def set_ratio(self, ratio: float) -> None:
        """Set compression ratio (e.g., 4.0 for 4:1)."""
        self.ratio = ratio

    def set_knee_width(self, knee_width: float) -> None:
        """Set soft-knee width in dB (0 = hard knee)."""
        self.knee_width = knee_width

    def set_makeup_gain(self, makeup_gain: float) -> None:
        """Set makeup gain in dB (applied after compression)."""
        self.makeup_gain = makeup_gain

    def target_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate instantaneous compression gain curve (pre-smoothing).

        Args:
            signal (np.ndarray): Input, shape (channels, samples).

        Returns:
            Linear gain (0-1), shape (samples,).
        """
        max_amplitude = self._compute_max_amplitude(signal)
        max_amplitude = np.maximum(max_amplitude, 1e-10)
        amplitude_dB = 20 * np.log10(max_amplitude)

        if self.knee_width == 0:
            # Hard knee compression
            output_dB = np.where(
                amplitude_dB > self.threshold,
                self.threshold + (amplitude_dB - self.threshold) / self.ratio,
                amplitude_dB
            )
        else:
            # Soft-knee compression
            output_dB = self._apply_soft_knee_compression(
                amplitude_dB, self.threshold, self.knee_width, self.ratio
            )

        output_linear = 10 ** (output_dB / 20)
        return np.clip(output_linear / max_amplitude, 0.0, 1.0)

    def process(self, input_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through compressor with makeup gain.

        Args:
            input_signal (np.ndarray): Audio, shape (channels, samples).
            sample_rate (int): Sample rate in Hz.

        Returns:
            Compressed audio with makeup gain applied.
        """
        result = super().process(input_signal, sample_rate)

        # Apply makeup gain
        gain_linear = 10 ** (self.makeup_gain / 20)
        return result * gain_linear
