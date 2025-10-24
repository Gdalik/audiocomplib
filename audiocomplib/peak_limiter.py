"""
Peak limiter with optional depth-based variable release (v0.2.0).

Ensures output never exceeds threshold ceiling while maintaining musicality.
Uses soft-knee transitions for smooth compression around threshold.
"""

import numpy as np
from .audio_dynamics import AudioDynamics


class PeakLimiter(AudioDynamics):
    """
    Peak limiter with optional depth-based variable release.

    Ensures output never exceeds threshold ceiling while maintaining musicality.
    Uses soft-knee transitions for smooth compression around threshold.
    Brickwall clipping as safety stage.

    ALGORITHM:
        - Infinite compression ratio (1e6) for true limiting above threshold
        - Soft-knee for smooth transition into limiting
        - Optional depth-based variable release (deeper limiting = slower release)
        - Brickwall clipping as safety stage to prevent any overshoot

    Example:
        limiter = PeakLimiter(threshold=-1.0, release_time_ms=1.0, variable_release=True)
        output = limiter.process(audio, sample_rate=48000)

    Defaults optimized for peak limiting:
        - Attack: 0.01 ms (ultra-fast catch)
        - Release: 1.0 ms (fast baseline)
        - Variable Release: True (recommended)
    """

    def __init__(
        self,
        threshold: float = -1.0,
        attack_time_ms: float = 0.01,
        release_time_ms: float = 1.0,
        knee_width: float = 2.0,
        realtime: bool = False,
        variable_release: bool = True,
        max_release_multiplier: float = 2.0
    ):
        """
        Initialize peak limiter.

        Args:
            threshold (float): Limiter ceiling in dBFS. Default: -1.0.
            attack_time_ms (float): Attack time (ms). Default: 0.01 (ultra-fast).
            release_time_ms (float): Base release time (ms). Default: 1.0 (fast for limiter).
            knee_width (float): Soft-knee width (dB). Default: 2.0 (0 = hard limiting).
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
        self.knee_width = knee_width
        self._total_clipped_samples = 0

    def set_knee_width(self, knee_width: float) -> None:
        """Set soft-knee width in dB (0 = hard limiting)."""
        self.knee_width = knee_width

    def reset(self) -> None:
        """Reset limiter state."""
        super().reset()
        self._total_clipped_samples = 0

    def target_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate limiting gain curve (pre-smoothing).

        Uses infinite ratio (1e6) for true limiting behavior.
        Soft-knee provides smooth transition into limiting.

        Args:
            signal (np.ndarray): Input, shape (channels, samples).

        Returns:
            Linear gain (0-1), shape (samples,).
        """
        max_amplitude = self._compute_max_amplitude(signal)
        max_amplitude = np.maximum(max_amplitude, 1e-10)
        amplitude_dB = 20 * np.log10(max_amplitude)

        if self.knee_width == 0:
            # Hard limiting: direct ceiling
            output_dB = np.where(amplitude_dB > self.threshold, self.threshold, amplitude_dB)
        else:
            # Soft-knee limiting: smooth transition
            output_dB = self._apply_soft_knee_compression(
                amplitude_dB, self.threshold, self.knee_width, ratio=1e6
            )

        output_linear = 10 ** (output_dB / 20)
        return np.clip(output_linear / max_amplitude, 0.0, 1.0)

    def process(self, input_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process signal through limiter with brickwall clipping safety.

        Applies variable release limiting, then hard clips as final safety stage.

        Args:
            input_signal (np.ndarray): Audio, shape (channels, samples).
            sample_rate (int): Sample rate in Hz.

        Returns:
            Limited audio, guaranteed not to exceed threshold.
        """
        result = super().process(input_signal, sample_rate)
        clip_level = self.threshold_linear

        # Brickwall clipping safety stage
        clipped_mask = (result > clip_level) | (result < -clip_level)
        n_clipped = np.count_nonzero(clipped_mask)
        if n_clipped > 0:
            self._total_clipped_samples += int(n_clipped)
            result = np.clip(result, -clip_level, clip_level)

        return result
