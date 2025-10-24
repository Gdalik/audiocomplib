"""
Base class for audio dynamics processing (v0.2.0).

Supports compression-depth-based variable release (switchable on/off).
Simple, psychoacoustically proven: deeper compression = slower release.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from .smooth_gain_reduction_init import smooth_gain_reduction


class AudioDynamics(ABC):
    """
    Base class for audio dynamics processing with optional depth-based variable release.

    DEPTH-BASED ADAPTIVE RELEASE (Psychoacoustically Proven):
        - Measures compression depth: how much gain reduction is happening
        - Deeper compression → slower release (prevents pumping artifacts)
        - Light compression → normal release (stays transparent)
        - At full compression: release time = base × 2.0 (proven industry standard)

    Example:
        compressor = AudioCompressor(threshold=-10.0, ratio=4.0, variable_release=True)
        output = compressor.process(audio, sample_rate=48000)

    Subclasses must implement: target_gain_reduction(signal) -> np.ndarray
    """

    def __init__(
        self,
        threshold: float,
        attack_time_ms: float,
        release_time_ms: float,
        realtime: bool = False,
        variable_release: bool = True,
        max_release_multiplier: float = 2.0
    ):
        """
        Initialize audio dynamics processor.

        Args:
            threshold (float): Threshold in dBFS where processing starts.
            attack_time_ms (float): Attack time in milliseconds.
                Minimum enforced: 0.01 ms. Shorter = faster transient catch.
            release_time_ms (float): Base release time in milliseconds.
                Minimum enforced: 1 ms.
                - If variable_release=True: scales with compression depth
                - If variable_release=False: applied uniformly
            realtime (bool): Enable real-time (chunked) processing mode.
                Carries over state between process() calls. Default: False.
            variable_release (bool): Enable depth-based variable release.
                True: deeper compression = slower release (recommended). Default: True.
                False: fixed release time.
            max_release_multiplier (float): Max release multiplier at full compression.
                Typical: 2.0 (proven, prevents pumping). Default: 2.0.
                Range: 1.0-5.0.
                - 1.0: fixed release (no adaptation)
                - 2.0: up to 2x release at full compression
                - 3.0: up to 3x release (more smooth but sluggish)
        """
        self.threshold = threshold
        self.attack_time_ms = max(0.01, attack_time_ms)
        self.release_time_ms = max(1.0, release_time_ms)
        self.variable_release = variable_release
        self.max_release_multiplier = np.clip(float(max_release_multiplier), 1.0, 5.0)
        self._realtime = realtime

        self._gain_reduction: Optional[np.ndarray] = None
        self._last_gain_reduction_loaded: Optional[float] = None
        self._sample_rate = 44100

    def reset(self) -> None:
        """Reset all internal state. Call when starting a new audio stream."""
        self._last_gain_reduction_loaded = None
        self._gain_reduction = None

    def set_threshold(self, threshold: float) -> None:
        """Set threshold level in dBFS."""
        self.threshold = threshold

    def set_attack_time(self, attack_time_ms: float) -> None:
        """Set attack time in milliseconds (minimum: 0.01 ms)."""
        self.attack_time_ms = max(0.01, attack_time_ms)

    def set_release_time(self, release_time_ms: float) -> None:
        """Set base release time in milliseconds (minimum: 1 ms)."""
        self.release_time_ms = max(1.0, release_time_ms)

    def set_realtime(self, realtime: bool) -> None:
        """Enable/disable real-time (chunked) processing mode."""
        self._realtime = realtime

    def set_variable_release(self, variable_release: bool) -> None:
        """Enable/disable depth-based variable release."""
        self.variable_release = variable_release

    def set_max_release_multiplier(self, multiplier: float) -> None:
        """
        Set maximum release multiplier at full compression (1.0-5.0).

        Example: 2.0 = at full compression, release time = base × 2.0
        """
        self.max_release_multiplier = np.clip(float(multiplier), 1.0, 5.0)

    def process(self, input_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio signal through dynamics processor.

        Main entry point. Automatically handles gain reduction calculation,
        variable release calculation, and exponential smoothing.

        Args:
            input_signal (np.ndarray): Audio signal, shape (channels, samples).
                Must be float32 or float64.
            sample_rate (int): Sample rate in Hz.

        Returns:
            np.ndarray: Processed audio signal, same shape and dtype as input.

        Raises:
            ValueError: If input signal format or sample rate is invalid.
        """
        self._validate_input_signal(input_signal, sample_rate)

        last_gr = self.last_gain_reduction if self._realtime else None
        self._load_last_gain_reduction(last_gr)

        try:
            self._calculate_gain_reduction(input_signal)
        except (IndexError, ValueError):
            self.reset()
            return input_signal

        output_signal = input_signal * self._gain_reduction

        if output_signal.dtype != input_signal.dtype:
            output_signal = output_signal.astype(dtype=input_signal.dtype)

        return output_signal

    def get_gain_reduction(self) -> Optional[np.ndarray]:
        """
        Get current gain reduction envelope in dB.

        Returns the smoothed gain reduction from the most recent process() call.
        Useful for visualization, metering, or debugging.

        Returns:
            np.ndarray or None: Gain reduction in dB (negative = reduction),
                or None if no processing has been done yet.
        """
        if self._gain_reduction is None:
            return None
        return 20 * np.log10(np.clip(self._gain_reduction, 1e-10, 1.0))

    def _load_last_gain_reduction(self, value: Optional[float]) -> None:
        """Load previous gain value for real-time envelope continuity."""
        self._last_gain_reduction_loaded = value

    @property
    def threshold_linear(self) -> float:
        """Convert threshold (dBFS) to linear amplitude (0-1)."""
        return 10 ** (self.threshold / 20)

    @property
    def last_gain_reduction(self) -> Optional[float]:
        """Get last smoothed gain value for real-time state carryover."""
        return self._gain_reduction[-1] if self._gain_reduction is not None else None

    def _validate_input_signal(self, signal: np.ndarray, sample_rate: int) -> None:
        """Validate signal format and update sample rate."""
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        self._sample_rate = sample_rate

        if signal.dtype not in (np.float32, np.float64):
            raise ValueError(f"Signal must be float32/float64, not {signal.dtype}!")

        if signal.ndim != 2:
            raise ValueError(f"Signal must be 2D (channels, samples), got {signal.ndim}D!")

    def _compute_max_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Compute max amplitude across channels (stereo-linking)."""
        return np.max(np.abs(signal), axis=0)

    def _apply_soft_knee_compression(
        self,
        amplitude_dB: np.ndarray,
        threshold: float,
        knee_width: float,
        ratio: float
    ) -> np.ndarray:
        """Apply soft-knee compression in dB domain."""
        knee_start = threshold - knee_width / 2
        knee_end = threshold + knee_width / 2

        x = amplitude_dB
        T = threshold
        W = knee_width
        R = ratio

        return np.where(
            x < knee_start,
            x,
            np.where(
                x <= knee_end,
                x + ((1 / R - 1) * (x - T + W / 2) ** 2) / (2 * W),
                T + (x - T) / R
            )
        )

    def _calculate_variable_release_times(self, target_gain_reduction: np.ndarray) -> np.ndarray:
        """
        Calculate per-sample release times based on compression depth.

        ALGORITHM:
            1. Compression depth = 1 - target_gain_reduction
               - 0 = no compression (gain = 1.0, full volume)
               - 1 = full compression (gain = 0.0, silence)
            2. Release time = base × (1 + depth × (multiplier - 1))
               - No compression: base release
               - Full compression: base × multiplier (default 2.0x)

        Args:
            target_gain_reduction: Unsmoothed gain (0-1), shape (n_samples,).

        Returns:
            Release times in milliseconds for each sample.
        """
        # Compression depth: 0 = no compression, 1 = full compression
        compression_depth = 1.0 - target_gain_reduction

        # Release time scales with depth
        # From base × 1.0 (no compression) to base × multiplier (full compression)
        release_times = self.release_time_ms * (1.0 + compression_depth * (self.max_release_multiplier - 1.0))

        return release_times

    @abstractmethod
    def target_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous (unsmoothed) gain reduction.

        Subclasses implement their specific dynamics algorithm here.
        The base class applies attack/release smoothing afterwards.

        Args:
            signal (np.ndarray): Input audio, shape (channels, samples).

        Returns:
            np.ndarray: Linear gain reduction (0-1), shape (samples,).
                1.0 = no reduction (unity)
                0.5 = -6 dB reduction
                0.0 = complete silence
        """
        pass

    def _calculate_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate smoothed gain reduction with optional variable release.

        Workflow:
            1. Get target (unsmoothed) gain from subclass
            2. If variable_release: calculate per-sample release times based on depth
               Else: use fixed release time
            3. Apply Cython smoothing with those release times
        """
        target_gain_reduction = self.target_gain_reduction(signal).astype(np.float64)

        if self.variable_release:
            # Calculate variable release times based on compression depth
            release_times = self._calculate_variable_release_times(target_gain_reduction)
        else:
            # Fixed release: same for all samples
            release_times = np.full_like(target_gain_reduction, self.release_time_ms)

        # Fast Cython smoothing with per-sample release times
        self._gain_reduction = smooth_gain_reduction(
            target_gain_reduction,
            self.attack_time_ms,
            release_times,
            self._sample_rate,
            last_gain_reduction=(
                self._last_gain_reduction_loaded if self._last_gain_reduction_loaded is not None else 1.0
            ),
        )

        return self._gain_reduction
