from abc import ABC, abstractmethod
import numpy as np
from .smooth_gain_reduction_init import smooth_gain_reduction
from typing import Optional


class AudioDynamics(ABC):
    """Base class for audio dynamics processing (e.g., compressors, limiters)."""

    def __init__(self, threshold: float, attack_time_ms: float, release_time_ms: float, realtime=False):
        """
        Initialize the audio dynamics processor.

        Args:
            threshold (float): The threshold level in dB. Signals above this level will be processed.
            attack_time_ms (float): The attack time in milliseconds. Determines how quickly the processor reacts to signals above the threshold.
            release_time_ms (float): The release time in milliseconds. Determines how quickly the processor stops processing after the signal falls below the threshold.
            realtime (bool): True if the effect is used for real-time processing (in chunks). Defaults to False.
        """
        self.threshold = threshold
        self.attack_time_ms = attack_time_ms
        self.release_time_ms = release_time_ms
        self._gain_reduction: Optional[np.ndarray] = None
        self._last_gain_reduction_loaded = None
        self._realtime = realtime

    def reset(self):
        """
        Reset the internal state of the dynamics processor.

        This clears any stored gain reduction values, which is particularly important:
        - When processing a new audio stream
        - When reusing the processor for different audio segments
        - To ensure clean state between independent processing calls
        """
        self._last_gain_reduction_loaded = None
        self._gain_reduction = None

    def set_threshold(self, threshold: float) -> None:
        """
        Set the threshold level in dB.

        Args:
            threshold (float): The new threshold level in dB.
        """
        self.threshold = threshold

    def set_attack_time(self, attack_time_ms: float) -> None:
        """
        Set the attack time in milliseconds.

        Args:
            attack_time_ms (float): The new attack time in milliseconds.
        """
        self.attack_time_ms = attack_time_ms

    def set_release_time(self, release_time_ms: float) -> None:
        """
        Set the release time in milliseconds.

        Args:
            release_time_ms (float): The new release time in milliseconds.
        """
        self.release_time_ms = release_time_ms

    def set_realtime(self, arg: bool):
        """
        Enable/disable realtime processing mode.

        Args:
            arg (bool): True (real-time mode enabled), False (real-time mode disabled)
        """
        self._realtime = arg

    def process(self, input_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process the input signal using the dynamics processor.

        Args:
            input_signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).
            sample_rate (int): The sample rate of the input signal in Hz.

        Returns:
            np.ndarray: The processed signal with the same shape as the input signal.
        """
        self._validate_input_signal(input_signal, sample_rate)
        last_gr = self.last_gain_reduction if self._realtime else None
        self._load_last_gain_reduction(last_gr)
        try:
            self._calculate_gain_reduction(input_signal)
        except IndexError:
            self.reset()
            return input_signal
        output_signal = input_signal * self._gain_reduction

        # Ensure that the data type of the output array is the same as the data type of the input array
        if output_signal.dtype != input_signal.dtype:
            output_signal = output_signal.astype(dtype=input_signal.dtype)
        return output_signal

    def get_gain_reduction(self) -> np.ndarray or None:
        """
        Get the gain reduction applied to the signal in dB.

        Returns:
            np.ndarray or None: The gain reduction values in dB if it has already been calculated.
        """
        if self._gain_reduction is None:
            return None
        return 20 * np.log10(self._gain_reduction)

    def _load_last_gain_reduction(self, value: np.float64) -> None:
        """
        In real-time processing, load the last gain reduction value from the previous chunk
        """
        self._last_gain_reduction_loaded = value

    @property
    def threshold_linear(self) -> float:
        """
        Convert the threshold from dB to linear scale.

        Returns:
            float: The threshold in linear scale.
        """
        return 10 ** (self.threshold / 20)

    @property
    def attack_coeff(self) -> float:
        """
        Compute the attack coefficient based on the attack time and sample rate.

        Returns:
            float: The attack coefficient.
        """
        return np.exp(-1 / max(1, int(self.attack_time_ms * self._sample_rate / 1000)))

    @property
    def release_coeff(self) -> float:
        """
        Compute the release coefficient based on the release time and sample rate.

        Returns:
            float: The release coefficient.
        """
        return np.exp(-1 / max(1, int(self.release_time_ms * self._sample_rate / 1000)))

    @property
    def last_gain_reduction(self) -> float or None:
        """
        Return the last value from the internal gain reduction array (if any).

        Returns:
            float or None: The last linear gain reduction value.
        """
        return self._gain_reduction[-1] if self._gain_reduction is not None else None

    def _validate_input_signal(self, signal: np.ndarray, sample_rate: int) -> None:
        """
        Validate the input signal and sample rate.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).
            sample_rate (int): The sample rate of the input signal in Hz.

        Raises:
            ValueError: If the input signal is not a 2D array or the sample rate is invalid.
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive value.")
        else:
            self._sample_rate = sample_rate
        if signal.dtype not in (np.float32, np.float64):
            raise ValueError(f'The data type of an input signal must be float32 or float64, not {signal.dtype}!')

    def _compute_max_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the maximum amplitude of the signal across channels.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).

        Returns:
            np.ndarray: The maximum amplitude for each sample across channels.
        """
        if signal.ndim != 2:
            raise ValueError("Input signal must be a 2D array with shape (channels, samples).")
        return np.max(np.abs(signal), axis=0)

    @abstractmethod
    def target_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the target gain reduction before attack/release smoothing.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).

        Returns:
            np.ndarray: The linear gain reduction values between 0 and 1.
        """
        pass

    def _calculate_gain_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the gain reduction to be applied to the signal.

        Args:
            signal (np.ndarray): The input signal as a 2D array with shape (channels, samples).

        Returns:
            np.ndarray: The gain reduction values to be applied to the signal.
        """
        gain_reduction = self.target_gain_reduction(signal)
        self._gain_reduction = smooth_gain_reduction(gain_reduction.astype(np.float64), self.attack_coeff,
                                                     self.release_coeff,
                                                     last_gain_reduction=self._last_gain_reduction_loaded)
        return self._gain_reduction