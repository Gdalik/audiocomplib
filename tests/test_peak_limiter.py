"""
Comprehensive unit tests for PeakLimiter (v0.2.0).

Tests cover:
- Parameter setting and validation
- Basic limiting functionality
- Variable release behavior
- Edge cases and error handling
- Real-time processing mode
"""

import unittest
import numpy as np
from audiocomplib import PeakLimiter
import time


class TestPeakLimiterBasics(unittest.TestCase):
    """Test basic peak limiter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.limiter = PeakLimiter(
            threshold=-1.0,
            attack_time_ms=0.1,
            release_time_ms=1.0,
            knee_width=2.0
        )
        self.signal = np.array([[0.5, 0.8, 1.2, 1.0, 0.3, 0.4, 1.5, 2.0, 0.7, 1.1]], dtype=np.float32)
        self.sample_rate = 44100

    def test_set_threshold(self):
        """Test threshold setter."""
        self.limiter.set_threshold(-0.5)
        self.assertEqual(self.limiter.threshold, -0.5)

    def test_set_attack_time(self):
        """Test attack time setter."""
        self.limiter.set_attack_time(0.2)
        self.assertEqual(self.limiter.attack_time_ms, 0.2)

    def test_set_release_time(self):
        """Test release time setter."""
        self.limiter.set_release_time(2.0)
        self.assertEqual(self.limiter.release_time_ms, 2.0)

    def test_set_knee_width(self):
        """Test soft-knee width setter."""
        self.limiter.set_knee_width(5.0)
        self.assertEqual(self.limiter.knee_width, 5.0)

    def test_peak_limiting(self):
        """Test that limiter reduces signal above threshold."""
        compressed = self.limiter.process(self.signal, self.sample_rate)
        # Signal should be attenuated where it exceeds threshold
        self.assertTrue(np.all(compressed <= self.signal))

    def test_hard_knee_limiting(self):
        """Test hard-knee (non-soft) limiting."""
        limiter = PeakLimiter(threshold=-1.0, knee_width=0.0)
        compressed = limiter.process(self.signal, self.sample_rate)
        # Hard knee should produce steeper curve
        self.assertTrue(np.all(compressed <= self.signal))

    def test_threshold_protection(self):
        """Test that output never exceeds threshold."""
        compressed = self.limiter.process(self.signal, self.sample_rate)
        threshold_linear = 10 ** (self.limiter.threshold / 20)
        self.assertTrue(np.all(np.abs(compressed) <= threshold_linear + 1e-6))


class TestPeakLimiterVariableRelease(unittest.TestCase):
    """Test variable release functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.limiter = PeakLimiter(
            threshold=-1.0,
            attack_time_ms=0.1,
            release_time_ms=10.0,
            variable_release=True,
            max_release_multiplier=2.0
        )
        self.sample_rate = 44100

    def test_variable_release_enabled(self):
        """Test that variable release can be enabled."""
        self.assertTrue(self.limiter.variable_release)

    def test_variable_release_disabled(self):
        """Test that variable release can be disabled."""
        self.limiter.set_variable_release(False)
        self.assertFalse(self.limiter.variable_release)

    def test_release_time_multiplier(self):
        """Test that multiplier is set correctly."""
        self.assertEqual(self.limiter.max_release_multiplier, 2.0)
        self.limiter.set_max_release_multiplier(3.0)
        self.assertEqual(self.limiter.max_release_multiplier, 3.0)

    def test_release_time_multiplier_clipping(self):
        """Test that multiplier is clipped to valid range."""
        self.limiter.set_max_release_multiplier(10.0)  # Should clip to 5.0
        self.assertEqual(self.limiter.max_release_multiplier, 5.0)
        self.limiter.set_max_release_multiplier(0.5)  # Should clip to 1.0
        self.assertEqual(self.limiter.max_release_multiplier, 1.0)


    def test_variable_vs_fixed_release_consistency(self):
        """Test that variable and fixed modes are consistent (both work)."""
        signal = np.array([[0.0, 1.5, 1.5, 1.5, 0.5, 0.0]], dtype=np.float32)

        # Both should work without errors
        self.limiter.set_variable_release(True)
        variable_out = self.limiter.process(signal, self.sample_rate)

        self.limiter.reset()
        self.limiter.set_variable_release(False)
        fixed_out = self.limiter.process(signal, self.sample_rate)

        # Both should limit the peak
        self.assertTrue(np.all(variable_out <= signal))
        self.assertTrue(np.all(fixed_out <= signal))


class TestPeakLimiterEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.limiter = PeakLimiter(threshold=-1.0)
        self.sample_rate = 44100

    def test_silent_signal(self):
        """Test with silent (zero) input."""
        silent = np.zeros((1, 10), dtype=np.float32)
        output = self.limiter.process(silent, self.sample_rate)
        self.assertTrue(np.allclose(output, silent))

    def test_extreme_levels(self):
        """Test with very loud signals."""
        extreme = np.array([[100.0, -100.0, 50.0]], dtype=np.float32)
        output = self.limiter.process(extreme, self.sample_rate)
        threshold_linear = 10 ** (self.limiter.threshold / 20)
        self.assertTrue(np.all(np.abs(output) <= threshold_linear + 1e-5))

    def test_multi_channel(self):
        """Test stereo processing."""
        stereo = np.array([[0.5, 1.5], [0.8, 1.2]], dtype=np.float32)
        output = self.limiter.process(stereo, self.sample_rate)
        self.assertEqual(output.shape, stereo.shape)
        self.assertTrue(np.all(output <= stereo))

    def test_float64_support(self):
        """Test float64 signal processing."""
        signal = np.array([[0.5, 1.5, 0.8]], dtype=np.float64)
        output = self.limiter.process(signal, self.sample_rate)
        self.assertEqual(output.dtype, np.float64)

    def test_invalid_dtype(self):
        """Test error handling for invalid dtype."""
        signal = np.array([[1, 2, 3]], dtype=np.int32)
        with self.assertRaises(ValueError):
            self.limiter.process(signal, self.sample_rate)

    def test_invalid_sample_rate(self):
        """Test error handling for invalid sample rate."""
        signal = np.array([[0.5]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.limiter.process(signal, 0)

    def test_invalid_signal_shape(self):
        """Test error handling for 1D signal."""
        signal = np.array([0.5, 1.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.limiter.process(signal, self.sample_rate)


class TestPeakLimiterRealtimeMode(unittest.TestCase):
    """Test real-time chunked processing."""

    def test_realtime_mode_enabled(self):
        """Test real-time mode can be enabled."""
        limiter = PeakLimiter(realtime=True)
        self.assertTrue(limiter._realtime)

    def test_realtime_chunked_processing(self):
        """Test processing in chunks maintains continuity."""
        limiter = PeakLimiter(
            threshold=-1.0,
            release_time_ms=10.0,
            realtime=True
        )

        # Create signal
        full_signal = np.array([[0.0, 1.5, 1.5, 1.5, 1.0, 0.5, 0.0, 0.5]], dtype=np.float32)
        sample_rate = 44100

        # Process as chunks
        chunk1 = full_signal[:, :4]
        chunk2 = full_signal[:, 4:]

        out1 = limiter.process(chunk1, sample_rate)
        out2 = limiter.process(chunk2, sample_rate)

        # Concatenate and process full signal
        limiter.reset()
        full_out = limiter.process(full_signal, sample_rate)

        # Chunks should maintain envelope continuity
        combined = np.concatenate([out1, out2], axis=1)
        self.assertEqual(combined.shape, full_out.shape)

    def test_reset_clears_state(self):
        """Test that reset clears processing state."""
        limiter = PeakLimiter(realtime=True)
        signal = np.array([[1.5]], dtype=np.float32)

        limiter.process(signal, 44100)
        self.assertIsNotNone(limiter.last_gain_reduction)

        limiter.reset()
        self.assertIsNone(limiter.last_gain_reduction)


class TestPeakLimiterGainReduction(unittest.TestCase):
    """Test gain reduction calculation and retrieval."""

    def test_get_gain_reduction_db(self):
        """Test retrieving gain reduction in dB."""
        limiter = PeakLimiter(threshold=-1.0)
        signal = np.array([[0.5, 1.5, 0.8]], dtype=np.float32)

        limiter.process(signal, 44100)
        gain_reduction_db = limiter.get_gain_reduction()

        self.assertIsNotNone(gain_reduction_db)
        self.assertEqual(gain_reduction_db.shape[0], signal.shape[1])
        self.assertTrue(np.all(gain_reduction_db <= 0))  # All reductions should be <= 0dB

    def test_get_gain_reduction_before_process(self):
        """Test that get_gain_reduction returns None before processing."""
        limiter = PeakLimiter()
        self.assertIsNone(limiter.get_gain_reduction())

    def test_last_gain_reduction(self):
        """Test retrieving last gain reduction value."""
        limiter = PeakLimiter(realtime=True)
        signal = np.array([[0.5, 1.5]], dtype=np.float32)

        limiter.process(signal, 44100)
        last_gr = limiter.last_gain_reduction

        self.assertIsNotNone(last_gr)
        self.assertLessEqual(last_gr, 1.0)
        self.assertGreaterEqual(last_gr, 0.0)


class TestPeakLimiterModes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        signal = np.random.randn(1_000_000) * 0.7
        signal[100000:100020] *= 5
        signal[700000:700010] *= 8
        self.signal = signal[np.newaxis, :]  # Now shape = (1, samples)
        self.threshold = 0.9
        self.attack = 0.002
        self.release = 0.05

    def test_variable_and_fixed_modes_give_different_results(self):
        # Fixed mode
        limiter_fixed = PeakLimiter(
            threshold=self.threshold,
            attack_time_ms=self.attack,
            release_time_ms=self.release,
            variable_release=False
        )
        output_fixed = limiter_fixed.process(self.signal, sample_rate=44100)

        # Variable mode
        limiter_variable = PeakLimiter(
            threshold=self.threshold,
            attack_time_ms=self.attack,
            release_time_ms=self.release,
            variable_release=True
        )
        output_variable = limiter_variable.process(self.signal, sample_rate=44100)

        # Assert outputs differ (should for big data with peaks)
        self.assertFalse(
            np.allclose(output_fixed, output_variable),
            "Fixed and variable modes of PeakLimiter produce identical results on complex input!"
        )


class TestPeakLimiterPerformance(unittest.TestCase):
    def test_performance(self):
        # Generate test signal: 1 second at 48kHz, stereo
        sr = 48000
        duration_sec = 1.0
        n_samples = int(sr * duration_sec)
        audio = np.random.randn(2, n_samples).astype(np.float32) * 0.1

        limiter = PeakLimiter(threshold=-1.0, variable_release=True)

        t0 = time.time()
        output = limiter.process(audio, sr)
        t1 = time.time()

        elapsed = t1 - t0
        real_time_ratio = elapsed / duration_sec

        # Basic output checks
        self.assertEqual(output.shape, audio.shape)
        self.assertEqual(output.dtype, audio.dtype)

        self.assertLess(real_time_ratio, 0.1, msg="PeakLimiter should be >10x real-time")


if __name__ == "__main__":
    unittest.main()
