"""
Comprehensive unit tests for AudioCompressor (v0.2.0).

Tests cover:
- Parameter setting and validation
- Compression functionality with various ratios
- Soft-knee behavior
- Variable release
- Makeup gain
- Edge cases and error handling
- Real-time processing
"""

import time
import unittest
import numpy as np
from audiocomplib import AudioCompressor


class TestAudioCompressorBasics(unittest.TestCase):
    """Test basic compressor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.compressor = AudioCompressor(
            threshold=-10.0,
            ratio=4.0,
            attack_time_ms=1.0,
            release_time_ms=100.0,
            knee_width=3.0
        )
        self.signal = np.clip(np.random.randn(2, 1000), -1, 1).astype(np.float32)
        self.sample_rate = 44100

    def test_set_threshold(self):
        """Test threshold setter."""
        self.compressor.set_threshold(-5.0)
        self.assertEqual(self.compressor.threshold, -5.0)

    def test_set_ratio(self):
        """Test compression ratio setter."""
        self.compressor.set_ratio(6.0)
        self.assertEqual(self.compressor.ratio, 6.0)

    def test_set_knee_width(self):
        """Test soft-knee width setter."""
        self.compressor.set_knee_width(6.0)
        self.assertEqual(self.compressor.knee_width, 6.0)

    def test_set_makeup_gain(self):
        """Test makeup gain setter."""
        self.compressor.set_makeup_gain(6.0)
        self.assertEqual(self.compressor.makeup_gain, 6.0)

    def test_compression_applied(self):
        """Test that compression reduces signal above threshold."""
        compressed = self.compressor.process(self.signal, self.sample_rate)
        # RMS should be reduced
        rms_in = np.sqrt(np.mean(self.signal ** 2))
        rms_out = np.sqrt(np.mean(compressed ** 2))
        self.assertLess(rms_out, rms_in)

    def test_compression_ratio_2_to_1(self):
        """Test 2:1 compression ratio."""
        self.compressor.set_ratio(2.0)
        compressed = self.compressor.process(self.signal, self.sample_rate)
        self.assertTrue(np.all(np.isfinite(compressed)))

    def test_compression_ratio_8_to_1(self):
        """Test 8:1 compression ratio."""
        self.compressor.set_ratio(8.0)
        compressed = self.compressor.process(self.signal, self.sample_rate)
        self.assertTrue(np.all(np.isfinite(compressed)))


class TestAudioCompressorKnee(unittest.TestCase):
    """Test soft-knee functionality."""

    def test_hard_knee(self):
        """Test hard-knee compression (no softening)."""
        compressor = AudioCompressor(threshold=-10.0, ratio=4.0, knee_width=0.0)
        signal = np.array([[-5.0, -10.0, -15.0]], dtype=np.float32)
        output = compressor.process(signal, 44100)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_soft_knee(self):
        """Test soft-knee compression."""
        compressor = AudioCompressor(threshold=-10.0, ratio=4.0, knee_width=3.0)
        signal = np.array([[-5.0, -10.0, -15.0]], dtype=np.float32)
        output = compressor.process(signal, 44100)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_knee_width_transition(self):
        """Test smooth transition with knee."""
        compressor = AudioCompressor(threshold=-10.0, ratio=4.0, knee_width=6.0)
        # Signal near threshold should show smooth compression
        signal = np.linspace(-15.0, -5.0, 100).reshape(1, -1).astype(np.float32)
        output = compressor.process(signal, 44100)
        # Output should be smooth (no discontinuities)
        self.assertTrue(np.all(np.isfinite(output)))


class TestAudioCompressorMakeupGain(unittest.TestCase):
    """Test makeup gain compensation."""

    def test_makeup_gain_applied(self):
        """Test that makeup gain is applied correctly."""
        compressor = AudioCompressor(
            threshold=-20.0,
            ratio=4.0,
            makeup_gain=6.0
        )
        signal = np.array([[0.1, 0.2, 0.15]], dtype=np.float32)
        output = compressor.process(signal, 44100)

        # With makeup gain, output should be boosted
        expected_boost = 10 ** (6.0 / 20)
        self.assertGreater(np.mean(output), np.mean(signal) * expected_boost * 0.9)

    def test_zero_makeup_gain(self):
        """Test with zero makeup gain."""
        compressor = AudioCompressor(makeup_gain=0.0)
        signal = np.array([[0.5]], dtype=np.float32)
        output = compressor.process(signal, 44100)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_negative_makeup_gain(self):
        """Test with negative makeup gain (reduction)."""
        compressor = AudioCompressor(makeup_gain=-6.0)
        signal = np.array([[0.5]], dtype=np.float32)
        output = compressor.process(signal, 44100)
        self.assertLess(np.abs(output[0]), np.abs(signal[0]))


class TestAudioCompressorVariableRelease(unittest.TestCase):
    """Test variable release functionality."""

    def test_variable_release_togglable(self):
        """Test that variable release can be enabled/disabled."""
        compressor = AudioCompressor(variable_release=True)
        self.assertTrue(compressor.variable_release)

        compressor.set_variable_release(False)
        self.assertFalse(compressor.variable_release)

    def test_multiplier_range(self):
        """Test release multiplier is within valid range."""
        compressor = AudioCompressor(max_release_multiplier=2.0)
        self.assertEqual(compressor.max_release_multiplier, 2.0)

        compressor.set_max_release_multiplier(10.0)  # Should clip
        self.assertEqual(compressor.max_release_multiplier, 5.0)


class TestAudioCompressorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.compressor = AudioCompressor()
        self.sample_rate = 44100

    def test_silent_signal(self):
        """Test with silent input."""
        silent = np.zeros((1, 100), dtype=np.float32)
        output = self.compressor.process(silent, self.sample_rate)
        self.assertTrue(np.allclose(output, silent, atol=1e-6))

    def test_very_loud_signal(self):
        """Test with extremely loud signal."""
        loud = np.array([[100.0, 50.0, -100.0]], dtype=np.float32)
        output = self.compressor.process(loud, self.sample_rate)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_multi_channel_stereo(self):
        """Test stereo processing."""
        stereo = np.random.randn(2, 1000).astype(np.float32)
        output = self.compressor.process(stereo, self.sample_rate)
        self.assertEqual(output.shape, stereo.shape)

    def test_multi_channel_5_1(self):
        """Test 5.1 surround processing."""
        surround = np.random.randn(6, 1000).astype(np.float32)
        output = self.compressor.process(surround, self.sample_rate)
        self.assertEqual(output.shape, surround.shape)

    def test_float64_support(self):
        """Test float64 signal."""
        signal = np.array([[0.5, 1.5]], dtype=np.float64)
        output = self.compressor.process(signal, self.sample_rate)
        self.assertEqual(output.dtype, np.float64)

    def test_invalid_dtype(self):
        """Test error on invalid dtype."""
        signal = np.array([[1, 2]], dtype=np.int16)
        with self.assertRaises(ValueError):
            self.compressor.process(signal, self.sample_rate)

    def test_invalid_shape(self):
        """Test error on 1D signal."""
        signal = np.array([0.5, 1.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.compressor.process(signal, self.sample_rate)

    def test_invalid_sample_rate(self):
        """Test error on invalid sample rate."""
        signal = np.array([[0.5]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.compressor.process(signal, -44100)


class TestAudioCompressorGainReduction(unittest.TestCase):
    """Test gain reduction calculations."""

    def test_gain_reduction_calculation(self):
        """Test gain reduction is calculated correctly."""
        compressor = AudioCompressor(threshold=-20.0, ratio=4.0)
        signal = np.array([[0.1, 0.5, 0.1]], dtype=np.float32)

        compressor.process(signal, 44100)
        gr_db = compressor.get_gain_reduction()

        self.assertIsNotNone(gr_db)
        self.assertEqual(gr_db.shape[0], 3)
        self.assertTrue(np.all(gr_db <= 0))  # Gain reduction should be <= 0dB


class TestAudioCompressorRealtimeMode(unittest.TestCase):
    """Test real-time chunked processing."""

    def test_realtime_chunked_processing(self):
        """Test processing maintains continuity across chunks."""
        compressor = AudioCompressor(realtime=True)

        # Create signal
        signal = np.concatenate([
            np.ones((1, 100)) * 0.1,
            np.ones((1, 100)) * 1.5,
            np.ones((1, 100)) * 0.1
        ], axis=1).astype(np.float32)

        # Process in chunks
        chunk_size = 100
        output_chunks = []
        for i in range(0, signal.shape[1], chunk_size):
            chunk = signal[:, i:i + chunk_size]
            output_chunks.append(compressor.process(chunk, 44100))

        output = np.concatenate(output_chunks, axis=1)
        self.assertEqual(output.shape, signal.shape)

    def test_realtime_reset(self):
        """Test state reset in real-time mode."""
        compressor = AudioCompressor(realtime=True)
        signal = np.array([[1.5]], dtype=np.float32)

        compressor.process(signal, 44100)
        self.assertIsNotNone(compressor.last_gain_reduction)

        compressor.reset()
        self.assertIsNone(compressor.last_gain_reduction)


class TestAudioCompressorPerformance(unittest.TestCase):
    def test_performance(self):
        # Generate test signal: 1 second at 48kHz, stereo
        sr = 48000
        duration_sec = 1.0
        n_samples = int(sr * duration_sec)
        audio = np.random.randn(2, n_samples).astype(np.float32) * 0.1

        limiter = AudioCompressor(threshold=-1.0, variable_release=True)

        t0 = time.time()
        output = limiter.process(audio, sr)
        t1 = time.time()

        elapsed = t1 - t0
        real_time_ratio = elapsed / duration_sec

        # Basic output checks
        self.assertEqual(output.shape, audio.shape)
        self.assertEqual(output.dtype, audio.dtype)

        self.assertLess(real_time_ratio, 0.1, msg="AudioCompressor should be >10x real-time")


if __name__ == "__main__":
    unittest.main()
