import unittest
import numpy as np
from audiocomplib.audio_compressor import AudioCompressor


class TestAudioCompressor(unittest.TestCase):

    def setUp(self):
        """Set up the compressor instance and signal before each test."""
        self.compressor = AudioCompressor(threshold=-10.0, ratio=4.0, attack_time_ms=1.0, release_time_ms=100.0)
        # Simulate a test signal with 1 channel and 10 samples
        self.signal = np.array([[0.5, 0.8, 1.2, 1.0, 0.3, 0.4, 1.5, 2.0, 0.7, 1.1]])
        self.sample_rate = 44100

    def test_set_threshold(self):
        """Test setting the threshold of the compressor."""
        self.compressor.set_threshold(-5.0)
        self.assertEqual(self.compressor.threshold, -5.0)

    def test_set_ratio(self):
        """Test setting the ratio of the compressor."""
        self.compressor.set_ratio(6.0)
        self.assertEqual(self.compressor.ratio, 6.0)

    def test_compression(self):
        """Test if the compressor applies compression correctly."""
        compressed_signal = self.compressor.process(self.signal, self.sample_rate)
        self.assertTrue(np.allclose(compressed_signal, self.signal * self.compressor._gain_reduction))

    def test_gain_reduction(self):
        """Test if the compressor calculates gain reduction correctly."""
        self.compressor.process(self.signal, self.sample_rate)
        gain_reduction_dbfs = self.compressor.get_gain_reduction()
        self.assertEqual(gain_reduction_dbfs.shape[0], self.signal.shape[1])

    def test_edge_case_zero_signal(self):
        """Test edge case when the input signal is silent."""
        silent_signal = np.zeros_like(self.signal)
        compressed_signal = self.compressor.process(silent_signal, self.sample_rate)
        self.assertTrue(np.allclose(compressed_signal, silent_signal))

    def test_compressor_with_soft_knee(self):
        """Test the compressor with soft knee applied."""
        self.compressor.set_knee_width(5.0)
        compressed_signal = self.compressor.process(self.signal, self.sample_rate)
        self.assertTrue(np.all(compressed_signal <= self.signal))  # Should be attenuated

if __name__ == "__main__":
    unittest.main()
