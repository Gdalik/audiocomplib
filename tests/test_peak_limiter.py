import unittest
import numpy as np
from audiocomplib import PeakLimiter


class TestPeakLimiter(unittest.TestCase):

    def setUp(self):
        """Set up the peak limiter instance and signal before each test."""
        self.limiter = PeakLimiter(threshold=-1.0, attack_time_ms=0.1, release_time_ms=1.0)
        # Simulate a test signal with 1 channel and 10 samples
        self.signal = np.array([[0.5, 0.8, 1.2, 1.0, 0.3, 0.4, 1.5, 2.0, 0.7, 1.1]])
        self.sample_rate = 44100

    def test_set_threshold(self):
        """Test setting the threshold of the peak limiter."""
        self.limiter.set_threshold(-0.5)
        self.assertEqual(self.limiter.threshold, -0.5)

    def test_set_attack_time(self):
        """Test setting the attack time of the peak limiter."""
        self.limiter.set_attack_time(0.2)
        self.assertEqual(self.limiter.attack_time_ms, 0.2)

    def test_set_release_time(self):
        """Test setting the release time of the peak limiter."""
        self.limiter.set_release_time(2.0)
        self.assertEqual(self.limiter.release_time_ms, 2.0)

    def test_peak_limiting(self):
        """Test if the peak limiter applies limiting correctly."""
        compressed_signal = self.limiter.process(self.signal, self.sample_rate)
        # We expect the signal to be reduced if it exceeds the threshold
        self.assertTrue(np.all(compressed_signal <= self.signal))  # Should be attenuated

    def test_edge_case_zero_signal(self):
        """Test edge case when the input signal is silent."""
        silent_signal = np.zeros_like(self.signal)
        compressed_signal = self.limiter.process(silent_signal, self.sample_rate)
        self.assertTrue(np.allclose(compressed_signal, silent_signal))

    def test_no_compression_when_below_threshold(self):
        """Test that no compression happens when the signal is below the threshold."""
        below_threshold_signal = np.array([[0.1, 0.2, 0.3, 0.1, 0.05, 0.05, 0.08, 0.2, 0.15, 0.05]])
        compressed_signal = self.limiter.process(below_threshold_signal, self.sample_rate)
        self.assertTrue(np.allclose(below_threshold_signal, compressed_signal))  # No change

if __name__ == "__main__":
    unittest.main()
