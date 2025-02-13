# Audio Compressor and Peak Limiter

This Python package provides two essential audio processing tools: **Audio Compressor** and **Peak Limiter**. These classes are designed for use in audio production applications and are implemented in Python with high performance in mind, including optional Cython-based optimizations.

## Features

- **Audio Compressor**: Applies dynamic range compression to audio signals, with flexible control over threshold, ratio, attack, release, and knee width.
- **Peak Limiter**: Applies peak limiting to audio signals, aiming to prevent the signal from exceeding a specified threshold while preserving dynamics as much as possible.

Both tools provide high-performance, fast, and high-quality audio processing, with customizable parameter options to tailor the processing for a wide range of specific use cases and requirements.
## Requirements

- Python 3.9+
- NumPy
- Cython

## Installation

## Usage

### Audio Compressor Example

```python
import numpy as np
from audiocomplib.audio_compressor import AudioCompressor

# Generate a sample audio signal (2 channels, 44100 samples)
input_signal = np.random.randn(2, 44100)

# Initialize compressor
compressor = AudioCompressor(threshold=-10.0, ratio=4.0, attack_time_ms=1.0, release_time_ms=100.0)

# Process the signal with compression
sample_rate = 44100
compressed_signal = compressor.process(input_signal, sample_rate)

# Retrieve the gain reduction in dBFS
gain_reduction_dbfs = compressor.get_gain_reduction()
```

### Peak Limiter Example

```python
import numpy as np
from audiocomplib.peak_limiter import PeakLimiter

# Generate a sample audio signal (2 channels, 44100 samples)
input_signal = np.random.randn(2, 44100)

# Initialize peak limiter
limiter = PeakLimiter(threshold=-1.0, attack_time_ms=0.1, release_time_ms=1.0)

# Process the signal with peak limiting
sample_rate = 44100
limited_signal = limiter.process(input_signal, sample_rate)
```