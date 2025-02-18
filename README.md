# Audio Compressor and Peak Limiter

This Python package provides two essential audio processing tools: **Audio Compressor** and **Peak Limiter**. These classes are designed for use in audio applications, scripts and libraries, and are implemented in Python with high performance in mind, including optional Cython-based optimizations.

## Features

- **Audio Compressor**: Applies dynamic range compression to audio signals, with flexible control over threshold, ratio, attack, release, and knee width.
- **Peak Limiter**: Applies peak limiting to audio signals, aiming to prevent the signal from exceeding a specified threshold while preserving dynamics as much as possible. Adjustable attack and release times.

## Requirements

- Python 3.9+
- NumPy
- Cython (optional, for performance optimization)

## Installation

### Option 1: Install from PyPI

The easiest way to install **audiocomplib** is from PyPI:

```bash
pip install audiocomplib
```

### Option 2: Install from GitHub

If you want to install the latest development version directly from the GitHub repository, use:

```bash
pip install git+https://github.com/Gdalik/audiocomplib.git
```

### Option 3: Clone and Install Locally

If you prefer to clone the repository and install it locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gdalik/audiocomplib.git
   cd audiocomplib
   ```

2. Install the package and its dependencies:
   ```bash
   pip install .
   ```

### Cython Optimization

If you want to use the Cython-optimized version of the `smooth_gain_reduction` function, ensure that `Cython` is installed. The package will automatically compile the Cython module during installation. If the Cython module is not available, it will fall back to a pure Python implementation.

## Usage

Both the **Audio Compressor** and **Peak Limiter** take **NumPy arrays** as input with the shape `(channels, samples)`. While this may not be the most common format across all libraries, it is a reasonable choice for handling multi-channel audio, especially for compatibility with libraries such as **[Pedalboard by Spotify](https://github.com/spotify/pedalboard)**.

However, some audio libraries use the `(samples, channels)` array shape instead. If you're working with such a library, you'll need to **transpose** the input array before processing with the Audio Compressor or Peak Limiter. You can easily do this with:

```python
# If the array is in (samples, channels) format, transpose it
input_signal = input_signal.T
```

### Audio Compressor Example

```python
import numpy as np
from audiocomplib import AudioCompressor

# Generate a sample audio signal (2 channels, 44100 samples)
input_signal = np.random.randn(2, 44100)

# Initialize compressor
compressor = AudioCompressor(threshold=-10.0, ratio=4.0, attack_time_ms=1.0, release_time_ms=100.0, knee_width=3.0)

# Process the signal with compression
sample_rate = 44100
compressed_signal = compressor.process(input_signal, sample_rate)

# Retrieve the gain reduction in dBFS
gain_reduction_dbfs = compressor.get_gain_reduction()
```

### Peak Limiter Example

```python
import numpy as np
from audiocomplib import PeakLimiter

# Generate a sample audio signal (2 channels, 44100 samples)
input_signal = np.random.randn(2, 44100)

# Initialize peak limiter
limiter = PeakLimiter(threshold=-1.0, attack_time_ms=0.1, release_time_ms=1.0)

# Process the signal with peak limiting
sample_rate = 44100
limited_signal = limiter.process(input_signal, sample_rate)

# Retrieve the gain reduction in dBFS
gain_reduction_dbfs = limiter.get_gain_reduction()
```

### Public Methods

Both `AudioCompressor` and `PeakLimiter` classes inherit from `AudioDynamics`, sharing common methods:

#### AudioDynamics Methods:
- `set_threshold(self, threshold)`: Sets the threshold level in dBFS.
- `set_attack_time(self, attack_time_ms)`: Sets the attack time in milliseconds.
- `set_release_time(self, release_time_ms)`: Sets the release time in milliseconds.
- `get_gain_reduction(self)`: Returns the current gain reduction in dBFS.
- `set_realtime(self, realtime: bool)`: Enables or disables real-time processing mode.

#### AudioCompressor Methods:
- `set_ratio(self, ratio)`: Sets the compression ratio.
- `set_knee_width(self, knee_width)`: Sets the knee width.
- `process(self, input_signal, sample_rate)`: Applies compression to the input signal.

#### PeakLimiter Methods:
- `process(self, input_signal, sample_rate)`: Applies peak limiting to the input signal.

### Enabling real-time mode

When processing audio in chunks, the `AudioCompressor` or `PeakLimiter` class object should be initialized with the `realtime` option set to True in order to activate the real-time mode. Alternatively, you can use the `set_realtime` method to enable real-time mode after initialization.

In real-time mode, the effect stores its last gain reduction value and uses it when applying the attack/release smoothing of the gain reduction curve at the beginning of the next processed chunk. This ensures smooth transitions between audio chunks and maintains the integrity of the dynamic range processing without producing artifacts at chunk edges.

### Real-Time Processing Example

This example demonstrates real-time audio processing and playback using the `audiocomplib` and `pedalboard` libraries. It showcases how to automate the threshold parameter of an audio compressor in real-time, gradually reducing the threshold during playback.

The short version (to get the idea):

```python
from pedalboard.io import AudioStream, AudioFile
from audiocomplib import AudioCompressor

# Initialize compressor
Comp = AudioCompressor(threshold=0, ratio=4, attack_time_ms=2, release_time_ms=100, knee_width=5, realtime=True)

with AudioFile('your_audio_file.wav') as f:     # Replace with path to an audio file (WAV, AIFF, FLAC, MP3 or OGG)
    samplerate = f.samplerate
    num_channels = f.num_channels
    with AudioStream(output_device_name=AudioStream.default_output_device_name, sample_rate=samplerate, 
                     num_output_channels=num_channels) as stream:
        buffer_size = 512

        while f.tell() < f.frames:
            chunk = f.read(buffer_size)
            Comp.set_threshold(round(Comp.threshold - 0.01, 2))   # Lower threshold in real-time
            chunk_comp = Comp.process(chunk, samplerate)   # Apply compression effect

            # Decode and play 512 samples at a time:
            stream.write(chunk_comp, samplerate)
```

The full version of this example is available [here](examples/realtime_processing_pedalboard.py). It is more stable (handling and preventing possible exceptions) and illustrative.

Before running the example, ensure you have [Pedalboard](https://github.com/spotify/pedalboard) Python library installed:

```bash
pip install pedalboard
```

## Performance Optimization

For improved performance, the `smooth_gain_reduction` function is implemented in Cython. This function is used internally by both the **Audio Compressor** and **Peak Limiter** to apply attack and release smoothing to the gain reduction.

The package will automatically use the Cython-optimized version if available. If the Cython module is not compiled or unavailable, it will fall back to a pure Python implementation. This fallback is handled internally, so users do not need to make any changes to their code.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Gdalik/audiocomplib). If you'd like to contribute code, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
