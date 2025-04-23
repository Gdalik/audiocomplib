# Audiocomplib

**Copyright (c) 2025, Gdaliy Garmiza**

![Example of Audiocomplib Compressor and Limiter Transfer Curves](https://github.com/Gdalik/audiocomplib/blob/main/examples/Images/TransferCurve.png)

This Python package provides two essential audio processing tools: **Audio Compressor** and **Peak Limiter**. These classes are designed for use in audio applications, scripts and libraries, and are implemented in Python with high performance in mind, including optional Cython-based optimizations.

The library supports real-time mode, maintaining smooth transitions between audio chunks.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Option 1: Install from PyPI](#option-1-install-from-pypi)
  - [Option 2: Install from GitHub](#option-2-install-from-github)
  - [Option 3: Clone and Install Locally](#option-3-clone-and-install-locally)
- [Performance Optimization](#performance-optimization)
- [Building from Source with Manual Cython Compilation](#building-from-source-with-manual-cython-compilation)
- [Usage](#usage)
  - [Audio Compressor Example](#audio-compressor-example)
  - [Peak Limiter Example](#peak-limiter-example)
  - [Public Methods](#public-methods)
    - [AudioDynamics Methods](#audiodynamics-methods)
    - [AudioCompressor Methods](#audiocompressor-methods)
    - [PeakLimiter Methods](#peaklimiter-methods)
  - [Enabling Real-Time Mode](#enabling-real-time-mode)
  - [Real-Time Processing Example](#real-time-processing-example)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Audio Compressor**: Applies dynamic range compression to audio signals, with flexible control over threshold, ratio, attack, release, knee width and make-up gain.
- **Peak Limiter**: Applies peak limiting to audio signals, aiming to prevent the signal from exceeding a specified threshold while preserving dynamics as much as possible. Adjustable attack and release times.

## Requirements

- Python 3.9+
- NumPy
- Cython (optional, for performance optimization)

## Quick Start

To quickly test the library, simply [install it](#installation) and proceed with running the [examples](#audio-compressor-example).

If you're eager to try the real-time processing feature, check out [this script](https://github.com/Gdalik/audiocomplib/blob/main/examples/realtime_processing_pedalboard.py) and experiment with your own compression parameters.

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

## Performance Optimization

For improved performance, the `smooth_gain_reduction` function is implemented in **Cython**. This function is used internally by both the **Audio Compressor** and **Peak Limiter** to apply attack and release smoothing to the gain reduction.

The package will automatically use the Cython-optimized version of the `smooth_gain_reduction` function if Cython is installed and the module is successfully compiled. If the Cython module is not available (e.g., Cython is not installed or compilation fails), the package will fall back to a pure Python implementation and raise a warning message. This fallback is handled internally, so users do not need to make any changes to their code.

## Building from Source with Manual Cython Compilation
If you encounter issues with the automatic Cython compilation or want to ensure the Cython-optimized version is used, you can clone the repository (see [Installation.Option 3, step 1](#option-3-clone-and-install-locally)), manually compile the Cython extension and then build `audiocomplib` from the source. After cloning the repository, follow these steps:

1. Navigate to the root `audiocomplib` directory (if you are not already there).

2. Ensure all the dependencies are installed by running:
   ```bash
   pip install -r requirements.txt

3. Manually compile the Cython extension:
   ```bash
   python setup.py build_ext --inplace
   ```

This will build the Cython module and enable the optimized version of the `smooth_gain_reduction` function.

4. Finally, build the package:
   ```bash
   pip install .
   ```

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

# Retrieve the gain reduction in dB
gain_reduction_db = compressor.get_gain_reduction()
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

# Retrieve the gain reduction in dB
gain_reduction_db = limiter.get_gain_reduction()
```

### Public Methods

Both `AudioCompressor` and `PeakLimiter` classes inherit from `AudioDynamics`, sharing common methods:

#### AudioDynamics Methods:
- `set_threshold(threshold: float)`: Sets the threshold level in dB.
- `set_attack_time(attack_time_ms: float)`: Sets the attack time in milliseconds.
- `set_release_time(release_time_ms: float)`: Sets the release time in milliseconds.
- `get_gain_reduction()`: Returns the gain reduction curve as a numpy array of values in dB.
- `target_gain_reduction(signal: np.ndarray)`: Returns the target (non-smoothed) gain reduction curve of given signal as a numpy array of linear values between 0 and 1. Useful for visualizations (see [example of computing and plotting transfer curves](https://github.com/Gdalik/audiocomplib/blob/main/examples/get_plot_transfer_curve.py)).
- `set_realtime(realtime: bool)`: Enables or disables real-time processing mode.
- `process(input_signal: np.ndarray, sample_rate: int)`: Process the input signal using the dynamics processor.
- `reset()`: Reset the internal state of the dynamics processor.

#### AudioCompressor Methods:
- `set_ratio(ratio: float)`: Sets the compression ratio.
- `set_knee_width(knee_width: float)`: Sets the knee width.
- `set_makeup_gain(makeup_gain: float)`: Sets the make-up gain in dB.

### Enabling real-time mode

When processing audio in chunks, the `AudioCompressor` or `PeakLimiter` class object should be initialized with the `realtime` option set to True in order to activate the real-time mode. Alternatively, you can use the `set_realtime` method to enable real-time mode after initialization.

In real-time mode, the effect stores its last gain reduction value and uses it when applying the attack/release smoothing of the gain reduction curve at the beginning of the next processed chunk. This ensures smooth transitions between audio chunks and maintains the integrity of the dynamic range processing without producing artifacts at chunk edges.

### Real-Time Processing Example

This example demonstrates real-time audio processing and playback using the `audiocomplib` and `pedalboard` libraries. It showcases how to automate the threshold and the make-up gain parameters of an audio compressor in real-time, gradually reducing the threshold during playback and increasing output gain to compensate the attenuation.

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
            Comp.set_threshold(round(Comp.threshold - 0.01, 2))   # Lower threshold
            Comp.set_makeup_gain(round(Comp.makeup_gain + 0.002, 3))  # Add make-up gain
            chunk_comp = Comp.process(chunk, samplerate)   # Apply compression effect

            # Decode and play 512 samples at a time:
            stream.write(chunk_comp, samplerate)

            if Comp.threshold <= -60:  # Stop playback when threshold reaches -60 dB
                break
```

The extended version of this example is available [here](https://github.com/Gdalik/audiocomplib/blob/main/examples/realtime_processing_pedalboard.py). It is more stable (handling and preventing possible exceptions) and illustrative.

Before running the example, ensure you have [Pedalboard](https://github.com/spotify/pedalboard) Python library installed:

```bash
pip install pedalboard
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Gdalik/audiocomplib). If you'd like to contribute code, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Gdalik/audiocomplib/blob/main/LICENSE) file for details.
