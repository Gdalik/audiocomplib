# Audiocomplib (v0.2.0)

**Copyright (c) 2025, Gdaliy Garmiza**

![Example of Audiocomplib Compressor and Limiter Transfer Curves](https://github.com/Gdalik/audiocomplib/blob/main/examples/Images/TransferCurve.png)

This Python package provides two essential audio processing tools: **Audio Compressor** and **Peak Limiter**. These classes are designed for use in audio applications, scripts and libraries, and are implemented in Python with high performance in mind, including optional Cython-based optimizations.

The library supports real-time mode, maintaining smooth transitions between audio chunks. Now with depth-based variable release for natural, transparent dynamics control with minimal artifacts.

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
- [Architecture](#architecture)
- [Usage](#usage)
    - [Array Format](#array-format)
    - [Audio Compressor Example](#audio-compressor-example)
    - [Peak Limiter Example](#peak-limiter-example)
    - [Variable Release](#variable-release)
    - [Public Methods](#public-methods)
        - [AudioDynamics Methods](#audiodynamics-methods)
        - [AudioCompressor Methods](#audiocompressor-methods)
        - [PeakLimiter Methods](#peaklimiter-methods)
    - [Enabling Real-Time Mode](#enabling-real-time-mode)
    - [Real-Time Processing Example](#real-time-processing-example)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)


## Features

- **Audio Compressor**: Dynamic range compression with flexible control over threshold, ratio, attack, release, soft-knee, and makeup gain.
- **Peak Limiter**: Transparent peak limiting with optional soft-knee for smooth limiting without artifacts.
- **Depth-Based Variable Release**: Psychoacoustically proven release behavior that scales with compression depth to prevent pumping artifacts.
- **Real-Time Mode**: Seamless streaming support with state carryover between audio chunks.
- **Cython Acceleration**: High-performance exponential smoothing with optional Python fallback.
- **Stereo \& Multi-Channel**: Built-in stereo-linking (maximum amplitude across channels).
- **Soft-Knee Compression**: Smooth quadratic transition around threshold for musical control.


## Requirements

- Python 3.9+
- NumPy
- Cython (optional, for performance)


## Quick Start

```python
import numpy as np
from audiocomplib import AudioCompressor, PeakLimiter

# Create test signal (2 channels, 44.1kHz, 1 second)
audio = np.random.randn(2, 44100).astype(np.float32)

# Compress with variable release
compressor = AudioCompressor(
    threshold=-10.0,
    ratio=4.0,
    attack_time_ms=1.0,
    release_time_ms=100.0
)
compressed = compressor.process(audio, sample_rate=44100)

# Limit with variable release
limiter = PeakLimiter(
    threshold=-1.0,
    attack_time_ms=0.1,
    release_time_ms=1.0
)
limited = limiter.process(audio, sample_rate=44100)
```


## Installation

### Option 1: Install from PyPI

```bash
pip install audiocomplib
```


### Option 2: Install from GitHub

```bash
pip install git+https://github.com/Gdalik/audiocomplib.git
```


### Option 3: Clone and Install Locally

```bash
git clone https://github.com/Gdalik/audiocomplib.git
cd audiocomplib
pip install .
```


## Performance Optimization

The `smooth_gain_reduction` function is implemented in **Cython** for high performance:

- **With Cython**: ~5ms for 1M samples (real-time safe ✓)
- **Pure Python fallback**: ~100ms for 1M samples (auto-enabled if Cython unavailable)

The package automatically detects and uses the optimal implementation. If Cython fails to compile, a warning is raised but the library continues to work with the Python fallback.

To manually compile Cython:

```bash
pip install -e . --force-reinstall --no-cache-dir
```


## Building from Source with Manual Cython Compilation

If you encounter issues with automatic Cython compilation or want to ensure the Cython-optimized version is used:

1. Clone the repository:

```bash
git clone https://github.com/Gdalik/audiocomplib.git
cd audiocomplib
```

2. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

3. Manually compile the Cython extension:

```bash
python setup.py build_ext --inplace
```

4. Build the package:

```bash
pip install .
```


## Architecture

### Class Hierarchy

```
AudioDynamics (Base Class)
├── AudioCompressor (Ratio-based compression)
└── PeakLimiter (Infinite ratio limiting)
```


### Processing Pipeline

```
Input Signal
    ↓
[1. Target Gain Calculation] (Subclass implementation)
    ↓ (unsmoothed gain curve)
[2. Variable Release Calculation] (Depth-based)
    ↓ (per-sample release times)
[3. Exponential Smoothing] (Cython/Python)
    ↓ (attack/release envelope)
Output Signal
```


### Key Components

- **`audio_dynamics.py`**: Base class with parameter management and gain calculation
- **`smooth_gain_reduction.pyx`**: Cython-accelerated envelope smoothing
- **`smooth_gain_reduction_py.py`**: Pure Python fallback
- **`audio_compressor.py`**: Ratio-based compressor implementation
- **`peak_limiter.py`**: Infinite ratio limiter implementation


## Usage

### Array Format

Both `AudioCompressor` and `PeakLimiter` accept NumPy arrays with shape **`(channels, samples)`**. This format is compatible with **[Pedalboard by Spotify](https://github.com/spotify/pedalboard)**.

If your audio library uses `(samples, channels)` format, transpose before processing:

```python
# If array is (samples, channels), transpose it
input_signal = input_signal.T
```


### Audio Compressor Example

```python
import numpy as np
from audiocomplib import AudioCompressor

# Generate sample signal (2 channels, 44100 samples)
input_signal = np.random.randn(2, 44100).astype(np.float32)

# Initialize compressor
compressor = AudioCompressor(
    threshold=-10.0,
    ratio=4.0,
    attack_time_ms=1.0,
    release_time_ms=100.0,
    knee_width=3.0,
    makeup_gain=6.0,
    variable_release=True
)

# Process signal
compressed_signal = compressor.process(input_signal, sample_rate=44100)

# Get gain reduction in dB
gain_reduction_db = compressor.get_gain_reduction()

# Adjust parameters
compressor.set_ratio(6.0)
compressor.set_makeup_gain(8.0)
```


### Peak Limiter Example

```python
import numpy as np
from audiocomplib import PeakLimiter

# Generate sample signal (2 channels, 44100 samples)
input_signal = np.random.randn(2, 44100).astype(np.float32)

# Initialize peak limiter
limiter = PeakLimiter(
    threshold=-1.0,
    attack_time_ms=0.01,
    release_time_ms=1.0,
    knee_width=2.0,
    variable_release=True
)

# Process signal
limited_signal = limiter.process(input_signal, sample_rate=44100)

# Get gain reduction in dB
gain_reduction_db = limiter.get_gain_reduction()
```


### Variable Release

Variable release scales release time based on compression depth:

```python
compressor = AudioCompressor(
    threshold=-10.0,
    ratio=4.0,
    release_time_ms=100.0,
    variable_release=True,
    max_release_multiplier=2.0
)

# Release times scale with compression depth:
# - No compression (depth=0):    100ms
# - 50% compression (depth=0.5): 150ms
# - Full compression (depth=1):  200ms

# Adjust multiplier for different material:
compressor.set_max_release_multiplier(3.0)

# Disable for fixed release:
compressor.set_variable_release(False)
```


### Public Methods

Both `AudioCompressor` and `PeakLimiter` inherit from `AudioDynamics`.

#### AudioDynamics Methods:

- `process(input_signal: np.ndarray, sample_rate: int)`: Process audio signal
- `set_threshold(threshold: float)`: Set threshold in dBFS
- `set_attack_time(attack_time_ms: float)`: Set attack time in milliseconds
- `set_release_time(release_time_ms: float)`: Set base release time in milliseconds
- `set_variable_release(variable_release: bool)`: Enable/disable variable release
- `set_max_release_multiplier(multiplier: float)`: Set max release multiplier (1.0-5.0)
- `set_realtime(realtime: bool)`: Enable/disable real-time mode
- `get_gain_reduction()`: Get smoothed gain reduction in dB
- `reset()`: Reset internal state


#### AudioCompressor Methods:

- `set_ratio(ratio: float)`: Set compression ratio
- `set_knee_width(knee_width: float)`: Set soft-knee width in dB
- `set_makeup_gain(makeup_gain: float)`: Set makeup gain in dB


#### PeakLimiter Methods:

- `set_knee_width(knee_width: float)`: Set soft-knee width in dB


### Enabling Real-Time Mode

For chunked audio processing, enable real-time mode to maintain envelope continuity:

```python
# Initialize with realtime=True
compressor = AudioCompressor(realtime=True)

# Or enable later
compressor.set_realtime(True)
```

In real-time mode, the processor stores the last gain reduction value and uses it at the beginning of the next chunk, ensuring smooth transitions without artifacts.

### Real-Time Processing Example

```python
from pedalboard.io import AudioStream, AudioFile
from audiocomplib import AudioCompressor

# Initialize compressor in real-time mode
comp = AudioCompressor(
    threshold=0,
    ratio=4,
    attack_time_ms=2,
    release_time_ms=100,
    knee_width=5,
    realtime=True
)

with AudioFile('audio.wav') as f:
    samplerate = f.samplerate
    num_channels = f.num_channels
    
    with AudioStream(output_device_name=AudioStream.default_output_device_name,
                     sample_rate=samplerate, num_output_channels=num_channels) as stream:
        buffer_size = 512
        
        while f.tell() < f.frames:
            chunk = f.read(buffer_size)
            
            # Automate parameters in real-time
            comp.set_threshold(round(comp.threshold - 0.01, 2))
            comp.set_makeup_gain(round(comp.makeup_gain + 0.002, 3))
            
            # Apply compression
            chunk_comp = comp.process(chunk, samplerate)
            stream.write(chunk_comp, samplerate)
            
            if comp.threshold <= -60:
                break
```

Install Pedalboard:

```bash
pip install pedalboard
```

For a more complete example, see [realtime_processing_pedalboard.py](https://github.com/Gdalik/audiocomplib/blob/main/examples/realtime_processing_pedalboard.py).

## Testing

Run comprehensive unit tests:

```bash
pytest tests/ -v
pytest tests/test_peak_limiter.py -v
pytest tests/test_audio_compressor.py -v
```


## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

Copyright (c) 2025 Gdaliy Garmiza

Licensed under the MIT License (see LICENSE file)
