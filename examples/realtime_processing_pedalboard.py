# This module provides functionality to process and play audio files in real-time using a dynamic audio compressor.
# The compressor parameters (threshold, ratio, attack time, release time, etc.) are adjusted during playback.
# The module also allows the user to select an audio output device and validates the audio file path.

# Dependencies:
# - Requires the `pedalboard` library for audio processing and streaming.
#   It can be installed with: `pip install pedalboard`
# - The `audiocomplib` module, which contains the `AudioCompressor` class, must be available in the Python path.

# Usage:
# 1. Run the script.
# 2. Enter the path to an audio file when prompted.
# 3. Select an audio output device from the list of valid devices.
# 4. The audio will be played in real-time with dynamic compression applied.
#    The compressor's parameters are automatically adjusted during playback.

# Notes:
# - The compressor's threshold is gradually lowered during playback, and makeup gain is increased.
# - The compressor's ratio, knee width, attack time and release time are increased.
# - Playback stops when the compressor's threshold reaches -60 dB.
# - The default buffer size for streaming is 512 samples.


import sys
from audiocomplib import AudioCompressor
from pedalboard.io import AudioStream, AudioFile, StreamResampler
from pathlib import Path

# Initialize compressor globally
Comp = AudioCompressor(threshold=0, ratio=4, attack_time_ms=2, release_time_ms=100, knee_width=0, realtime=True)


def valid_audio_outputs() -> list:
    """Check audio outputs and make a list of valid ones"""
    print('Checking audio outputs...')
    valid_outputs = []
    for SD in AudioStream.output_device_names:
        try:
            with AudioStream(output_device_name=SD) as s:
                pass
            valid_outputs.append(SD)
        except:
            continue
    return valid_outputs


def get_audio_file_path() -> str:
    """Prompt the user to enter the path to an audio file and validate it."""
    while True:
        filename = input('Enter path to audio file:')
        if Path(filename).exists() and Path(filename).is_file():
            return filename
        print("Invalid file path. Please try again.")


def select_playback_device(valid_outputs: list) -> str:
    """Prompt the user to select a playback device from the list of valid outputs."""
    if len(valid_outputs) == 1:
        return valid_outputs[0]
    device_name = AudioStream.default_output_device_name
    for ind, out in enumerate(valid_outputs):
        default_lab = '\t[Default]' if out == AudioStream.default_output_device_name else ''
        print(f'{ind + 1}. {out}{default_lab}')

    while True:
        device_num = input(f'Enter playback device number (1-{len(valid_outputs)}) or press Enter to use the default output:')
        if not device_num:
            break
        try:
            device_num = int(device_num)
        except ValueError:
            continue
        if 0 < device_num <= len(valid_outputs):
            device_name = valid_outputs[device_num - 1]
            break
    return device_name


def process_and_play_audio(filename: str, output_device_name: str):
    """Process and play the audio file through the selected output device."""
    with AudioFile(filename) as f:
        samplerate = f.samplerate
        num_channels = f.num_channels

        with AudioStream(output_device_name=output_device_name, sample_rate=samplerate,
                         num_output_channels=num_channels) as stream:
            print('Streaming audio from audiofile, applying AudioCompressor in real time...')
            buffer_size = 512
            device_samplerate = stream.sample_rate
            Resample = StreamResampler(samplerate, device_samplerate, num_channels) if samplerate != device_samplerate else None

            while f.tell() < f.frames:
                chunk = f.read(buffer_size)

                # Adjusting compression parameters in real-time
                Comp.set_threshold(round(Comp.threshold - 0.01, 2))  # Lower threshold
                Comp.set_makeup_gain(round(Comp.makeup_gain + 0.0025, 4))  # Raise make-up gain
                Comp.set_ratio(round(Comp.ratio + 0.001, 3))    # Raise ratio
                Comp.set_attack_time(round(Comp.attack_time_ms + 0.005, 3)) # Raise attack time
                Comp.set_release_time(round(Comp.release_time_ms + 0.2, 1)) # Raise release time
                Comp.set_knee_width(round(Comp.knee_width + 0.001, 3))  # Raise knee width

                # Show automated values
                sys.stdout.write(f'\rThreshold: {Comp.threshold} dB'
                                 f' | Ratio: {Comp.ratio}'
                                 f' | Attack: {Comp.attack_time_ms} ms'
                                 f' | Release: {Comp.release_time_ms} ms'
                                 f' | Knee Width: {Comp.knee_width} dB'
                                 f' | Make-Up Gain: +{Comp.makeup_gain} dB')

                chunk_comp = Comp.process(chunk, samplerate)  # Apply compression effect

                if samplerate != device_samplerate:  # Resample audio if audio device samplerate is different
                    chunk_comp = Resample.process(chunk_comp)

                # Decode and play 512 samples at a time:
                stream.write(chunk_comp, device_samplerate)

                if Comp.threshold <= -60:  # Stop playback when threshold reaches -60 dB
                    break


def main():
    # Get list of valid audio outputs
    outputs = valid_audio_outputs()
    if not outputs:
        print('No valid audio outputs!')
        return

    # Choose the audio file
    filename = get_audio_file_path()

    # Choose the playback device
    device_name = select_playback_device(outputs)

    # Process and stream audio in realtime, automating the compressor threshold parameter change
    process_and_play_audio(filename, device_name)


if __name__ == '__main__':
    main()
