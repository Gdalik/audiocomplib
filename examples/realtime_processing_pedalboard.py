# Requires pedalboard library. It can be installed with: pip install pedalboard
import sys
from pedalboard.io import AudioStream, AudioFile, StreamResampler
from audiocomplib import AudioCompressor
from pathlib import Path


def valid_audio_outputs() -> list:
    """Check audio outputs and make a list of valid ones"""
    valid_outputs = []
    for SD in AudioStream.output_device_names:
        try:
            with AudioStream(output_device_name=SD) as s:
                pass
            valid_outputs.append(SD)
        except:
            continue
    return valid_outputs


def play_audio(filename: str, output_device_name=AudioStream.default_output_device_name):
    """Playback the audio file through the output device"""
    with AudioFile(filename) as f:
        samplerate = f.samplerate
        num_channels = f.num_channels
        with AudioStream(output_device_name=output_device_name, sample_rate=samplerate, num_output_channels=num_channels) as stream:
            buffer_size = 512

            while f.tell() < f.frames:
                chunk = f.read(buffer_size)

                Comp.set_threshold(round(Comp.threshold - 0.01, 2))   # Lower threshold in real-time
                sys.stdout.write('\rThreshold: {}'.format(Comp.threshold))  # Show threshold value

                if Comp.threshold <= -60:   # Stop playback when threshold reaches -60 dB
                    break

                chunk_comp = Comp.process(chunk, samplerate)   # Apply compression effect

                if stream.sample_rate != samplerate:    # Resample audio if audio device samplerate is different
                    Resample = StreamResampler(samplerate, stream.sample_rate, num_channels)
                    chunk_comp = Resample.process(chunk_comp)
                    samplerate = stream.sample_rate

                # Decode and play 512 samples at a time:
                stream.write(chunk_comp, samplerate)


if __name__ == '__main__':
    # Initialize compressor
    Comp = AudioCompressor(threshold=0, ratio=4, attack_time_ms=2, release_time_ms=100, knee_width=5, realtime=True)

    # Get list of valid audio outputs
    outputs = valid_audio_outputs()

    # Choose the audio file
    while True:
        filename = input('Enter path to audio file:')
        if Path(filename).exists() and Path(filename).is_file():
            break
    if not outputs:
        print('No valid audio outputs!')
        sys.exit()

    # Choose the playback device
    device_num = 0
    device_name = AudioStream.default_output_device_name
    for ind, out in enumerate(outputs):
        default_lab = '\t[Default]' if out == AudioStream.default_output_device_name else ''
        print(f'{ind + 1}. {out}{default_lab}')
    while True:
        device_num = input('Enter playback device number or press Enter to use the default output:')
        if not device_num:
            break
        try:
            device_num = int(device_num)
        except ValueError:
            continue
        if 0 < device_num <= len(outputs):
            device_name = outputs[device_num - 1]
            break

    # Process and stream audio in realtime, automating the compressor threshold parameter change
    play_audio(filename, device_name)
