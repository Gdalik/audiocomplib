# Requires pedalboard library. It can be installed with: pip install pedalboard
import sys
from audiocomplib import AudioCompressor
from pedalboard.io import AudioStream, AudioFile, StreamResampler
from pathlib import Path

# Initialize compressor globally
Comp = AudioCompressor(threshold=0, ratio=4, attack_time_ms=2, release_time_ms=100, knee_width=5, realtime=True)


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

            while f.tell() < f.frames:
                chunk = f.read(buffer_size)

                Comp.set_threshold(round(Comp.threshold - 0.01, 2))  # Lower threshold
                Comp.set_makeup_gain(round(Comp.makeup_gain + 0.002, 3))  # Add make-up gain

                # Show threshold and make-up gain values
                sys.stdout.write(f'\rThreshold: {Comp.threshold} dB | Make-Up Gain: +{Comp.makeup_gain} dB')

                chunk_comp = Comp.process(chunk, samplerate)  # Apply compression effect

                if stream.sample_rate != samplerate:  # Resample audio if audio device samplerate is different
                    Resample = StreamResampler(samplerate, stream.sample_rate, num_channels)
                    chunk_comp = Resample.process(chunk_comp)
                    samplerate = stream.sample_rate

                # Decode and play 512 samples at a time:
                stream.write(chunk_comp, samplerate)

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