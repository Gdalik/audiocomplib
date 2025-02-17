# Requires pedalboard library: pip install pedalboard
from pedalboard.io import AudioStream, AudioFile
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


def play_audio(filename: str, output: str):
    with AudioFile(filename) as f:
        samplerate = f.samplerate
        num_channels = f.num_channels
        with AudioStream(output_device_name=output, sample_rate=samplerate, num_output_channels=num_channels) as stream:
            buffer_size = 512
            last_gr = None      # Last gain reduction value of the previous chunk (if exists)

            while f.tell() < f.frames:
                chunk = f.read(buffer_size)

                #Comp.set_threshold(round(Comp.threshold - 0.01, 2))   # Lower threshold in real-time
                print(f'Threshold: {Comp.threshold}')
                if Comp.threshold <= -60:   # Stop playback when threshold reaches -60 dB
                    break

                chunk_comp = Comp.process(chunk, samplerate, last_gain_reduction=last_gr)   # Apply compression effect

                # Get and store the last gain reduction value of current chunk for the next iteration
                last_gr = Comp.last_gain_reduction

                # Decode and play 512 samples at a time:
                stream.write(chunk_comp, samplerate)


if __name__ == '__main__':
    Comp = AudioCompressor(threshold=-40, ratio=4, attack_time_ms=5, release_time_ms=200, knee_width=5)
    outputs = valid_audio_outputs()
    while True:
        filename = input('Enter absolute path to audio file:')
        if Path(filename).exists() and Path(filename).is_file():
            break
    if not outputs:
        print('No valid audio outputs!')
    else:
        play_audio(filename, outputs[0])
