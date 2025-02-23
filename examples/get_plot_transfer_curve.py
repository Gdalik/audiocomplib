# This is an example of computing and drawing AudioCompressor or PeakLimiter transfer curves.
# Requires Matplotlib for plotting. To install it, run:
# pip install matplotlib
from collections.abc import Iterable
import numpy as np
from audiocomplib import AudioCompressor, PeakLimiter
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')


def get_transfer_curve(effect: AudioCompressor or PeakLimiter, num_points=20, min_value=1e-10):
    """
    Compute the transfer curve of an audio effect in the linear domain.

    Args:
        effect: An instance of AudioCompressor or PeakLimiter.
        num_points: Number of points to compute.
        min_value: Minimum input value in the linear domain.

    Returns:
        List of (input, output) tuples in the linear domain.
    """
    input_points = np.logspace(np.log10(min_value), 0, num=num_points)
    input_points_rs = input_points.reshape(1, num_points)
    gain_reduction = effect.target_gain_reduction(input_points_rs)
    output_points = input_points_rs * gain_reduction
    return [(float(In), float(Out)) for In, Out in zip(input_points_rs[0], output_points[0])]


def get_transfer_curve_db(effect: AudioCompressor or PeakLimiter, num_points=20, min_value_db=-96.0):
    """
    Compute the transfer curve of an audio effect in the dB domain.

    Args:
        effect: An instance of AudioCompressor or PeakLimiter.
        num_points: Number of points to compute.
        min_value_db: Minimum input value in dB.

    Returns:
        List of (input, output) tuples in the dB domain.
    """
    min_value_lin = 10 ** (min_value_db / 20)
    transfer_curve = get_transfer_curve(effect, num_points=num_points, min_value=min_value_lin)

    # Convert both input and output to dB
    transfer_curve_db = [(float(20 * np.log10(max(In, 1e-10))), float(20 * np.log10(max(Out, 1e-10))))
                         for In, Out in transfer_curve]
    return transfer_curve_db


def plot_transfer_curve(effects: Iterable[list or tuple], labels, num_points=1000, min_value_db=-96.0):
    """
    Plot the transfer curve of multiple audio effects in the dB scale.

    Args:
        effects: A list of instances of AudioCompressor or PeakLimiter.
        labels: A list of labels for each effect.
        num_points: Number of points to compute.
        min_value_db: Minimum input value in dB.
    """
    # Create plot
    plt.figure(figsize=(5, 5))

    # Plot transfer curve for each effect
    for effect, label in zip(effects, labels):
        # Get transfer curve data in dB
        transfer_curve_db = get_transfer_curve_db(effect, num_points=num_points, min_value_db=min_value_db)

        # Extract input and output values
        input_db, output_db = zip(*transfer_curve_db)

        # Plot the curve
        plt.plot(input_db, output_db, linestyle='-', label=label, linewidth=1.25)

    # Customize plot

    plt.xlabel('Input (dB)')
    plt.ylabel('Output (dB)')
    plt.xlim(-96, 0)
    plt.ylim(-96, 0)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Create an AudioCompressor instance
    Comp = AudioCompressor(threshold=-40, ratio=4, knee_width=8)
    print(f'\nTransfer curve of audio compressor with '
          f'threshold={Comp.threshold}dB, ratio={Comp.ratio}:1, knee width={Comp.knee_width}dB:')
    print(f'\nLinear (Input, Output):\n{get_transfer_curve(Comp, num_points=20)}')
    print(f'\nLogarithmic (Input, Output in dB):\n{get_transfer_curve_db(Comp, num_points=20)}')

    # Create a PeakLimiter instance
    Lim = PeakLimiter(threshold=-20, release_time_ms=1)
    print(f'\nTransfer curve of peak limiter with threshold={Lim.threshold}dB:')
    print(f'\nLinear (Input, Output):\n{get_transfer_curve(Lim, num_points=20)}')
    print(f'\nLogarithmic (Input, Output in dB):\n{get_transfer_curve_db(Lim, num_points=20)}')

    # Plot transfer curves for both compressor and limiter
    plot_transfer_curve([Comp, Lim], ['Compressor', 'Limiter'])