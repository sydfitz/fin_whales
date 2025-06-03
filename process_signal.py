#
#  Handles processing of data from .csv to numpy array, as well as
#  smoothing and boxcar representation.
#
#  Author: Sydney Fitzgerald

import numpy as np
import pandas as pd
import metadata
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_and_parse_spectra(file_path, start_time, end_time): #TODO: change to frames ?
    """
    Converts a 2D csv file generated from a spectrogram in Raven Pro to an array.

    Parameters:
    - file_path: string indicating file to parse; CSV is formatted such that rows are time steps,
                 columns are frequencies
    - start_time: index (in frames) for which the signal should start in time
    - end_time: index (in frames) for which the signal should end in time

    Returns:
    - flat_signal: 1D numpy array for the signal restricted to min -> max frequency and truncated to
              data within start_time -> end_time (or entire signal if given indices are out of bounds)

    """
    df = pd.read_csv(file_path, header=None, skiprows=1)    # Read csv into a dataframe
    df = df.dropna(axis=1, how='all')                       # Remove NaN columns

    # Frequency bins (assuming even spacing from 0 to max_freq)
    n_freq_bins = df.shape[1]
    freqs = np.linspace(0, metadata.MAX_FREQ, n_freq_bins)
    
    # Find indices of min & max frequencies & truncate
    idx_min = np.searchsorted(freqs, metadata.MIN_CALL_FREQ, side='left')
    idx_max = np.searchsorted(freqs, metadata.MAX_CALL_FREQ, side='right')

    print(f"Frequency bins: {n_freq_bins}, slicing columns from {idx_min} to {idx_max} corresponding to {metadata.MIN_CALL_FREQ}Hz to {metadata.MAX_CALL_FREQ}Hz \n")
    
    # Keep only columns in desired frequency range
    signal_df = df.iloc[:, idx_min:idx_max]
    signal_array = signal_df.to_numpy()

    # Avoid out of bounds indexing
    if (start_time < 0):
        start_time = 0
    if (end_time > signal_array.shape[0]):
        end_time = signal_array.shape[0]

    # Condense to 1D by averaging across bins, only within given time period
    flat_signal = np.mean(signal_array[start_time:end_time, :], axis=1)
    
    return flat_signal


def smooth_signal(signal_array, kernel_size, kernel_type):
    """
    Smooth a signal using convolution with a selected kernel.

    Parameters:
    - signal: 1D numpy array of spectrogram data
    - kernel_size: number of points in the kernel (should be odd)
    - kernel_type: 'gaussian' or 'box'

    Returns:
    - smoothed: 1D signal with noise reduced
    """

    if kernel_type == 'gaussian':
        x = np.linspace(-1, 1, kernel_size)
        kernel = np.exp(-x**2 * 4)
    elif kernel_type == 'box':
        kernel = np.ones(kernel_size)
    else:
        raise ValueError("kernel_type must be 'gaussian' or 'box'")
    
    kernel = kernel / np.sum(kernel)
    smoothed = np.convolve(signal_array, kernel, mode='same')

    return smoothed


def generate_boxcars(smoothed_signal, boxcar_width, boxcar_height=1.0, prominence=5.0):
    """
    Find peaks on a pre-smoothed signal and place boxcars centered on them.

    Parameters:
    - smoothed_signal (np.array): 1D smoothed signal (no raw noise).
    - boxcar_width (int): Width of each boxcar (in samples).
    - boxcar_height (float): Height (amplitude) of each boxcar.
    - prominence (float): Required prominence of peaks to detect.

    Returns:
    - boxcar_signal (np.array): Signal with boxcars at peak positions.
    - peak_indices (np.array): Indices of detected peaks.
    """
    p = 0.2 * np.max(smoothed_signal)  # Dynamically set prominence
    peak_indices, _ = find_peaks(smoothed_signal, prominence=p)

    boxcar_signal = np.zeros_like(smoothed_signal)
    half_width = boxcar_width // 2

    for peak in peak_indices:
        start = max(peak - half_width, 0)
        end = min(peak + half_width + 1, len(smoothed_signal))
        boxcar_signal[start:end] = boxcar_height

    return boxcar_signal, peak_indices


def plot_boxcar_results(smoothed, boxcars, peaks):
    """
    Plots the result of reduced smoothed signal to boxcars.

    Parameters:
    - 

    """
    frame_indices = np.arange(len(smoothed))
    time = frame_indices / metadata.FPS
    plt.plot(time, smoothed, label="Smoothed Signal")
    plt.plot(time, boxcars, label="Boxcars", color="magenta")
    plt.scatter(time[peaks], smoothed[peaks], color="red", label="Detected Peaks")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.title("Boxcars from Smoothed Signal")
    plt.grid(True)
    plt.show()







