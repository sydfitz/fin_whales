#
#  Contains functions for cross correlating whale call signals
#  and plotting the resulting data.
#
#  Author: Sydney Fitzgerald

import numpy as np
import pandas as pd
import metadata
import process_signal as ps
from scipy.signal import correlate
import matplotlib.pyplot as plt


def run_pairwise_analysis(target_path, segment_path, use_boxcar=False):

    """
    Runs a cross correlation analysis between two signals.

    Parameters:
    - target_path: string indicating path to file containing a known call
    - segment_path: string indicating path to file to be analyzed for the target call
    - use_boxcar: if true, run the cross correlation on the boxcar version of the signal
                  instead of the original signal itself

    Returns:
    - signal:

    """

    # Convert parameters from seconds to frames
    window_frames = int(metadata.WINDOW_SIZE * metadata.FPS)
    tolerance_frames = int(metadata.TOLERANCE * metadata.FPS)
    call_frame = int(metadata.CALL_TIME * metadata.FPS)

    # Define start & end indices for both signals
    start_a = call_frame - window_frames // 2
    end_a = start_a + window_frames

    search_start = call_frame - window_frames // 2 - tolerance_frames
    search_end = call_frame + window_frames // 2 + tolerance_frames

    print(f"Channel A slice: start={start_a}, end={end_a}")
    print(f"Channel B search slice: start={search_start}, end={search_end}")

    target = ps.load_and_parse_spectra(target_path, start_a, end_a)
    segment = ps.load_and_parse_spectra(segment_path, search_start, search_end)

    # Optionally convert signals to boxcars
    if use_boxcar:
        print("Converting to boxcar signals...\n")
        target = ps.smooth_signal(target, metadata.KERNEL_SIZE, metadata.KERNEL_TYPE)
        segment = ps.smooth_signal(segment, metadata.KERNEL_SIZE, metadata.KERNEL_TYPE)
        boxcar_width_samples = int(metadata.CALL_LENGTH * metadata.FPS) 
        target, _ = ps.generate_boxcars(target, boxcar_width_samples)
        segment, _ = ps.generate_boxcars(segment, boxcar_width_samples)

    # Cross-correlation 
    if end_a <= target.shape[0] and search_end <= segment.shape[0]:

        target = (target - np.mean(target)) #/ np.std(target)
        segment = (segment - np.mean(segment)) #/ np.std(segment)

        if target.size > 0 and segment.size >= target.size:
            corr = correlate(segment, target, mode='full')
            lags = np.arange(-len(target) + 1, len(segment))
            best_idx = np.argmax(corr)
            best_lag_frames = lags[best_idx]
            best_lag_seconds = best_lag_frames / metadata.FPS
            match_time_index = search_start + best_lag_frames
            confidence = corr[best_idx]

            print(f"\nBest match at lag = {best_lag_seconds:.2f} sec")
            print(f"Channel B match frame = {match_time_index}")
            print(f"Confidence (correlation peak): {confidence:.2f}")
        else:
            print("Error: target or segment is empty or segment is too short")
    else:
        print("Error: slice indices out of range")

    return target, segment, lags, corr, best_lag_seconds, confidence


def plot_functions(target, segment, lags, corr, best_lag_seconds):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # --- 1. Plot Target Signal ---
    axs[0].plot(np.arange(len(target)) / metadata.FPS, target, color='blue')
    axs[0].set_title('Target Signal (Channel A)')
    axs[0].set_ylabel('Mean Amplitude')
    axs[0].grid(True)

    # --- 2. Plot Segment from Channel B ---
    axs[1].plot(np.arange(len(segment)) / metadata.FPS, segment, color='green')
    axs[1].set_title('Segment Signal (Channel B Search Window)')
    axs[1].set_ylabel('Mean Amplitude')
    axs[1].grid(True)

    # --- 3. Plot Cross-Correlation with Lag ---
    axs[2].plot(lags / metadata.FPS, corr, color='purple')
    axs[2].axvline(best_lag_seconds, color='red', linestyle='--', label='Best Match')
    axs[2].set_title('Cross-Correlation Between Signals')
    axs[2].set_xlabel('Lag (seconds)')
    axs[2].set_ylabel('Correlation Value')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
