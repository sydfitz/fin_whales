import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt

# === File paths ===
channel_a_path = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"  # Known whale call
channel_b_path = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch01.csv"  # Compare to A

window_seconds = 100   # Duration of window to extract around the call
tolerance = 30         # Seconds to search ± from known call
max_freq = 1000        # Max frequency of the spectrogram
fps = 7.8125           # Frames per second of spectrogram (calculated to match Raven Pro settings & sampling rate)

def run_pairwise_analysis(reference, new, call_time):
    channel_a_path = reference
    channel_b_path = new
    call_time_index = call_time #Time index where a whale call starts in A (in frames; roughly 25s)

    # Load the spectrogram data
    spec_a = load_and_parse_spectra(channel_a_path)
    spec_b = load_and_parse_spectra(channel_b_path)

    print(f"Loaded Channel A: {spec_a.shape}")
    print(f"Loaded Channel B: {spec_b.shape}")

    # === Convert window and tolerance to frame counts ===
    window_frames = int(window_seconds * fps)
    tolerance_frames = int(tolerance * fps)

    # === Define slicing ===
    start_a = max(call_time_index - window_frames // 2, 0)
    end_a = min(start_a + window_frames, spec_a.shape[0])

    search_start = max(call_time_index - window_frames // 2 - tolerance_frames, 0)
    search_end = min(call_time_index + window_frames // 2 + tolerance_frames, spec_b.shape[0])

    print(f"Channel A slice: start={start_a}, end={end_a}")
    print(f"Channel B search slice: start={search_start}, end={search_end}")

    # === Cross-correlation ===
    if end_a <= spec_a.shape[0] and search_end <= spec_b.shape[0]:
        # Average across frequency bins
        target = np.mean(spec_a[start_a:end_a, :], axis=1)
        segment = np.mean(spec_b[search_start:search_end, :], axis=1)

        target = (target - np.mean(target)) / np.std(target)
        segment = (segment - np.mean(segment)) / np.std(segment)
        if target.size > 0 and segment.size >= target.size:
            corr = correlate(segment, target, mode='full')
            lags = np.arange(-len(target) + 1, len(segment))
            best_idx = np.argmax(corr)
            best_lag_frames = lags[best_idx]
            best_lag_seconds = best_lag_frames / fps
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


# === Load and parse Raven Pro spectrogram CSV ===
def load_and_parse_spectra(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    df = df.dropna(axis=1, how='all')  # Remove empty columns

    # Frequency bins (assuming even spacing from 0 to max_freq)
    n_freq_bins = df.shape[1]
    freqs = np.linspace(0, max_freq, n_freq_bins)
    
    # Find indices for 10Hz and 35Hz
    freq_min = 10
    freq_max = 35
    idx_min = np.searchsorted(freqs, freq_min, side='left')
    idx_max = np.searchsorted(freqs, freq_max, side='right')
    
    print(f"Frequency bins: {n_freq_bins}, slicing columns from {idx_min} to {idx_max} corresponding to {freq_min}Hz to {freq_max}Hz \n")
    
    # Keep only columns in desired frequency range
    df_trunc = df.iloc[:, idx_min:idx_max]
    
    return df_trunc.to_numpy()


def plot_functions(target, segment, lags, corr, best_lag_seconds):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    # --- 1. Plot Target Signal ---
    axs[0].plot(np.arange(len(target)) / fps, target, color='blue')
    axs[0].set_title('Target Signal (Channel A)')
    axs[0].set_ylabel('Mean Amplitude')
    axs[0].grid(True)

    # --- 2. Plot Segment from Channel B ---
    axs[1].plot(np.arange(len(segment)) / fps, segment, color='green')
    axs[1].set_title('Segment Signal (Channel B Search Window)')
    axs[1].set_ylabel('Mean Amplitude')
    axs[1].grid(True)

    # --- 3. Plot Cross-Correlation with Lag ---
    axs[2].plot(lags / fps, corr, color='purple')
    axs[2].axvline(best_lag_seconds, color='red', linestyle='--', label='Best Match')
    axs[2].set_title('Cross-Correlation Between Signals')
    axs[2].set_xlabel('Lag (seconds)')
    axs[2].set_ylabel('Correlation Value')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

#run_pairwise_analysis(channel_a_path, channel_b_path, 195, plot=True)
