import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import metadata

# === METADATA ===
FPS = metadata.FPS
MIN_CALL_FREQ = metadata.MIN_CALL_FREQ
MAX_CALL_FREQ = metadata.MAX_CALL_FREQ
MAX_FREQ = 1000  # Hz

# === FILES ===
channel_a_path = "data/NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"
channel_b_path = "data/NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch07.csv"

# === LOAD DATA ===
def load_and_parse_spectra(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1).dropna(axis=1, how='all')
    n_freq_bins = df.shape[1]
    freqs = np.linspace(0, MAX_FREQ, n_freq_bins)
    idx_min = np.searchsorted(freqs, MIN_CALL_FREQ)
    idx_max = np.searchsorted(freqs, MAX_CALL_FREQ)
    return df.iloc[:, idx_min:idx_max].to_numpy()

# === FEATURE EXTRACTION ===
def frequency_centroid(signal_2d):
    freqs = np.linspace(MIN_CALL_FREQ, MAX_CALL_FREQ, signal_2d.shape[1])
    return np.sum(signal_2d * freqs, axis=1) / (np.sum(signal_2d, axis=1) + 1e-6)

def generate_boxcars(smoothed, boxcar_width=20, prominence=8.0):
    boxcar = np.zeros_like(smoothed)
    peaks, _ = find_peaks(smoothed, prominence=prominence)
    for p in peaks:
        start = max(p - boxcar_width//2, 0)
        end = min(p + boxcar_width//2, len(smoothed))
        boxcar[start:end] = 1.0
    return boxcar, peaks

# === DOPPLER AMBIGUITY SURFACE ===
def generate_ambiguity_surface_2d(signal_ref, signal_test, doppler_range=np.linspace(0.95, 1.05, 100)):
    surface = []
    t_orig = np.arange(len(signal_test)) / FPS

    for factor in doppler_range:
        t_scaled = np.arange(len(signal_test)) / (FPS * factor)
        interp = interp1d(t_orig, signal_test, bounds_error=False, fill_value=0.0)
        scaled = interp(t_scaled)

        # Pad to same length
        max_len = max(len(signal_ref), len(scaled))
        ref_padded = np.pad(signal_ref, (0, max_len - len(signal_ref)), mode='constant')
        scaled_padded = np.pad(scaled, (0, max_len - len(scaled)), mode='constant')

        corr = np.correlate(scaled_padded, ref_padded, mode='full')
        surface.append(corr)

    surface = np.array(surface)
    lags = np.arange(-max_len + 1, max_len) / FPS
    return doppler_range, lags, surface

# === PLOTTING ===
def plot_ambiguity_heatmap(doppler_range, lags_seconds, surface, title=None):
    plt.figure(figsize=(12, 6))
    extent = [lags_seconds[0], lags_seconds[-1], doppler_range[0], doppler_range[-1]]
    plt.imshow(surface, aspect='auto', extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label='Cross-correlation')
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Doppler Factor')
    plt.title(title or 'Doppler Ambiguity Surface')
    plt.tight_layout()
    plt.show()

# === MAIN ANALYSIS ===
def analyze_whale_signals(file_a, file_b, plot_only_every=3):
    sig_2d_a = load_and_parse_spectra(file_a)
    sig_2d_b = load_and_parse_spectra(file_b)
    signal_len = min(len(sig_2d_a), len(sig_2d_b))

    win_sec = 100
    step_sec = 50
    win_samples = int(win_sec * FPS)
    step_samples = int(step_sec * FPS)

    for i, start in enumerate(range(0, signal_len - win_samples + 1, step_samples)):
        end = start + win_samples
        sig_window_a = sig_2d_a[start:end]
        sig_window_b = sig_2d_b[start:end]

        smoothed_a = frequency_centroid(sig_window_a)
        smoothed_b = frequency_centroid(sig_window_b)

        doppler_range, lags, ambiguity_surface = generate_ambiguity_surface_2d(sig_window_a, sig_window_b)

        max_idx = np.unravel_index(np.argmax(ambiguity_surface), ambiguity_surface.shape)
        best_doppler = doppler_range[max_idx[0]]
        best_lag = lags[max_idx[1]]

        print(f"[{start//FPS}s–{end//FPS}s] Max Corr @ Lag={best_lag:.3f}s | Doppler={best_doppler:.4f}")

        if i % plot_only_every == 0:  # only show every nth window
            plot_ambiguity_heatmap(doppler_range, lags, ambiguity_surface,
                                   title=f"Window {start//FPS}-{end//FPS}s")

# === GO! ===
analyze_whale_signals(channel_a_path, channel_b_path, plot_only_every=4)
