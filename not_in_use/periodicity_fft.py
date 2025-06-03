import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cross_correlation as cc
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

signal_path = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"
signal_spectra = np.nanmean(cc.load_and_parse_spectra(signal_path), axis=1)

fps = 7.8125

def produce_fft(signal):
    N = len(signal)
    T = 1 / fps  # seconds per frame

    fft_vals = fft(signal)
    fft_freqs = fftfreq(N, T)

    # Only keep the positive frequencies (real signals are symmetric)
    positive_freqs = fft_freqs[:N//2]
    positive_magnitudes = np.abs(fft_vals[:N//2])

    return positive_freqs, positive_magnitudes

def analyze_peaks(freqs, mags):
    peaks, _ = find_peaks(mags)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mags, label='FFT Magnitude')
    plt.plot(freqs[peaks], mags[peaks], "x", color='red', label='Detected Peaks')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Detected Frequency Peaks in Whale Call Activity')
    plt.xlim(0, 0.2)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fft(freqs, mags):
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mags)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Whale Call Intensity Over Time')
    plt.xlim(0, 0.2)  # Zoom in to low freqs (repetitions)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

freqs, mags = produce_fft(signal_spectra)

# Calculate frequency limit
freq_limit = 1 / 30  # 0.0333 Hz

# Apply mask to filter frequencies BELOW the limit
mask = freqs < freq_limit
truncated_freqs = freqs[mask]
truncated_mags = mags[mask]

plot_fft(truncated_freqs, truncated_mags)


