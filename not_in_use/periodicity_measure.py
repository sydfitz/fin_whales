import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cross_correlation as cc
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

signal_path = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"
signal_spectra = cc.load_and_parse_spectra(signal_path)

fps = 7.8125
threshold = np.mean(signal_spectra) + 3 * np.std(signal_spectra)
min_spacing_frames = 30 * fps

def analyze_peaks(signal):
    peaks, _ = find_peaks(signal, height=threshold, distance=min_spacing_frames)
    peak_times = peaks / fps 
    intervals = np.diff(peak_times)  # seconds between consecutive calls

    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval  # coefficient of variation
    print(f"\nCalculated coefficient of variation: {cv}")
    print(f"\nCalculated mean interval: {mean_interval}")

    return peak_times, intervals, mean_interval

def plot_autocorr(peak_times, intervals, mean_interval):
    plt.figure(figsize=(10,4))
    plt.plot(peak_times[1:], intervals, marker='o', label="Inter-call intervals")
    plt.axhline(mean_interval, color='r', linestyle='--', label='Mean interval')
    plt.xlabel("Time (s)")
    plt.ylabel("Time Between Calls (s)")
    plt.title("Whale Call Interval Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

peaks, ints, mean = analyze_peaks(signal_spectra)
plot_autocorr(peaks, ints, mean)


