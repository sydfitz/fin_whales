import metadata
import matplotlib.pyplot as plt
import process_signal as ps
import numpy as np
import cross_correlation as cc

# File paths
channel_a_path = "data/NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"    # Known whale call
channel_b_path = "data/NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch01.csv"    # Input signal

# Generate plot of signal
signal1 = ps.load_and_parse_spectra(channel_a_path, 0, 500)
smoothed1 = ps.smooth_signal(signal1, metadata.KERNEL_SIZE, metadata.KERNEL_TYPE)
boxcar_width_samples = int(metadata.CALL_LENGTH * metadata.FPS) 
boxcars, peaks = ps.generate_boxcars(smoothed1, boxcar_width_samples)
ps.plot_boxcar_results(smoothed1, boxcars, peaks)

# signal2 = ps.load_and_parse_spectra(channel_b_path, 0,)

reference, new, lags, corr, best_lag_seconds, confidence = cc.run_pairwise_analysis(channel_a_path, channel_b_path, use_boxcar=True)
cc.plot_functions(reference, new, lags, corr, best_lag_seconds)





