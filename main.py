import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt

# === CONFIG ===
channel_a_path = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch01.csv"
channel_b_path = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"

call_time_index = 150  # Time index where whale call is known in Channel A
fps = 1.0              # Frames per second of spectrogram (adjust this!)
window_seconds = 180   # 3 minutes = 180 seconds
tolerance = 30         # Search ±30 sec in Channel B

# === LOAD CSV FUNCTION ===
def load_spectra(file_path):
    """
    Loads CSV assuming numerical spectrogram data with no header.
    Returns numpy array.
    """
    # Using pandas for quick, clean load
    df = pd.read_csv(file_path, header=None, skiprows=1)
    return df.to_numpy()

# === LOAD DATA ===
spec_a = load_spectra(channel_a_path)
spec_b = load_spectra(channel_b_path)

print(f"Loaded spec_a shape: {spec_a.shape}")
print(f"Loaded spec_b shape: {spec_b.shape}")

# === PARAMETERS CONVERTED TO FRAMES ===
window_frames = int(window_seconds * fps)
tolerance_frames = int(tolerance * fps)

# === SLICE COMPUTATION ===
start_a = max(call_time_index - window_frames // 2, 0)
end_a = start_a + window_frames

search_start = max(call_time_index - window_frames // 2 - tolerance_frames, 0)
search_end = min(call_time_index + window_frames // 2 + tolerance_frames, spec_b.shape[1])

print(f"Channel A slice: start={start_a}, end={end_a}")
print(f"Channel B search slice: start={search_start}, end={search_end}")

# === CROSS-CORRELATION ===
if end_a <= spec_a.shape[1] and search_end <= spec_b.shape[1]:
    target = np.mean(spec_a[:, start_a:end_a], axis=0)
    segment = np.mean(spec_b[:, search_start:search_end], axis=0)

    if target.size > 0 and segment.size > 0:
        corr = correlate(segment, target, mode='valid')
        lags = np.arange(len(corr)) - (len(segment) - len(target))

        best_idx = np.argmax(corr)
        best_lag_frames = lags[best_idx]
        best_lag_seconds = best_lag_frames / fps
        match_time_index = search_start + best_lag_frames
        confidence = corr[best_idx]

        print(f"📍 Best match at lag = {best_lag_seconds:.2f} sec")
        print(f"🔁 Channel B match index = {match_time_index} (frame)")
        print(f"🌊 Confidence (correlation peak): {confidence:.2f}")

        # === PLOT THE DRAMA ===
        plt.figure(figsize=(10,4))
        plt.plot(np.arange(len(corr)) / fps - tolerance, corr)
        plt.axvline(best_lag_seconds, color='r', linestyle='--', label='Best Match')
        plt.xlabel('Lag (seconds)')
        plt.ylabel('Cross-correlation')
        plt.title('Cross-correlation between Channels')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("❌ Error: target or segment is empty!")
else:
    print("❌ Error: Slice indices out of range!")
