# import cross_correlation as cc
# import matplotlib.pyplot as plt
# import numpy as np

# reference_signal = "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch00.csv"
# aux_signals = ["NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch01.csv", "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch02.csv",
#                "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch03.csv", "NOPP6_EST_20090327_153000.Spectrogram 1.samples-ch04.csv"]
# results = []

# fps = 7.8125

# for signal in aux_signals:
#     target, segment, lags, corr, best_lag_seconds, confidence = cc.run_pairwise_analysis(reference_signal, signal, 195)

#     results.append({
#         "signal": signal,
#         "confidence": confidence,
#         "target": target,
#         "segment": segment,
#         "lags": lags,
#         "corr": corr,
#         "best_lag_seconds": best_lag_seconds
#     })


# top3 = sorted(results, key=lambda x: x['confidence'], reverse=True)[:3]

# # Set up 9-row plot (3 rows per match)
# fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(14, 30), sharex=False)

# for i, result in enumerate(top3):
#     base = i * 2  # offset for each match's 3 rows

#     target = result['target']
#     segment = result['segment']
#     lags = result['lags']
#     corr = result['corr']
#     best_lag_seconds = result['best_lag_seconds']
#     signal_name = result['signal']
#     confidence = result['confidence']

#     # # --- Plot 1: Target ---
#     # axs[base].plot(np.arange(len(target)) / fps, target, color='blue')
#     # axs[base].set_title(f'Match {i+1}: Target Signal (Channel A)')
#     # axs[base].set_ylabel('Amplitude')
#     # axs[base].grid(True)

#     # --- Plot 2: Segment ---
#     axs[base].plot(np.arange(len(segment)) / fps, segment, color='green')
#     axs[base].set_title(f'Match {i+1}: Segment from {signal_name}')
#     axs[base].set_ylabel('Amplitude')
#     axs[base].grid(True)

#     # --- Plot 3: Correlation ---
#     axs[base + 1].plot(lags / fps, corr, color='purple')
#     axs[base + 1].axvline(best_lag_seconds, color='red', linestyle='--', label=f'Best Match @ {best_lag_seconds:.2f}s')
#     axs[base + 1].set_title(f'Match {i+1}: Cross-Correlation (Confidence: {confidence:.2f})')
#     axs[base + 1].set_xlabel('Lag (seconds)')
#     axs[base + 1].set_ylabel('Corr')
#     axs[base + 1].legend()
#     axs[base + 1].grid(True)

# plt.show()