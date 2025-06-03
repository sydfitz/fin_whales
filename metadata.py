#
#  Defines constants for use in signal processing, according to spectrogram data.
#  Parameters can be adjusted as needed.
#
#  Author: Sydney Fitzgerald

WINDOW_SIZE = 100           # Size of window to extract around a call in cc (s)
FPS = 7.8125                # Calculation: 
TOLERANCE = 30              # Duration of window to examine around a call in cc (s)
MAX_FREQ = 1000             # Maximum frequency measured by spectrogram (Hz)

CALL_TIME = 24.35           # Time at which a whale call pattern starts (s)

CALL_LENGTH = 1.5           # Length of whale call (s); assume all calls of relatively similar length
SAMPLING_RATE = 2000        # Sampling rate (Hz)
MIN_CALL_FREQ = 10          # Lower bound of frequency range (Hz)
MAX_CALL_FREQ = 35          # Upper bound of frequency range (Hz)

KERNEL_SIZE = 11            # Kernel size to use in smoothing convolution
KERNEL_TYPE = 'gaussian'    # Type of convolution to implement (must be 'gaussian' or 'box')
