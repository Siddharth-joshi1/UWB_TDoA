# config.py
import numpy as np

C = 3e8  # speed of light (m/s)

AREA_SIZE = 5
HEIGHT = 3 
GRID_RES = 0.2

#PHY parameters
FS = 5e9   # sampling rate (5 GHz)
PULSE_WIDTH = 1e-9
NOISE_STD_PHY = 0.01


# ANCHORS = np.array([
#     [0, 0],
#     [5, 0],
#     [5, 5],
#     [0, 5]
# ])


ANCHORS = np.array([
    [0, 0, 2],
    [5, 0, 2],
    [5, 5, 2],
    [0, 5, 2]
])

NOISE_STD = 0.1e-9
SYNC_ERROR_STD=1e-9
SYNC_ERROR_LIST = [
    0,
    0.5e-9,
    1e-9,
    2e-9,
    5e-9,
    10e-9
]

MC_RUNS = 10   # Monte Carlo runs
MAX_ERROR = 10  # cap error for visualization