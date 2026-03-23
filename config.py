# config.py
import numpy as np

C = 3e8  # speed of light (m/s)

AREA_SIZE = 5
GRID_RES = 0.2

ANCHORS = np.array([
    [0, 0],
    [5, 0],
    [5, 5],
    [0, 5]
])

NOISE_STD = 0.1e-9
SYNC_ERROR_STD = 1e-9

MC_RUNS = 100   # Monte Carlo runs
MAX_ERROR = 10  # cap error for visualization