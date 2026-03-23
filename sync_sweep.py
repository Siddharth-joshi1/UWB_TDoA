# sync_sweep.py
import numpy as np
import matplotlib.pyplot as plt

from config import AREA_SIZE, GRID_RES, ANCHORS, MC_RUNS, SYNC_ERROR_LIST
from geometry import generate_grid
from tdoa import generate_tdoa
from solver import solve_tdoa
from metrics import compute_error

points = generate_grid(AREA_SIZE, GRID_RES)

rmse_list = []

for sync_std in SYNC_ERROR_LIST:
    print(f"Running for sync error = {sync_std*1e9:.2f} ns")

    errors = []

    for p in points:
        point_errors = []

        for _ in range(MC_RUNS):
            tdoa = generate_tdoa(p, ANCHORS, sync_std)
            est = solve_tdoa(ANCHORS, tdoa)

            err = compute_error(p, est)

            if np.isfinite(err):
                point_errors.append(err)

        if len(point_errors) > 0:
            errors.extend(point_errors)

    if len(errors) > 0:
        rmse = np.sqrt(np.mean(np.array(errors)**2))
    else:
        rmse = np.nan

    rmse_list.append(rmse)

# Convert to ns for plotting
sync_ns = [s * 1e9 for s in SYNC_ERROR_LIST]

# Plot
plt.figure(figsize=(6,5))
plt.plot(sync_ns, rmse_list, marker='o')

plt.xlabel("Synchronization Error (ns)")
plt.ylabel("RMSE (m)")
plt.title("TDoA Accuracy vs Synchronization Error")
plt.grid(True)

plt.tight_layout()
plt.show()