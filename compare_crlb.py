# compare_crlb.py
import numpy as np
import matplotlib.pyplot as plt

from config import AREA_SIZE, GRID_RES, ANCHORS, MC_RUNS, NOISE_STD, SYNC_ERROR_STD
from geometry import generate_grid
from tdoa import generate_tdoa
from solver import solve_tdoa
from metrics import compute_error
from crlb import compute_crlb

points = generate_grid(AREA_SIZE, GRID_RES)

sim_errors = []
crlb_vals = []

for p in points:
    # --- Simulation error ---
    point_errors = []

    for _ in range(MC_RUNS):
        tdoa = generate_tdoa(p, ANCHORS, SYNC_ERROR_STD)
        est = solve_tdoa(ANCHORS, tdoa)

        err = compute_error(p, est)

        if np.isfinite(err):
            point_errors.append(err)

    if len(point_errors) > 0:
        sim_errors.append(np.mean(point_errors))
    else:
        sim_errors.append(np.nan)

    # --- CRLB ---
    crlb_vals.append(compute_crlb(p, ANCHORS, NOISE_STD))

sim_errors = np.array(sim_errors)
crlb_vals = np.array(crlb_vals)

# Remove NaNs
mask = ~np.isnan(sim_errors) & ~np.isnan(crlb_vals)

plt.figure(figsize=(6,5))
plt.scatter(crlb_vals[mask], sim_errors[mask], alpha=0.5)

plt.xlabel("CRLB (m)")
plt.ylabel("Simulated Error (m)")
plt.title("Simulation vs CRLB")

plt.grid(True)
plt.tight_layout()
plt.show()