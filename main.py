# main.py
import numpy as np
from config import AREA_SIZE, HEIGHT, GRID_RES, ANCHORS, MC_RUNS, MAX_ERROR
from geometry import generate_grid
from tdoa import generate_tdoa
from solver import solve_tdoa
from metrics import compute_error
from visualization import plot_heatmap
from visualization import plot_3d_surface,plot_3d_points

points = generate_grid(AREA_SIZE, HEIGHT, GRID_RES)

errors = []

for idx, p in enumerate(points):
    point_errors = []

    for _ in range(MC_RUNS):
        tdoa = generate_tdoa(p, ANCHORS,sync_error_std=1e-9)
        est = solve_tdoa(ANCHORS, tdoa)

        err = compute_error(p, est)

        # ignore failed solves
        if np.isfinite(err):
            point_errors.append(err)

    if len(point_errors) < MC_RUNS * 0.5:
        avg_error = np.nan
    else:
        avg_error = np.mean(point_errors)
   
    errors.append(avg_error)

errors = np.array(errors)

# plot_heatmap(points, errors, title="TDoA Localization Error (Averaged)")
# plot_3d_surface(points, errors, title="TDoA Error Surface")

plot_3d_points(points, errors)
print("done")