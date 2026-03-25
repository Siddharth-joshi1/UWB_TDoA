# anchor_geometry_analysis.py

#####directly eun this file.



import numpy as np
import matplotlib.pyplot as plt

from config import AREA_SIZE, HEIGHT, GRID_RES, MC_RUNS
from geometry import generate_grid
from tdoa import generate_tdoa
from solver import solve_tdoa
from metrics import compute_error


# ==========================
# Anchor Configurations
# ==========================

def get_anchor_sets_2d():

    return {
        "Square": np.array([
            [0,0,2],
            [5,0,2],
            [5,5,2],
            [0,5,2]
        ]),

        "Rectangle": np.array([
            [0,0,2],
            [6,0,2],
            [6,3,2],
            [0,3,2]
        ]),

        "Clustered": np.array([
            [1,1,2],
            [2,1,2],
            [2,2,2],
            [1,2,2]
        ]),

        "Random": np.array([
            [0,0,2],
            [4,1,2],
            [3,5,2],
            [1,4,2]
        ])
    }


def get_anchor_sets_3d():

    return {
        "Square_3D": np.array([
            [0,0,2],
            [5,0,2],
            [5,5,2],
            [0,5,2]
        ]),

        "Elevated": np.array([
            [0,0,3],
            [5,0,1],
            [5,5,3],
            [0,5,1]
        ]),

        "Ceiling": np.array([
            [0,0,3],
            [5,0,3],
            [5,5,3],
            [0,5,3]
        ]),

        "Random_3D": np.array([
            [0,0,2],
            [4,1,3],
            [3,5,1],
            [1,4,2]
        ])
    }


# ==========================
# Simulation
# ==========================

def run_simulation(anchors):

    points = generate_grid(AREA_SIZE, HEIGHT, GRID_RES)

    errors = []

    for p in points:

        point_errors = []

        for _ in range(MC_RUNS):

            tdoa = generate_tdoa(p, anchors,sync_error_std=1e-9)
            est = solve_tdoa(anchors, tdoa)

            err = compute_error(p, est)

            if np.isfinite(err):
                point_errors.append(err)

        if len(point_errors) > 0:
            errors.append(np.mean(point_errors))
        else:
            errors.append(np.nan)

    return points, np.array(errors)


# ==========================
# Plot RMSE Comparison
# ==========================

def compare_anchor_sets(anchor_sets):

    names = []
    rmses = []

    for name, anchors in anchor_sets.items():

        print(f"Running {name}")

        points, errors = run_simulation(anchors)

        rmse = np.sqrt(np.nanmean(errors**2))

        names.append(name)
        rmses.append(rmse)

    plt.figure(figsize=(8,5))
    plt.bar(names, rmses)

    plt.ylabel("RMSE (m)")
    plt.title("Anchor Geometry Comparison")

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    print("2D Geometry Comparison")
    compare_anchor_sets(get_anchor_sets_2d())

    print("3D Geometry Comparison")
    compare_anchor_sets(get_anchor_sets_3d())