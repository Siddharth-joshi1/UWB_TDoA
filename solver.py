# solver.py
# Iterative Gauss-Newton solver for TDOA multilateration.
# More initial guesses + stricter convergence than original.

import numpy as np
from config import C, AREA_SIZE, HEIGHT


def _residual(x, anchors, tdoa):
    d    = np.linalg.norm(anchors - x, axis=1)
    pred = [(d[i] - d[j]) / C
            for i in range(len(d)) for j in range(i + 1, len(d))]
    return np.array(pred) - tdoa


def _jacobian(x, anchors):
    d = np.linalg.norm(anchors - x, axis=1)
    J = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if d[i] < 1e-9 or d[j] < 1e-9:
                return None
            J.append((x - anchors[i]) / d[i] / C - (x - anchors[j]) / d[j] / C)
    return np.array(J)


def solve_tdoa(anchors, tdoa):
    """
    Gauss-Newton TDOA solver with multiple initial guesses.
    Returns estimated position or None if all guesses fail.
    """
    a = AREA_SIZE
    h = HEIGHT

    guesses = [
        np.mean(anchors, axis=0),
        np.array([a * 0.25, a * 0.25, h * 0.5]),
        np.array([a * 0.75, a * 0.25, h * 0.5]),
        np.array([a * 0.25, a * 0.75, h * 0.5]),
        np.array([a * 0.75, a * 0.75, h * 0.5]),
        np.array([a * 0.5,  a * 0.5,  h * 0.5]),
        np.array([a * 0.5,  a * 0.5,  h * 0.3]),
    ]

    best_x, best_res = None, np.inf

    for x0 in guesses:
        x = x0.copy().astype(float)
        prev_r = np.inf

        for _ in range(100):
            r = _residual(x, anchors, tdoa)
            J = _jacobian(x, anchors)

            if J is None or np.any(~np.isfinite(J)):
                break

            try:
                dx, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)
            except Exception:
                break

            # Damping: limit step size to 2 m
            step = np.linalg.norm(dx)
            if step > 2.0:
                dx = dx * 2.0 / step

            x = x + dx

            cur_r = np.linalg.norm(r)
            if cur_r < best_res:
                best_res = cur_r
                best_x   = x.copy()

            if np.linalg.norm(dx) < 1e-8:
                break

            prev_r = cur_r

    # Sanity check: solution must be inside the room (+1 m margin)
    if best_x is not None:
        margin = 1.0
        if (best_x[0] < -margin or best_x[0] > AREA_SIZE + margin or
                best_x[1] < -margin or best_x[1] > AREA_SIZE + margin or
                best_x[2] < -margin or best_x[2] > HEIGHT + margin):
            return None

    return best_x